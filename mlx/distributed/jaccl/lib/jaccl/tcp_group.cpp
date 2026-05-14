// Copyright © 2026 Apple Inc.

#include "jaccl/tcp_group.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "jaccl/local.h"
#include "jaccl/reduction_ops.h"
#include "jaccl/split_impl.h"
#include "jaccl/types.h"

namespace jaccl {

namespace {

// "host:port" -> (host, port). Falls back to (addr, 29500) if no ':' present.
// Uses rfind(':') so IPv6 addresses written as "[::1]:30000" are parsed
// correctly (we don't strip the brackets here — the address is forwarded as
// is to SideChannel which uses parse_address from tcp.h).
inline void parse_host_port(
    const std::string& addr,
    std::string& host,
    int& port) {
  auto colon = addr.rfind(':');
  if (colon == std::string::npos) {
    host = addr;
    port = 29500;
    return;
  }
  host = addr.substr(0, colon);
  port = std::atoi(addr.substr(colon + 1).c_str());
  if (port <= 0) {
    port = 29500;
  }
}

} // namespace

TCPGroup::TCPGroup(int rank, int size, const std::string& coordinator)
    : rank_(rank),
      size_(size),
      side_channel_(rank, size, coordinator.c_str()) {
  parse_host_port(coordinator, coordinator_host_, coordinator_port_);

  // Barrier — make sure all ranks have completed SideChannel construction
  // before any starts issuing collectives. SideChannel's own constructor
  // only synchronizes rank<->rank0; this final all_gather ensures all
  // workers see all peers as connected.
  side_channel_.all_gather<int>(0);
}

void TCPGroup::all_sum(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    tcp_all_reduce<T>(input, output, n_bytes, SumOp<T>{});
  });
}

void TCPGroup::all_max(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    tcp_all_reduce<T>(input, output, n_bytes, MaxOp<T>{});
  });
}

void TCPGroup::all_min(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    tcp_all_reduce<T>(input, output, n_bytes, MinOp<T>{});
  });
}

void TCPGroup::all_gather(const void* input, void* output, size_t n_bytes) {
  auto in = static_cast<const char*>(input);
  std::vector<char> my_data(in, in + n_bytes);
  auto all_data = side_channel_.all_gather(my_data);

  auto out = static_cast<char*>(output);
  for (int i = 0; i < size_; ++i) {
    // SideChannel::all_gather pads to the max length and resizes back to the
    // original sender size. Within a single TCPGroup collective every rank
    // contributes the same n_bytes so all slices match — defensive copy
    // bounded by n_bytes anyway.
    std::memcpy(out + i * n_bytes, all_data[i].data(), n_bytes);
  }
}

void TCPGroup::send(const void* /*input*/, size_t /*n_bytes*/, int /*dst*/) {
  throw std::runtime_error(
      "[jaccl] TCPGroup does not support send (no point-to-point in TCP star)");
}

void TCPGroup::recv(void* /*output*/, size_t /*n_bytes*/, int /*src*/) {
  throw std::runtime_error(
      "[jaccl] TCPGroup does not support recv (no point-to-point in TCP star)");
}

void TCPGroup::barrier() {
  // Two scalar all_gathers — same pattern SideChannel::barrier() uses.
  // Twice has been more robust to startup ordering in practice (see rdma.h).
  side_channel_.all_gather<int>(0);
  side_channel_.all_gather<int>(0);
}

std::shared_ptr<Group> TCPGroup::split(int color, int key) {
  // All three SideChannel collectives inside compute_split MUST run on every
  // rank regardless of color, otherwise the parent SideChannel deadlocks.
  auto decision = detail::compute_split(
      side_channel_,
      rank_,
      size_,
      coordinator_host_,
      coordinator_port_,
      color,
      key);

  if (color < 0) {
    // MPI_UNDEFINED semantics: this rank is not part of any sub-group.
    return nullptr;
  }
  if (decision.new_size <= 1) {
    // Size-1 sub-group: no TCP socket needed; identity copies suffice.
    return std::make_shared<LocalGroup>();
  }
  // Multi-rank sub-group: another TCPGroup. Recursive use is intentional
  // and supported — the parent's SideChannel is independent of the child's.
  return std::make_shared<TCPGroup>(
      decision.new_rank, decision.new_size, decision.sub_coordinator);
}

template <typename T, typename ReduceOp>
void TCPGroup::tcp_all_reduce(
    const void* input,
    void* output,
    size_t n_bytes,
    ReduceOp reduce_op) {
  // Local sanity: the buffer length must be a whole multiple of sizeof(T).
  // Caller passes raw bytes; getting this wrong would silently truncate.
  const size_t count = n_bytes / sizeof(T);
  auto in_chars = static_cast<const char*>(input);
  std::vector<char> my_data(in_chars, in_chars + n_bytes);

  // Each entry of all_data has length n_bytes (same per-rank contribution).
  auto all_data = side_channel_.all_gather(my_data);

  // Seed output with rank 0's slice, then accumulate slices 1..size-1.
  // This includes our own slice when rank_ != 0 (intentional — the sum is
  // over all ranks, not "other" ranks).
  std::memcpy(output, all_data[0].data(), n_bytes);
  auto out_ptr = static_cast<T*>(output);
  for (int i = 1; i < size_; ++i) {
    reduce_op(
        reinterpret_cast<const T*>(all_data[i].data()), out_ptr, count);
  }
}

// Explicit instantiations not required: tcp_all_reduce is only instantiated
// from the in-translation-unit calls in all_sum/all_max/all_min above, and
// the linker keeps them.

} // namespace jaccl
