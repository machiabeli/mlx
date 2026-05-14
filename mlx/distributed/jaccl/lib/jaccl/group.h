// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>
#include <memory>

namespace jaccl {

/**
 * Abstract base class for a JACCL communication group.
 */
class Group {
 public:
  virtual ~Group() {}

  virtual int rank() = 0;
  virtual int size() = 0;

  virtual void
  all_sum(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_max(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_min(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void all_gather(const void* input, void* output, size_t n_bytes) = 0;

  virtual void send(const void* input, size_t n_bytes, int dst) = 0;
  virtual void recv(void* output, size_t n_bytes, int src) = 0;
  virtual void barrier() = 0;

  /**
   * Whether this group supports point-to-point send/recv.
   *
   * MLX-side wrappers that schedule send/recv on a worker thread (see
   * mlx/distributed/jaccl/jaccl.cpp) MUST check this on the main thread
   * before dispatching. Exceptions thrown inside the worker-thread lambda
   * escape past nanobind's bridge and crash the process via libc++abi.
   *
   * Defaults to true (MeshGroup / RingGroup); TCPGroup overrides to false
   * because TCP star topology has no peer-to-peer transport.
   */
  virtual bool supports_send_recv() const {
    return true;
  }

  /**
   * Split this group into sub-groups based on color and key (MPI_Comm_split
   * semantics). All ranks in this group must call split() collectively in the
   * same order.
   *
   *   color: ranks with the same color end up in the same sub-group.
   *          A negative color removes this rank from any sub-group; the
   *          returned Group is nullptr. The rank still participates in the
   *          parent SideChannel collectives required by split() so other
   *          ranks remain synchronized.
   *   key:   tiebreaker for sub-group rank ordering. Sub-group ranks are
   *          ordered by (key, parent_rank). A negative key defaults to
   *          parent_rank.
   *
   * Returns: a shared_ptr to a Group representing this rank's sub-group,
   * or nullptr if color < 0.
   *
   * Implementation note: for RDMA-backed parent groups (MeshGroup,
   * RingGroup), sub-groups always use TCP transport because Apple's
   * Thunderbolt RDMA driver does not support multiple ibv_context
   * instances on the same physical device. Size-1 sub-groups return
   * a no-op LocalGroup.
   */
  virtual std::shared_ptr<Group> split(int color, int key = -1) = 0;
};

/**
 * Type IDs for dispatch in the standalone JACCL library.
 *
 * Users pass one of these to all_sum/all_max/all_min so JACCL knows how to
 * interpret the data for typed reduction operations.
 */
enum Dtype {
  Bool = 0,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float16,
  BFloat16,
  Float32,
  Float64,
  Complex64,
};

} // namespace jaccl
