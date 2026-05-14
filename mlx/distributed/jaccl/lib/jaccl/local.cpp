// Copyright © 2026 Apple Inc.

#include "jaccl/local.h"

#include <cstring>
#include <memory>
#include <stdexcept>

namespace jaccl {

namespace {

// Copy when source and destination differ; otherwise no-op. The size-1 group
// is allowed to be called with the same pointer for both input and output
// (matches the common in-place reduction pattern in callers).
inline void identity_copy(const void* in, void* out, size_t n_bytes) {
  if (n_bytes == 0) {
    return;
  }
  if (in != out) {
    std::memcpy(out, in, n_bytes);
  }
}

} // namespace

void LocalGroup::all_sum(
    const void* in,
    void* out,
    size_t n_bytes,
    int /*dtype*/) {
  identity_copy(in, out, n_bytes);
}

void LocalGroup::all_max(
    const void* in,
    void* out,
    size_t n_bytes,
    int /*dtype*/) {
  identity_copy(in, out, n_bytes);
}

void LocalGroup::all_min(
    const void* in,
    void* out,
    size_t n_bytes,
    int /*dtype*/) {
  identity_copy(in, out, n_bytes);
}

void LocalGroup::all_gather(const void* in, void* out, size_t n_bytes) {
  identity_copy(in, out, n_bytes);
}

void LocalGroup::send(const void* /*in*/, size_t /*n_bytes*/, int /*dst*/) {
  throw std::runtime_error("[jaccl] LocalGroup cannot send (size-1 group)");
}

void LocalGroup::recv(void* /*out*/, size_t /*n_bytes*/, int /*src*/) {
  throw std::runtime_error("[jaccl] LocalGroup cannot recv (size-1 group)");
}

std::shared_ptr<Group> LocalGroup::split(int color, int /*key*/) {
  if (color < 0) {
    return nullptr;
  }
  return std::make_shared<LocalGroup>();
}

} // namespace jaccl
