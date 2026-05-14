// Copyright © 2026 Apple Inc.

#pragma once

#include "jaccl/group.h"

namespace jaccl {

/**
 * No-op communication group for the size-1 case. Used as the return value of
 * Group::split() when only this rank ends up in the sub-group.
 *
 * - all_sum / all_max / all_min / all_gather perform an identity copy from
 *   input to output (memcpy, skipped when input == output).
 * - send / recv throw, since a size-1 group has no remote peer.
 * - barrier is a no-op.
 * - split() returns another LocalGroup (no collective handshake required —
 *   a single rank cannot disagree with itself).
 */
class LocalGroup : public Group {
 public:
  int rank() override {
    return 0;
  }
  int size() override {
    return 1;
  }

  void all_sum(const void* input, void* output, size_t n_bytes, int dtype)
      override;
  void all_max(const void* input, void* output, size_t n_bytes, int dtype)
      override;
  void all_min(const void* input, void* output, size_t n_bytes, int dtype)
      override;
  void all_gather(const void* input, void* output, size_t n_bytes) override;
  void send(const void* input, size_t n_bytes, int dst) override;
  void recv(void* output, size_t n_bytes, int src) override;
  void barrier() override {}

  std::shared_ptr<Group> split(int color, int key) override;
};

} // namespace jaccl
