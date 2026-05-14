// Copyright © 2026 Apple Inc.

#pragma once

#include <string>

#include "jaccl/group.h"
#include "jaccl/rdma.h" // for SideChannel

namespace jaccl {

/**
 * Collective communication group implemented over a TCP star topology.
 *
 * Used in two scenarios:
 *  1. As the product type of Group::split() when the parent is RDMA-backed.
 *     Apple's Thunderbolt RDMA driver does not allow opening a second
 *     ibv_context on a device already held by the parent, so sub-groups
 *     must fall back to TCP transport.
 *  2. As a low-bandwidth control plane between processes that don't need
 *     RDMA's bandwidth or latency.
 *
 * Wire layout: rank 0 listens on `coordinator`, all other ranks connect.
 * The internal SideChannel performs all collectives by gathering to rank 0
 * and broadcasting back; this is O(size) socket traffic per call.
 *
 * Implements: all_sum, all_max, all_min (gather + local reduction);
 * all_gather (concatenate per-rank slices); barrier (empty all_gather);
 * split (recursive — sub-group is another TCPGroup or a LocalGroup).
 *
 * Does NOT implement: send / recv. There is no peer-to-peer in the star
 * topology; both throw std::runtime_error.
 */
class TCPGroup : public Group {
 public:
  // `coordinator` is parsed as "host:port"; if no ':' is present the port
  // defaults to 29500. The constructor also runs a SideChannel barrier so
  // that every rank has its TCP connection established before any
  // collective is called.
  TCPGroup(int rank, int size, const std::string& coordinator);

  int rank() override {
    return rank_;
  }
  int size() override {
    return size_;
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
  void barrier() override;

  std::shared_ptr<Group> split(int color, int key) override;

 private:
  template <typename T, typename ReduceOp>
  void tcp_all_reduce(
      const void* input,
      void* output,
      size_t n_bytes,
      ReduceOp reduce_op);

  int rank_;
  int size_;
  SideChannel side_channel_;
  // Original coordinator host:port for sub-group port derivation in split().
  std::string coordinator_host_;
  int coordinator_port_;
};

} // namespace jaccl
