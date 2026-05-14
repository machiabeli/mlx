// Copyright © 2026 Apple Inc.

#pragma once

#include <string>

#include "jaccl/rdma.h" // for SideChannel

namespace jaccl::detail {

/**
 * The local decision a single rank takes from a parent Group::split() call:
 * its new rank in the sub-group, the sub-group size, and the coordinator
 * address each member of the sub-group will connect to.
 *
 * For color < 0 the decision is:
 *   new_rank == -1, new_size == 0, sub_coordinator.empty().
 * The caller (typically the corresponding Group subclass) is responsible for
 * returning nullptr in this case.
 *
 * For a non-negative color whose class has only one member:
 *   new_rank == 0, new_size == 1, sub_coordinator.empty().
 * The caller should return a LocalGroup.
 *
 * Otherwise sub_coordinator holds the "host:port" string each rank in the
 * sub-group should use to construct the new (TCP-backed) group.
 */
struct SplitDecision {
  int new_rank;
  int new_size;
  std::string sub_coordinator;
};

/**
 * Drive the SideChannel handshake required by Group::split() and return this
 * rank's local decision.
 *
 * The function performs three collective operations on `parent_side_channel`,
 * in this order, on every rank of the parent group regardless of `color`:
 *
 *   1. all_gather<{color, key}>             — share split info
 *   2. all_gather<std::string>(my_ip)       — share the IP each rank will
 *                                              advertise as the sub-group
 *                                              coordinator IF it ends up at
 *                                              sub-rank 0 of its class
 *   3. all_gather<int>(0)                   — barrier
 *
 * Negative `color` is honored as MPI_UNDEFINED: the rank still participates in
 * all three collectives (so other ranks observe a consistent SideChannel
 * sequence), and the returned decision is {-1, 0, ""}.
 *
 * `key` < 0 is treated as `parent_rank` (matches MPI_Comm_split semantics
 * and the cbfea8bb behavior).
 *
 * The sub-group port is derived as parent_coord_port + 1000 + color. This
 * spread is large enough that simultaneous splits on neighbouring colors do
 * not collide with the parent port, and the +1000 keeps it clear of common
 * ephemeral ranges. The coordinator host is taken from the
 * `MLX_JACCL_MY_IP` environment variable when set, otherwise from
 * `parent_coord_host`.
 */
SplitDecision compute_split(
    SideChannel& parent_side_channel,
    int parent_rank,
    int parent_size,
    const std::string& parent_coord_host,
    int parent_coord_port,
    int color,
    int key);

} // namespace jaccl::detail
