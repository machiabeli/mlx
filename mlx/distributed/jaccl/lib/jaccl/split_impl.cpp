// Copyright © 2026 Apple Inc.

#include "jaccl/split_impl.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace jaccl::detail {

namespace {

// What every rank gathers in step 1. POD so it goes over the wire as bytes.
struct SplitInfo {
  int color;
  int key;
};
static_assert(
    std::is_trivially_copyable_v<SplitInfo>,
    "SplitInfo must be trivially copyable for SideChannel::all_gather<T>");

// What each rank sorts by inside its color class. Sorting by
// (key * parent_size + parent_rank) preserves the cbfea8bb tiebreaker:
// equal `key` falls back to parent rank order; equal `(key, parent_rank)`
// is impossible since parent_rank is unique.
struct Member {
  int sort_key;
  int parent_rank;
};

// Detect the local LAN IPv4 reachable by peers in the cluster.
//
// Each rank in a JACCL parent group lives on a different host, so when a sub-
// group's coordinator binds its listening socket, it MUST bind on a local
// interface — not on the parent group's coordinator IP (which belongs to a
// different host on a different machine).
//
// The original cbfea8bb implementation fell back to `parent_coord_host` if
// MLX_JACCL_MY_IP wasn't set, which only worked when the sub-coordinator
// happened to be the same rank as the parent coordinator. This helper closes
// that gap by auto-discovering the local LAN IP via getifaddrs(2).
//
// Preference order (matches `python/mlx/_distributed_utils/config.py:52-61`
// which uses `ipconfig getifaddr en0 || ipconfig getifaddr en1`):
//   1. en0 (canonical wired/wifi on macOS Mac Studio / MacBook)
//   2. en1 (secondary; some configs route through en1)
//   3. any other en*  (Thunderbolt bridge ports come up as enN — useful
//                      fallback if the host has no Ethernet/Wi-Fi)
//
// Returns empty string on failure, in which case the caller may fall back
// to `parent_coord_host` (which still produces a working sub-group when
// every rank shares that host — i.e. local single-host test setups).
std::string get_local_ipv4() {
  struct ifaddrs* ifap = nullptr;
  if (getifaddrs(&ifap) != 0) {
    return std::string();
  }

  std::string result;
  for (int pass = 0; pass < 3 && result.empty(); ++pass) {
    for (struct ifaddrs* ifa = ifap; ifa; ifa = ifa->ifa_next) {
      if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
      if (!(ifa->ifa_flags & IFF_UP)) continue;
      if (ifa->ifa_flags & IFF_LOOPBACK) continue;
      const char* name = ifa->ifa_name ? ifa->ifa_name : "";
      bool match = false;
      if (pass == 0) {
        match = (std::strcmp(name, "en0") == 0);
      } else if (pass == 1) {
        match = (std::strcmp(name, "en1") == 0);
      } else { // pass == 2: any other enN
        match = (std::strncmp(name, "en", 2) == 0);
      }
      if (!match) continue;
      char buf[INET_ADDRSTRLEN] = {0};
      auto* addr = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
      if (inet_ntop(AF_INET, &addr->sin_addr, buf, sizeof(buf))) {
        result = buf;
        break;
      }
    }
  }

  freeifaddrs(ifap);
  return result;
}

} // namespace

SplitDecision compute_split(
    SideChannel& parent_side_channel,
    int parent_rank,
    int parent_size,
    const std::string& parent_coord_host,
    int parent_coord_port,
    int color,
    int key) {
  // Per MPI_Comm_split: key < 0 defaults to the parent rank order.
  if (key < 0) {
    key = parent_rank;
  }

  // ---- Step 1: all_gather of (color, key) ------------------------------
  // Every rank participates regardless of color, so the SideChannel
  // protocol stays in lockstep.
  SplitInfo mine{color, key};
  std::vector<SplitInfo> all_info = parent_side_channel.all_gather(mine);

  // Identify sub-group members.
  std::vector<Member> members;
  if (color >= 0) {
    members.reserve(parent_size);
    for (int i = 0; i < parent_size; ++i) {
      if (all_info[i].color == color) {
        members.push_back({all_info[i].key * parent_size + i, i});
      }
    }
    std::sort(
        members.begin(), members.end(), [](const Member& a, const Member& b) {
          return a.sort_key < b.sort_key;
        });
  }

  // ---- Step 2: all_gather of advertised IPs ----------------------------
  // Every rank must participate, even those with color < 0 — otherwise
  // SideChannel deadlocks waiting on this rank's payload.
  //
  // Each rank reports its OWN local LAN IP, not the parent coordinator's IP.
  // Discovery order: env override → getifaddrs(en0/en1/en*) → parent
  // coordinator (only correct for single-host test setups).
  const char* env_ip = std::getenv("MLX_JACCL_MY_IP");
  std::string my_ip;
  if (env_ip) {
    my_ip = env_ip;
  } else {
    my_ip = get_local_ipv4();
    if (my_ip.empty()) {
      my_ip = parent_coord_host;
    }
  }
  std::vector<std::string> all_ips =
      parent_side_channel.all_gather(my_ip);

  // ---- Step 3: barrier -------------------------------------------------
  // Final synchronization before any rank starts opening sub-group
  // sockets. Same pattern as TCPGroup::barrier() (single all_gather<int>(0)
  // is sufficient here because the prior two collectives already gave us
  // two rounds of synchronization).
  parent_side_channel.all_gather<int>(0);

  // ---- Local decision (no collectives below this line) -----------------
  SplitDecision out{-1, 0, std::string()};

  if (color < 0) {
    return out;
  }

  out.new_size = static_cast<int>(members.size());
  // members always contains at least this rank when color >= 0, since we
  // put our own (color, key) into all_info[parent_rank].
  for (int i = 0; i < out.new_size; ++i) {
    if (members[i].parent_rank == parent_rank) {
      out.new_rank = i;
      break;
    }
  }

  if (out.new_size <= 1) {
    // Size-1 sub-group: caller will substitute a LocalGroup. No coordinator
    // needed.
    return out;
  }

  // Sub-group coordinator: the rank that became sub-rank 0 advertises its IP,
  // and a per-color port offset keeps simultaneous splits from colliding.
  int sub_rank0_parent = members[0].parent_rank;
  const std::string& host = all_ips[sub_rank0_parent];
  int sub_port = parent_coord_port + 1000 + color;
  out.sub_coordinator = host + ":" + std::to_string(sub_port);
  return out;
}

} // namespace jaccl::detail
