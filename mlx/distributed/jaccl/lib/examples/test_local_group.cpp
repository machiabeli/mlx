// Copyright © 2026 Apple Inc.
//
// Standalone test for jaccl::LocalGroup, the no-op size-1 group used as a
// fallback for Group::split() when a color class has a single member. Runs
// in a single process; no SideChannel needed.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <jaccl/group.h>
#include <jaccl/local.h>

namespace {

// Tiny test harness — prints which check failed and aborts on first failure.
int failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cerr << "FAIL (line " << __LINE__ << "): " #cond << "\n";   \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

} // namespace

int main() {
  jaccl::LocalGroup g;

  // Identity properties
  CHECK(g.rank() == 0);
  CHECK(g.size() == 1);

  // all_sum: identity copy
  float in_f = 3.14f, out_f = 0.0f;
  g.all_sum(&in_f, &out_f, sizeof(float), jaccl::Dtype::Float32);
  CHECK(out_f == 3.14f);

  // all_max: identity copy
  int32_t in_i = -7, out_i = 0;
  g.all_max(&in_i, &out_i, sizeof(int32_t), jaccl::Dtype::Int32);
  CHECK(out_i == -7);

  // all_min: identity copy
  in_i = 12;
  out_i = 0;
  g.all_min(&in_i, &out_i, sizeof(int32_t), jaccl::Dtype::Int32);
  CHECK(out_i == 12);

  // all_gather: identity copy
  int v_in = 42, v_out = 0;
  g.all_gather(&v_in, &v_out, sizeof(int));
  CHECK(v_out == 42);

  // In-place identity (input == output) must not crash
  uint64_t in_place = 0xDEADBEEFCAFEBABEull;
  g.all_sum(&in_place, &in_place, sizeof(in_place), jaccl::Dtype::UInt64);
  CHECK(in_place == 0xDEADBEEFCAFEBABEull);

  // barrier is a no-op (must not throw or block)
  g.barrier();

  // split with color < 0 returns nullptr
  auto null_sub = g.split(-1, 0);
  CHECK(null_sub == nullptr);

  // split with color >= 0 returns another LocalGroup (size 1)
  auto sub = g.split(0, 0);
  CHECK(sub != nullptr);
  CHECK(sub->size() == 1);
  CHECK(sub->rank() == 0);

  // send/recv must throw: a size-1 group has no remote peer
  bool send_threw = false;
  try {
    g.send(&in_f, sizeof(float), 0);
  } catch (const std::runtime_error&) {
    send_threw = true;
  }
  CHECK(send_threw);

  bool recv_threw = false;
  try {
    g.recv(&out_f, sizeof(float), 0);
  } catch (const std::runtime_error&) {
    recv_threw = true;
  }
  CHECK(recv_threw);

  // Recursive split() on the sub-group should also work (size-1 -> size-1)
  auto subsub = sub->split(0, 0);
  CHECK(subsub != nullptr);
  CHECK(subsub->size() == 1);
  CHECK(subsub->rank() == 0);

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "test_local_group: OK\n";
  return 0;
}
