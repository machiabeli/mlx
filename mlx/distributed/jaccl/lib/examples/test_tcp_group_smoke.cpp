// Copyright © 2026 Apple Inc.
//
// Smoke test for jaccl::TCPGroup. Forks one child process so we have two
// real OS-level ranks talking over TCP on localhost. Rank 0 listens on
// 127.0.0.1:30000, rank 1 connects. Each rank fills a small float buffer
// and calls all_sum; we then verify both ranks see the correct sum.
//
// Why fork instead of threads: SideChannel installs file-descriptor state
// on rank 0 (listen socket, accept loop) that isn't thread-safe without
// reworking the API. fork() gives us cheap, faithful two-process IPC.
//
// This file compiles after Phase A; it only links after Phase B2 introduces
// TCPGroup. CMake intentionally keeps it in the build set so the linker
// failure signals "TCPGroup missing" rather than a silent skip.

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <jaccl/group.h>
#include <jaccl/tcp_group.h>

namespace {

constexpr int kSize = 2;
constexpr const char* kCoordinator = "127.0.0.1:30000";
constexpr int kElements = 16;

// Stagger rank 1's startup so rank 0 has time to listen before connect.
void wait_for_listener_if_child(int rank) {
  if (rank == 1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

int run_rank(int rank) {
  wait_for_listener_if_child(rank);

  jaccl::TCPGroup g(rank, kSize, kCoordinator);
  if (g.rank() != rank || g.size() != kSize) {
    std::cerr << "rank " << rank << ": bad rank/size after construct\n";
    return 1;
  }

  // Each rank fills with (rank+1) * 1.0..16.0 so sum is (1+2)*(1..16) = 3,6,9,...
  std::array<float, kElements> in{};
  std::array<float, kElements> out{};
  for (int i = 0; i < kElements; ++i) {
    in[i] = static_cast<float>((rank + 1) * (i + 1));
  }
  g.all_sum(in.data(), out.data(), sizeof(in), jaccl::Dtype::Float32);

  for (int i = 0; i < kElements; ++i) {
    float expected = static_cast<float>(3 * (i + 1)); // 1*(i+1) + 2*(i+1)
    if (out[i] != expected) {
      std::cerr << "rank " << rank << ": out[" << i << "] = " << out[i]
                << ", expected " << expected << "\n";
      return 1;
    }
  }

  // all_gather: each rank contributes a 4-byte int that equals its rank.
  int32_t my = rank;
  std::array<int32_t, kSize> gathered{};
  g.all_gather(&my, gathered.data(), sizeof(int32_t));
  for (int i = 0; i < kSize; ++i) {
    if (gathered[i] != i) {
      std::cerr << "rank " << rank << ": all_gather[" << i
                << "] = " << gathered[i] << ", expected " << i << "\n";
      return 1;
    }
  }

  // barrier must not throw or hang
  g.barrier();

  // send/recv must throw (TCP star has no point-to-point)
  bool send_threw = false;
  try {
    g.send(in.data(), sizeof(float), (rank + 1) % kSize);
  } catch (const std::runtime_error&) {
    send_threw = true;
  }
  if (!send_threw) {
    std::cerr << "rank " << rank << ": send did not throw\n";
    return 1;
  }

  std::cout << "rank " << rank << ": TCPGroup smoke OK\n";
  return 0;
}

} // namespace

int main() {
  pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "fork() failed\n";
    return 1;
  }

  if (pid == 0) {
    // Child = rank 1
    int rc = run_rank(1);
    std::_Exit(rc);
  }

  // Parent = rank 0
  int parent_rc = run_rank(0);

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::cerr << "waitpid failed\n";
    return 1;
  }
  int child_rc = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

  if (parent_rc != 0 || child_rc != 0) {
    std::cerr << "test_tcp_group_smoke FAILED (parent=" << parent_rc
              << ", child=" << child_rc << ")\n";
    return 1;
  }
  std::cout << "test_tcp_group_smoke: OK\n";
  return 0;
}
