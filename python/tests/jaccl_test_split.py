# Copyright © 2026 Apple Inc.
#
# Distributed-only integration tests for jaccl Group.split(). Run with
# mlx.launch --backend jaccl (or jaccl-ring) on a multi-rank cluster — there
# is no single-process mode for this test, splitting a real comm group only
# makes sense when there is one.
#
# Tests:
#   test_groups            — basic same/different color sub-groups
#   test_split_same_color  — color 0 everywhere -> sub-group identical to parent
#   test_split_two_groups  — color = rank % 2  -> two sub-groups of even/odd ranks
#   test_split_three_groups — color = rank % 3 -> three sub-groups (parent >=3)
#   test_split_with_key    — same color, reversed key -> sub-rank order reversed
#   test_split_size_one    — color = rank      -> every rank gets a size-1 sub-group
#   test_split_send_recv   — TCP-backed sub-group raises on send/recv
#
# Adapted from cbfea8bb's standalone tests/test_jaccl_split.py to the
# unittest.TestCase style used by upstream main's distributed test suite
# (mpi_test_distributed.py, ring_test_distributed.py, etc.).

import mlx.core as mx
import mlx_distributed_tests
import mlx_tests


class TestJacclDistributed(mlx_distributed_tests.MLXDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        _ = mx.distributed.init(strict=True, backend="jaccl")
        cls.atol = 1e-6
        cls.rtol = 1e-4

    def test_groups(self):
        world = mx.distributed.init()
        self.assertTrue(world.size() >= 2, "jaccl tests require >= 2 ranks")
        self.assertTrue(0 <= world.rank() < world.size())

        # Two-color split with even/odd grouping.
        sub = world.split(world.rank() % 2)
        expected_size = (world.size() + 1 - (world.rank() % 2)) // 2
        self.assertEqual(sub.size(), expected_size)
        self.assertEqual(sub.rank(), world.rank() // 2)

    def test_split_same_color(self):
        world = mx.distributed.init()
        sub = world.split(0)
        self.assertEqual(sub.size(), world.size())
        self.assertEqual(sub.rank(), world.rank())

        # all_sum inside the sub-group should equal the parent all_sum
        # because every rank is in it.
        x = mx.ones(10) * (sub.rank() + 1)
        y = mx.distributed.all_sum(x, group=sub)
        mx.eval(y)
        expected = sum(range(1, sub.size() + 1))
        self.assertTrue(mx.all(y == expected).item())

    def test_split_two_groups(self):
        world = mx.distributed.init()
        if world.size() < 2:
            self.skipTest("test_split_two_groups requires >= 2 ranks")
        color = world.rank() % 2
        sub = world.split(color)
        expected_size = (world.size() + 1 - color) // 2
        self.assertEqual(sub.size(), expected_size)

        # all_sum inside sub-group: sum of [1, 2, ..., sub.size()]
        x = mx.ones(10) * (sub.rank() + 1)
        y = mx.distributed.all_sum(x, group=sub)
        mx.eval(y)
        expected = sum(range(1, sub.size() + 1))
        self.assertTrue(mx.all(y == expected).item())

    def test_split_three_groups(self):
        world = mx.distributed.init()
        if world.size() < 3:
            self.skipTest("test_split_three_groups requires >= 3 ranks")
        color = world.rank() % 3
        sub = world.split(color)
        expected_size = (world.size() - color + 2) // 3
        self.assertEqual(sub.size(), expected_size)

        x = mx.ones(10) * (sub.rank() + 1)
        y = mx.distributed.all_sum(x, group=sub)
        mx.eval(y)
        expected = sum(range(1, sub.size() + 1))
        self.assertTrue(mx.all(y == expected).item())

    def test_split_with_key(self):
        world = mx.distributed.init()
        if world.size() < 2:
            self.skipTest("test_split_with_key requires >= 2 ranks")
        # Reverse the rank order via key.
        sub = world.split(0, key=world.size() - 1 - world.rank())
        self.assertEqual(sub.size(), world.size())
        self.assertEqual(sub.rank(), world.size() - 1 - world.rank())

    def test_split_size_one(self):
        world = mx.distributed.init()
        # Every rank has a unique color -> each sub-group has exactly one
        # member (a no-op LocalGroup at the JACCL layer).
        sub = world.split(world.rank())
        self.assertEqual(sub.size(), 1)
        self.assertEqual(sub.rank(), 0)

        # all_sum is an identity copy in a size-1 group.
        x = mx.array([7.0, 8.0, 9.0])
        y = mx.distributed.all_sum(x, group=sub)
        mx.eval(y)
        self.assertTrue(mx.all(y == x).item())

    def test_split_send_recv(self):
        # TCP-backed sub-groups (the product of any JACCL split() when
        # new_size > 1) do not implement send/recv — exercising them should
        # raise. We only check this if at least one same-color pair exists.
        world = mx.distributed.init()
        if world.size() < 2:
            self.skipTest("test_split_send_recv requires >= 2 ranks")
        sub = world.split(0)  # everyone in the same sub-group
        # JACCL sub-groups go through TCPGroup; send/recv must throw.
        # We only attempt send/recv at the sub-rank-0/sub-rank-1 pair.
        if sub.rank() == 0:
            x = mx.array([42.0, 43.0, 44.0])
            with self.assertRaises(RuntimeError):
                # mx.distributed.send returns a new array marked with a Send
                # primitive; we must evaluate THAT (not the input x, which is
                # already materialized) to trigger the actual send op.
                sent = mx.distributed.send(x, dst=1, group=sub)
                mx.eval(sent)
        elif sub.rank() == 1:
            with self.assertRaises(RuntimeError):
                y = mx.distributed.recv_like(mx.zeros(3), src=0, group=sub)
                mx.eval(y)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
