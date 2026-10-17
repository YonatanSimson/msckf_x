"""Tests for python/lamaria_to_euroc.py's frame handling.

cd python && python -m unittest discover tests
"""

import unittest

import numpy as np

from lamaria_to_euroc import drop_frozen_poses, imu_poses, mean_world_specific_force, rigid
from msckf_gates.common import quat_to_R, quat_wxyz

GRAVITY = np.array([0.0, 0.0, 9.81])


def random_xyzw(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def pgt_row(t_ns, T):
    """A pGT line 't tx ty tz qx qy qz qw' of the pose T."""
    w, x, y, z = quat_wxyz(T[:3, :3])
    return np.array([t_ns, *T[:3, 3], x, y, z, w])


def pose(row):
    """4x4 pose of a 't tx ty tz qw qx qy qz' row."""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(row[4:8])
    T[:3, 3] = row[1:4]
    return T


class QuaternionTest(unittest.TestCase):
    def test_matrix_round_trip_keeps_rotation(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            R = rigid(random_xyzw(rng), np.zeros(3))[:3, :3]
            np.testing.assert_allclose(quat_to_R(quat_wxyz(R)), R, atol=1e-12)


class ImuPosesTest(unittest.TestCase):
    def test_imu_rig_reorders_quaternion_to_wxyz(self):
        row = np.array([5, 1.0, 2.0, 3.0, 0.0, 0.0, np.sin(0.3), np.cos(0.3)])

        gt = imu_poses(row[None, :], np.eye(4))

        np.testing.assert_allclose(gt[0], [5, 1.0, 2.0, 3.0, np.cos(0.3), 0.0, 0.0, np.sin(0.3)], atol=1e-12)

    def test_cam0_rig_recovers_world_from_imu(self):
        rng = np.random.default_rng(2)
        T_wi = rigid(random_xyzw(rng), rng.normal(size=3))
        T_ic = rigid(random_xyzw(rng), rng.normal(size=3))
        row = pgt_row(7, T_wi @ T_ic)

        gt = imu_poses(row[None, :], np.linalg.inv(T_ic))

        np.testing.assert_allclose(pose(gt[0]), T_wi, atol=1e-9)


class SpecificForceTest(unittest.TestCase):
    def test_tilted_device_at_rest_gives_world_gravity(self):
        T_wi = rigid([np.sin(0.2), 0.0, 0.0, np.cos(0.2)], np.zeros(3))
        gt = imu_poses(np.stack([pgt_row(0, T_wi), pgt_row(1000, T_wi)]), np.eye(4))
        acc = T_wi[:3, :3].T @ GRAVITY
        imu = np.array([[t, 0, 0, 0, *acc] for t in range(0, 1000, 100)], dtype=np.float64)

        f = mean_world_specific_force(imu, gt)

        np.testing.assert_allclose(f, GRAVITY, atol=1e-9)


class DropFrozenPosesTest(unittest.TestCase):
    def test_drops_leading_run_and_mid_sequence_repeat(self):
        poses = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0], [3, 0, 0]]
        pgt = np.array([[k, *p, 0, 0, 0, 1] for k, p in enumerate(poses)], dtype=float)

        kept = drop_frozen_poses(pgt)

        np.testing.assert_array_equal(kept[:, 0], [3, 6])

    def test_keeps_moving_poses(self):
        pgt = np.array([[k, k, 0, 0, 0, 0, 0, 1] for k in range(5)], dtype=float)

        np.testing.assert_array_equal(drop_frozen_poses(pgt), pgt)


if __name__ == "__main__":
    unittest.main()
