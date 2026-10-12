"""Keep historical BVH mapping reproducible and validate strict foot aliases."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import (
    BVH_POSITION_SCALE,
    ParsedBvh,
    ParsedJoint,
    _bvh_ik_target_positions,
    _resolve_bvh_ik_joint_names,
    load_bvh_references,
)


def make_source_bvh(foot_offsets: dict[str, tuple[float, float, float]]) -> ParsedBvh:
    """Build a three-frame source with root translation and nontrivial yaw."""
    root_channels = (
        "Xposition",
        "Yposition",
        "Zposition",
        "Yrotation",
    )
    joints = {
        "Hips": ParsedJoint(
            name="Hips",
            parent=None,
            children=tuple(foot_offsets),
            offset=np.zeros(3, dtype=np.float32),
            channels=root_channels,
            channel_indices=tuple(range(4)),
        )
    }
    for name, offset in foot_offsets.items():
        joints[name] = ParsedJoint(
            name=name,
            parent="Hips",
            children=(),
            offset=np.array(offset, dtype=np.float32),
            channels=(),
            channel_indices=(),
        )
    return ParsedBvh(
        joints=joints,
        channels=tuple(("Hips", channel) for channel in root_channels),
        motion=np.array(
            [[4, 20, 6, 0], [5, 21, 7, 90], [6, 22, 8, -90]], dtype=np.float32
        ),
        frames=3,
        frame_time=0.01,
        root_name="Hips",
    )


def expected_world_targets(
    bvh: ParsedBvh,
    names: tuple[str, ...],
    alignment: np.ndarray,
    source_origin: np.ndarray,
    target_origin: np.ndarray,
) -> np.ndarray:
    """Compute independent FK for this fixture, without production FK helpers."""
    targets = []
    for frame in bvh.motion:
        yaw = np.deg2rad(float(frame[3]))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
        frame_targets = []
        for name in names:
            position = frame[:3] + rotation @ bvh.joints[name].offset
            converted = position[[2, 0, 1]] * BVH_POSITION_SCALE
            frame_targets.append(
                alignment @ (converted - source_origin) + target_origin
            )
        targets.append(frame_targets)
    return np.asarray(targets)


class BvhIkTargetMappingTests(unittest.TestCase):
    """Assert named source joints survive ordering, frame changes, and aliases."""

    def setUp(self) -> None:
        self.alignment = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        self.source_origin = np.array([6, 4, 20], dtype=np.float32) * BVH_POSITION_SCALE
        self.target_origin = np.array([-0.4, 0.8, 1.5], dtype=np.float32)
        self.offsets = {"RightToe": (-2, -10, 1), "LeftToe": (2, -10, 1)}

    def targets(self, source: ParsedBvh, *, strict: bool) -> np.ndarray:
        return _bvh_ik_target_positions(
            source,
            self.alignment,
            self.source_origin,
            self.target_origin,
            strict_ik_targets=strict,
        )

    def test_toe_aliases_match_independent_fk_and_world_alignment(self) -> None:
        source = make_source_bvh(self.offsets)
        names = _resolve_bvh_ik_joint_names(source)
        self.assertEqual(names, ("RightToe", "LeftToe"))
        actual = self.targets(source, strict=True)
        expected = expected_world_targets(
            source, names, self.alignment, self.source_origin, self.target_origin
        )
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        self.assertEqual(actual.shape, (3, 2, 3))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(
            np.linalg.norm(actual[:, 0] - actual[:, 1], axis=1),
            4 * BVH_POSITION_SCALE,
            atol=1e-7,
        )

    def test_canonical_names_are_accepted(self) -> None:
        source = make_source_bvh(
            {"RightToeBase": (-2, -10, 1), "LeftToeBase": (2, -10, 1)}
        )
        names = _resolve_bvh_ik_joint_names(source)
        self.assertEqual(names, ("RightToeBase", "LeftToeBase"))
        np.testing.assert_array_equal(
            self.targets(source, strict=True), self.targets(source, strict=False)
        )

    def test_canonical_names_take_precedence_over_aliases(self) -> None:
        source = make_source_bvh(
            {
                **self.offsets,
                "RightToeBase": (-3, -9, 2),
                "LeftToeBase": (3, -9, 2),
            }
        )
        names = _resolve_bvh_ik_joint_names(source)
        self.assertEqual(names, ("RightToeBase", "LeftToeBase"))
        np.testing.assert_allclose(
            self.targets(source, strict=True),
            expected_world_targets(
                source, names, self.alignment, self.source_origin, self.target_origin
            ),
            atol=2e-7,
        )

    def test_mixed_canonical_and_alias_names_preserve_right_left_order(self) -> None:
        for offsets, expected_names in (
            (
                {"LeftToe": (2, -10, 1), "RightToeBase": (-2, -10, 1)},
                ("RightToeBase", "LeftToe"),
            ),
            (
                {"LeftToeBase": (2, -10, 1), "RightToe": (-2, -10, 1)},
                ("RightToe", "LeftToeBase"),
            ),
        ):
            with self.subTest(names=expected_names):
                self.assertEqual(
                    _resolve_bvh_ik_joint_names(make_source_bvh(offsets)),
                    expected_names,
                )

    def test_missing_one_or_both_required_feet_fail_closed(self) -> None:
        for offsets in ({"RightToe": (-2, -10, 1)}, {"LeftToe": (2, -10, 1)}, {}):
            with self.subTest(names=tuple(offsets)):
                source = make_source_bvh(offsets)
                with self.assertRaises(ValueError):
                    _resolve_bvh_ik_joint_names(source)
                with self.assertRaises(ValueError):
                    self.targets(source, strict=True)

    def test_historical_alias_only_source_retains_hips_fallback(self) -> None:
        source = make_source_bvh(self.offsets)
        actual = self.targets(source, strict=False)
        expected = expected_world_targets(
            source,
            ("Hips", "Hips"),
            self.alignment,
            self.source_origin,
            self.target_origin,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
        default = _bvh_ik_target_positions(
            source, self.alignment, self.source_origin, self.target_origin
        )
        np.testing.assert_array_equal(actual, default)

    def test_historical_missing_left_only_retains_target_specific_fallback(
        self,
    ) -> None:
        source = make_source_bvh({"RightToeBase": (-2, -10, 1)})
        expected = expected_world_targets(
            source,
            ("RightToeBase", "Hips"),
            self.alignment,
            self.source_origin,
            self.target_origin,
        )
        np.testing.assert_allclose(
            self.targets(source, strict=False), expected, atol=2e-7
        )

    def test_source_motion_and_frame_transforms_are_not_mutated(self) -> None:
        source = make_source_bvh(self.offsets)
        arrays = (source.motion, self.alignment, self.source_origin, self.target_origin)
        originals = [array.copy() for array in arrays]
        self.targets(source, strict=True)
        for actual, original in zip(arrays, originals, strict=True):
            np.testing.assert_array_equal(actual, original)

    def test_public_loader_forwards_strict_choice(self) -> None:
        for strict in (False, True):
            with self.subTest(strict=strict):
                with patch("bvh_reference.BvhMotionLibrary.from_bvh_paths") as load:
                    expected = load.return_value.to_reference_batch.return_value
                    actual = load_bvh_references(
                        ("not-read.bvh",),
                        ("left_hip_x",),
                        np.zeros(1),
                        -np.ones(1),
                        np.ones(1),
                        strict_ik_targets=strict,
                    )
                    self.assertIs(actual, expected)
                    self.assertIs(load.call_args.kwargs["strict_ik_targets"], strict)


class StrictHostMarkerResolutionTests(unittest.TestCase):
    """Prevent missing XML markers from shifting the source target indices."""

    def test_missing_or_reordered_markers_fail_before_any_ik_solve(self) -> None:
        names = ("metatarsal_midpoint_right", "metatarsal_midpoint_left")
        for resolved_names in ((), names[:1], names[1:], names[::-1]):
            with self.subTest(resolved_names=resolved_names):
                env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
                env._strict_bvh_ik_targets = True
                env._resolve_host_ik_markers = Mock(
                    return_value=[{"name": name} for name in resolved_names]
                )
                with self.assertRaises(ValueError):
                    env._retarget_reference_qpos_to_ik_targets(
                        np.zeros((1, 2, 1), dtype=np.float32),
                        np.zeros((1, 2, 2, 3), dtype=np.float32),
                        names,
                        np.zeros((1, 2, 3), dtype=np.float32),
                        np.tile([1.0, 0.0, 0.0, 0.0], (1, 2, 1)),
                        np.array([2]),
                    )


if __name__ == "__main__":
    unittest.main()
