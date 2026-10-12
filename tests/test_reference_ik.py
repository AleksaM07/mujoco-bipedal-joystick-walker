"""Periodic IK warmup preserves source inputs and endpoint semantics."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from reference_ik import fit_periodic_ik


def make_inputs(
    counts: tuple[int, ...] = (4,), modes: tuple[int, ...] = (1,)
) -> dict[str, object]:
    """Make distinguishable valid frames and intentionally different padding."""
    clips = len(counts)
    frames = max(counts) + 2
    joints = np.arange(clips * frames, dtype=np.float32).reshape(clips, frames, 1)
    return {
        "qpos_targets": joints,
        "ik_target_positions": np.repeat(joints[..., None], 3, axis=-1),
        "ik_target_names": ("foot",),
        "root_pos_targets": np.repeat(joints, 3, axis=-1),
        "root_quat_targets": np.tile([1.0, 0.0, 0.0, 0.0], (clips, frames, 1)),
        "frame_counts": np.array(counts, dtype=np.int32),
        "loop_modes": np.array(modes, dtype=np.int32),
    }


def causal_fit(joints, targets, names, roots, quaternions, counts):
    """Use one previous input node, making the periodic predecessor observable."""
    del targets, names, roots, quaternions, counts
    result = joints.copy()
    result[:, 1:] += 0.1 * joints[:, :-1]
    return result


class ReferenceIKTests(unittest.TestCase):
    """Cover convergence gates independently of the production solver or model."""

    def test_environment_older_versions_bypass_warmup(self) -> None:
        env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
        env._periodic_ik_warmstart = False
        inputs = make_inputs()
        expected = np.zeros_like(inputs["qpos_targets"])
        env._fit_reference_ik_once = Mock(return_value=expected)
        arguments = {key: value for key, value in inputs.items() if key != "loop_modes"}
        with patch("biomechanics_env.fit_periodic_ik") as warmed:
            result = env._retarget_reference_qpos_to_ik_targets(**arguments)
        self.assertIs(result, expected)
        env._fit_reference_ik_once.assert_called_once()
        warmed.assert_not_called()

    def test_environment_v5_forwards_loop_modes_and_retains_diagnostics(self) -> None:
        env = BiomechanicsJoystickEnv.__new__(BiomechanicsJoystickEnv)
        env._periodic_ik_warmstart = True
        inputs = make_inputs()
        env._bvh_reference_loop_modes = inputs["loop_modes"]
        env._fit_reference_ik_once = Mock(side_effect=causal_fit)
        arguments = {key: value for key, value in inputs.items() if key != "loop_modes"}
        result = env._retarget_reference_qpos_to_ik_targets(**arguments)
        self.assertAlmostEqual(float(result[0, 0, 0]), 0.2)
        self.assertEqual(env._reference_ik_warmstart_diagnostics[0]["warmup_cycles"], 4)
        self.assertEqual(env._fit_reference_ik_once.call_count, 1)

    def test_wrap_predecessor_is_last_unique_node_not_endpoint(self) -> None:
        inputs = make_inputs()
        callback = Mock(side_effect=causal_fit)
        fitted, diagnostics = fit_periodic_ik(callback, **inputs)
        self.assertEqual(callback.call_count, 1)
        passed = callback.call_args.args
        np.testing.assert_array_equal(passed[0][0, :, 0], [0, 1, 2] * 4 + [0, 1, 2, 3])
        np.testing.assert_array_equal(passed[5], [16])
        np.testing.assert_allclose(fitted[0, :, 0], [0.2, 1, 2.1, 3.2, 3.2, 3.2])
        self.assertEqual(diagnostics[0]["unique_frame_count"], 3)
        self.assertEqual(diagnostics[0]["warmup_cycles"], 4)
        self.assertEqual(diagnostics[0]["max_cycle_difference_rad"], 0.0)
        np.testing.assert_allclose(diagnostics[0]["initial_previous_pose_rad"], [2.1])

    def test_all_source_arrays_repeat_in_same_order(self) -> None:
        inputs = make_inputs()
        callback = Mock(side_effect=causal_fit)
        fit_periodic_ik(callback, **inputs, max_warmup_cycles=2)
        for argument, name in (
            (0, "qpos_targets"),
            (1, "ik_target_positions"),
            (3, "root_pos_targets"),
            (4, "root_quat_targets"),
        ):
            source = inputs[name]
            expected = np.concatenate(
                [source[:, :3], source[:, :3], source[:, :4]], axis=1
            )
            np.testing.assert_array_equal(callback.call_args.args[argument], expected)
        self.assertEqual(callback.call_args.args[2], ("foot",))

    def test_clamp_and_singleton_do_not_warm(self) -> None:
        for count, mode in ((4, 0), (1, 0), (1, 1)):
            with self.subTest(count=count, mode=mode):
                inputs = make_inputs((count,), (mode,))
                callback = Mock(side_effect=causal_fit)
                fitted, diagnostics = fit_periodic_ik(callback, **inputs)
                passed = callback.call_args.args
                expected = causal_fit(*passed)
                np.testing.assert_array_equal(
                    passed[0], inputs["qpos_targets"][:, :count]
                )
                np.testing.assert_array_equal(fitted[:, :count], expected)
                self.assertEqual(diagnostics[0]["warmup_cycles"], 0)
                self.assertIsNone(diagnostics[0]["max_cycle_difference_rad"])
                self.assertIsNone(diagnostics[0]["initial_previous_pose_rad"])

    def test_two_frame_wrap_uses_single_unique_node(self) -> None:
        inputs = make_inputs((2,), (1,))
        inputs["qpos_targets"][0, :2, 0] = [2.0, 100.0]
        fitted, diagnostics = fit_periodic_ik(causal_fit, **inputs)
        np.testing.assert_allclose(fitted[0, :2, 0], [2.2, 100.2])
        self.assertEqual(diagnostics[0]["unique_frame_count"], 1)

    def test_mixed_clips_use_independent_history_and_hold_padding(self) -> None:
        inputs = make_inputs((4, 2, 1), (1, 0, 1))
        callback = Mock(side_effect=causal_fit)
        fitted, diagnostics = fit_periodic_ik(callback, **inputs)
        self.assertEqual(callback.call_count, 3)
        self.assertEqual(fitted.dtype, np.float32)
        self.assertEqual([row["warmup_cycles"] for row in diagnostics], [4, 0, 0])
        for clip, count in enumerate(inputs["frame_counts"]):
            np.testing.assert_array_equal(
                fitted[clip, count:],
                np.repeat(fitted[clip, count - 1 : count], fitted.shape[1] - count, 0),
            )
        self.assertEqual(fitted[1, 0, 0], inputs["qpos_targets"][1, 0, 0])
        self.assertEqual(fitted[2, 0, 0], inputs["qpos_targets"][2, 0, 0])

    def test_nonconvergent_kernel_fails_explicitly(self) -> None:
        def drifting_fit(joints, *unused):
            return np.arange(joints.shape[1], dtype=float).reshape(joints.shape)

        with self.assertRaisesRegex(ValueError, "did not converge for clip 0"):
            fit_periodic_ik(drifting_fit, **make_inputs())

    def test_tolerance_controls_convergence_not_number_of_valid_frames(self) -> None:
        def slow_fit(joints, *unused):
            return np.arange(joints.shape[1], dtype=float).reshape(joints.shape) * 0.01

        fitted, diagnostics = fit_periodic_ik(slow_fit, **make_inputs(), tolerance=0.04)
        self.assertEqual(fitted.shape, (1, 6, 1))
        self.assertAlmostEqual(diagnostics[0]["max_cycle_difference_rad"], 0.03)

    def test_inputs_immutable_even_when_callback_mutates_its_arguments(self) -> None:
        inputs = make_inputs()
        snapshots = {
            name: value.copy()
            for name, value in inputs.items()
            if isinstance(value, np.ndarray)
        }

        def mutating_fit(joints, targets, names, roots, quaternions, counts):
            for values in (joints, targets, roots, quaternions, counts):
                values[:] = 0
            return joints

        fit_periodic_ik(mutating_fit, **inputs)
        for name, original in snapshots.items():
            np.testing.assert_array_equal(inputs[name], original)

    def test_invalid_settings_rejected(self) -> None:
        for key, invalid in (
            ("max_warmup_cycles", 0),
            ("max_warmup_cycles", -1),
            ("max_warmup_cycles", 1.5),
            ("max_warmup_cycles", True),
            ("tolerance", 0),
            ("tolerance", -1),
            ("tolerance", float("nan")),
            ("tolerance", float("inf")),
            ("tolerance", True),
        ):
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                fit_periodic_ik(causal_fit, **make_inputs(), **{key: invalid})

    def test_invalid_counts_modes_and_shapes_rejected(self) -> None:
        for key, invalid in (
            ("qpos_targets", np.zeros((1, 4))),
            ("qpos_targets", np.zeros((0, 4, 1))),
            ("ik_target_positions", np.zeros((1, 6, 2, 3))),
            ("root_pos_targets", np.zeros((1, 6, 2))),
            ("root_quat_targets", np.zeros((1, 6, 4))),
            ("frame_counts", np.array([0])),
            ("frame_counts", np.array([7])),
            ("frame_counts", np.array([4.0])),
            ("frame_counts", np.array([True])),
            ("frame_counts", np.array([[4]])),
            ("loop_modes", np.array([2])),
            ("loop_modes", np.array([1.0])),
            ("loop_modes", np.array([True])),
            ("ik_target_names", ()),
            ("ik_target_names", ("foot", "foot")),
            ("ik_target_names", ("",)),
        ):
            inputs = make_inputs()
            inputs[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_periodic_ik(causal_fit, **inputs)

    def test_nonfinite_source_arrays_rejected(self) -> None:
        for key in (
            "qpos_targets",
            "ik_target_positions",
            "root_pos_targets",
            "root_quat_targets",
        ):
            inputs = make_inputs()
            inputs[key].flat[0] = np.nan
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "finite"):
                fit_periodic_ik(causal_fit, **inputs)

    def test_invalid_callback_results_rejected(self) -> None:
        for callback in (
            lambda joints, *args: joints[:, :-1],
            lambda joints, *args: np.full(joints.shape, np.nan),
            lambda joints, *args: np.full(joints.shape, 1e100),
            None,
        ):
            with self.subTest(callback=callback), self.assertRaises(ValueError):
                fit_periodic_ik(callback, **make_inputs())


if __name__ == "__main__":
    unittest.main()
