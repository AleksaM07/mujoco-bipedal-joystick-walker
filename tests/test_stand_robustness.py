"""Check push timing/units and acceptance without expensive humanoid training."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from config import EnvConfig
from skills.stand_metrics import StandThresholds
from tools.evaluate_stand import collect_control_sample, summarize_rollout
from tools.evaluate_stand_robustness import (
    PROBES,
    StandProbe,
    force_schedule,
    main,
    recovery_seconds,
    summarize_probe,
)


def sample(**changes: float) -> dict[str, float]:
    """Build one eligible, finite physical sample."""
    return {
        "finite": 1.0,
        "physical_failure": 0.0,
        "done": 0.0,
        "success": 1.0,
        "dwell_steps": 50.0,
        "distance": 0.02,
        "speed": 0.01,
        "torso_up": 0.99,
        "height": 1.49,
        "height_ratio": 0.99,
        "eligible": 1.0,
        "reward": 3.0,
        **changes,
    }


class StandRobustnessTests(unittest.TestCase):
    def test_control_trace_reads_correct_actuator_and_joint_indices_under_jit(
        self,
    ) -> None:
        env = SimpleNamespace(
            _actuator_joint_names=("left_ankle_z", "right_ankle_z"),
            _actuator_qpos_indices=jp.array([3, 1]),
        )

        @jax.jit
        def collect(action, smoothed, control, qpos, force):
            state = SimpleNamespace(
                info={"last_action": smoothed},
                data=SimpleNamespace(ctrl=control, qpos=qpos, actuator_force=force),
            )
            return collect_control_sample(env, state, action)

        sample = collect(
            jp.array([0.9, -0.8]),
            jp.array([0.7, -0.6]),
            jp.array([0.13, -0.02]),
            jp.array([1.0, -0.1, 2.0, 0.2]),
            jp.array([12.0, -13.0]),
        )
        expected = {
            "action_left_ankle_z": 0.9,
            "action_right_ankle_z": -0.8,
            "smoothed_action_left_ankle_z": 0.7,
            "smoothed_action_right_ankle_z": -0.6,
            "target_left_ankle_z": 0.13,
            "target_right_ankle_z": -0.02,
            "position_left_ankle_z": 0.2,
            "position_right_ankle_z": -0.1,
            "force_left_ankle_z": 12.0,
            "force_right_ankle_z": -13.0,
        }
        self.assertEqual(set(sample), set(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(float(sample[name]), value, places=6)

    def test_successful_subset_is_not_reported_as_full_robustness(self) -> None:
        for subset in (False, True):
            with (
                self.subTest(subset=subset),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = root / "runs/result"
                env = SimpleNamespace(
                    xml_path="unused",
                    dt=0.02,
                    thresholds=StandThresholds(),
                    reset=lambda key: key,
                    action_size=26,
                )
                arguments = [
                    "evaluate",
                    "--checkpoints",
                    str(root / "000000409600"),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                ]
                if subset:
                    arguments += ["--probes", "push_x_neg"]
                with (
                    patch("sys.argv", arguments),
                    patch("tools.evaluate_stand_robustness.PROJECT_ROOT", root),
                    patch(
                        "tools.evaluate_stand_robustness.load_stand_config",
                        return_value=EnvConfig(locomotion_skill="stand"),
                    ),
                    patch(
                        "tools.evaluate_stand_robustness.read_checkpoint_network_config",
                        return_value={},
                    ),
                    patch(
                        "tools.evaluate_stand_robustness.make_environment",
                        return_value=env,
                    ),
                    patch(
                        "tools.evaluate_stand_robustness.file_sha256",
                        return_value="hash",
                    ),
                    patch("tools.evaluate_stand_robustness.make_rollout_step"),
                    patch("tools.evaluate_stand_robustness.validate_xml_manifest"),
                    patch(
                        "tools.evaluate_stand_robustness.validate_action_compatibility"
                    ),
                    patch("tools.evaluate_stand_robustness.brax_checkpoint.load"),
                    patch(
                        "tools.evaluate_stand_robustness.run_probe",
                        return_value={"passed": True},
                    ),
                ):
                    exit_code = main()
                report = json.loads(
                    (output / "report.json").read_text(encoding="utf-8")
                )
                self.assertTrue(report["requested_cases_passed"])
                self.assertEqual(report["passed"], not subset)
                self.assertEqual(report["full_suite"], not subset)
                self.assertEqual(len(report["results"]), 1 if subset else 5)
                self.assertEqual(exit_code, 1 if subset else 0)

    def test_push_has_exact_half_open_interval_direction_and_impulse(self) -> None:
        for probe in PROBES[1:]:
            with self.subTest(probe=probe.name):
                forces = force_schedule(probe, mass=75, gravity=9.81, dt=0.02)
                active = np.flatnonzero(np.any(forces != 0, axis=1))
                np.testing.assert_array_equal(active, np.arange(250, 260))
                np.testing.assert_allclose(
                    forces[250], np.asarray(probe.direction) * 36.7875, rtol=1e-6
                )
                np.testing.assert_allclose(
                    forces.sum(axis=0) * 0.02,
                    np.asarray(probe.direction) * 7.3575,
                    rtol=1e-6,
                )
                self.assertFalse(np.any(forces[260:]))
        self.assertFalse(
            np.any(force_schedule(PROBES[0], mass=75, gravity=9.81, dt=0.02))
        )

    def test_cartesian_force_at_com_has_expected_native_mujoco_impulse(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 0" timestep="0.005"/><worldbody><body name="test"><freejoint/><geom type="sphere" size="0.1" mass="75"/></body></worldbody></mujoco>'
        )
        for probe in PROBES[1:]:
            with self.subTest(probe=probe.name):
                data = mujoco.MjData(model)
                forces = force_schedule(probe, mass=75, gravity=9.81, dt=0.02)
                for force in forces[250:261]:
                    data.xfrc_applied[:] = 0
                    data.xfrc_applied[1, :3] = force
                    mujoco.mj_step(model, data, nstep=4)
                np.testing.assert_allclose(
                    data.qvel[:3], np.asarray(probe.direction) * 0.0981, atol=1e-7
                )
                np.testing.assert_allclose(data.qvel[3:], 0, atol=1e-7)
                self.assertFalse(np.any(data.xfrc_applied))

    def test_recovery_needs_new_dwell_after_release_and_resets_on_break(self) -> None:
        rows = [sample()] * 100 + [sample(eligible=0)] * 10 + [sample()] * 50
        self.assertAlmostEqual(
            recovery_seconds(rows, release_step=100, dt=0.02, dwell_steps=50), 1.2
        )
        self.assertIsNone(
            recovery_seconds(rows[:-1], release_step=100, dt=0.02, dwell_steps=50)
        )
        rows = [sample()] * 49 + [sample(eligible=0)] + [sample()] * 50
        self.assertAlmostEqual(
            recovery_seconds(rows, release_step=0, dt=0.02, dwell_steps=50), 2.0
        )
        rows[10] = sample(physical_failure=1)
        self.assertIsNone(
            recovery_seconds(rows, release_step=0, dt=0.02, dwell_steps=50)
        )

    def test_long_rollout_cannot_pass_with_only_five_hundred_steps(self) -> None:
        rows = [sample()] * 500
        self.assertTrue(
            summarize_rollout(rows, seed=11, dt=0.02, dwell_steps=50)["passed"]
        )
        self.assertFalse(
            summarize_probe(rows, PROBES[0], dt=0.02, dwell_steps=50)["passed"]
        )
        rows = [sample()] * 3000
        self.assertTrue(
            summarize_probe(rows, PROBES[0], dt=0.02, dwell_steps=50)["passed"]
        )
        rows[1000] = sample(eligible=0)
        self.assertFalse(
            summarize_probe(rows, PROBES[0], dt=0.02, dwell_steps=50)["passed"]
        )
        with self.assertRaises(ValueError):
            summarize_rollout(
                rows, seed=11, dt=0.02, dwell_steps=50, requested_steps=499
            )

    def test_late_recovery_or_final_failure_does_not_pass_push(self) -> None:
        rows = [sample()] * 1000
        result = summarize_probe(rows, PROBES[1], dt=0.02, dwell_steps=50)
        self.assertTrue(result["passed"])
        self.assertEqual(result["recovery_seconds"], 1.0)
        rows[260:461] = [sample(eligible=0)] * 201
        self.assertFalse(
            summarize_probe(rows, PROBES[1], dt=0.02, dwell_steps=50)["passed"]
        )
        rows = [sample()] * 999 + [sample(success=0, dwell_steps=0)]
        self.assertFalse(
            summarize_probe(rows, PROBES[1], dt=0.02, dwell_steps=50)["passed"]
        )

    def test_invalid_physical_schedule_is_rejected(self) -> None:
        for dt in (0, -1, float("nan")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                force_schedule(PROBES[1], mass=75, gravity=9.81, dt=dt)
        with self.assertRaises(ValueError):
            force_schedule(
                StandProbe("bad", 20, (2, 0, 0), 0.05), mass=75, gravity=9.81, dt=0.02
            )


if __name__ == "__main__":
    unittest.main()
