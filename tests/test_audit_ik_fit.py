"""Native single-joint fixtures check diagnostic IK signs, bounds and isolation."""

import copy
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from tools.audit_ik_fit import _FrameObjective, inspect_ik_fit


def make_fixture(target_angle: float = 0.3) -> tuple:
    """Return a tiny fixed-root capture with one reachable revolute marker."""
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
        <body><joint name="left_hip_x" axis="0 1 0" range="-.5 .5"/>
          <geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
          <site name="metatarsal_midpoint_left" pos=".5 0 0"/>
        </body></body></worldbody>
        <compiler angle="radian"/>
        <actuator><position joint="left_hip_x" ctrlrange="-.5 .5"/></actuator>
      </mujoco>
    """)
    env = SimpleNamespace(
        _mj_model=model,
        _init_q_np=model.qpos0.copy(),
        _actuator_qpos_indices_np=np.array([7]),
        _actuator_dof_indices_np=np.array([6]),
        _actuator_joint_names=("left_hip_x",),
        _actuator_ctrl_lower_limits_np=np.array([-0.5]),
        _actuator_ctrl_upper_limits_np=np.array([0.5]),
        FOOT_CONTACT_HEIGHT=0.0,
    )
    for name in (
        "_host_marker_position",
        "_host_marker_jacobian",
        "_resolve_host_ik_markers",
    ):
        setattr(env, name, getattr(BiomechanicsJoystickEnv, name).__get__(env))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[7] = target_angle
    mujoco.mj_forward(model, data)
    stage = {
        "joints": np.zeros((1, 2, 1)),
        "roots": np.repeat(model.qpos0[None, None, :3], 2, axis=1),
        "quaternions": np.repeat(model.qpos0[None, None, 3:7], 2, axis=1),
        "targets": np.repeat(data.site_xpos[None, None].copy(), 2, axis=1),
    }
    stages = [
        {"name": name, **copy.deepcopy(stage)}
        for name in ("normalized_ik_input", "ik_solution", "stability_prior")
    ]
    metadata = {
        "names": ("metatarsal_midpoint_left",),
        "counts": np.array([2]),
        "source_start_frames": np.array([79]),
    }
    return env, metadata, stages


class AuditIKFitTests(unittest.TestCase):
    def test_native_jacobian_matches_finite_difference(self) -> None:
        env, metadata, stages = make_fixture()
        objective = _FrameObjective(
            env,
            env._resolve_host_ik_markers(metadata["names"]),
            env._init_q_np,
            stages[0]["targets"][0, 0],
            np.array([0]),
            np.array([0.0]),
            np.array([0.1]),
            0.20,
            0.08,
        )
        coordinate = np.array([0.2])
        step = 1e-6
        numeric = (
            objective.residual(coordinate + step)
            - objective.residual(coordinate - step)
        ) / (2 * step)
        np.testing.assert_allclose(
            objective.jacobian(coordinate)[:, 0], numeric, atol=1e-9
        )

    def test_refit_improves_target_and_preserves_inputs(self) -> None:
        env, metadata, stages = make_fixture()
        originals = copy.deepcopy(stages)
        result = inspect_ik_fit(env, metadata, stages)
        self.assertEqual(result["selected_frames"], [0, 1])
        for row in result["frames"]:
            baseline = row["production_ik"]["weighted_marker_squared_error_m2"]
            fitted = row["fits"]
            self.assertLess(
                fitted["converged_regularized"]["weighted_marker_squared_error_m2"],
                baseline,
            )
            self.assertLess(
                fitted["no_regularization"]["weighted_marker_squared_error_m2"], 1e-18
            )
            self.assertAlmostEqual(
                fitted["no_regularization"]["actuator_pose_rad"][0], 0.3
            )
            self.assertEqual(row["root_xyz_m"], [0.0, 0.0, 1.0])
        for before, after in zip(originals, stages, strict=True):
            for key in ("joints", "roots", "quaternions", "targets"):
                np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_array_equal(env._init_q_np, env._mj_model.qpos0)

    def test_unreachable_target_respects_bound(self) -> None:
        env, metadata, stages = make_fixture(target_angle=0.8)
        result = inspect_ik_fit(env, metadata, stages, frame_ids=[0])
        fitted = result["frames"][0]["fits"]["no_regularization"]
        self.assertEqual(fitted["active_upper_bounds"], ["left_hip_x"])
        self.assertAlmostEqual(fitted["actuator_pose_rad"][0], 0.5)
        self.assertGreater(fitted["weighted_marker_squared_error_m2"], 0.01)
        self.assertEqual(fitted["max_physical_joint_limit_violation_rad"], 0.0)

    def test_invalid_frames_stages_and_markers_fail_closed(self) -> None:
        env, metadata, stages = make_fixture()
        for frames in ([], [-1], [2], [0.25]):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                inspect_ik_fit(env, metadata, stages, frame_ids=frames)
        with self.assertRaisesRegex(ValueError, "Missing captured"):
            inspect_ik_fit(env, metadata, stages[:1])
        metadata["names"] = ("missing_marker",)
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            inspect_ik_fit(env, metadata, stages)

    def test_physical_joint_limits_are_reported_separately(self) -> None:
        env, metadata, stages = make_fixture(target_angle=0.4)
        joint_id = env._mj_model.joint("left_hip_x").id
        env._mj_model.jnt_range[joint_id] = [-0.2, 0.2]
        result = inspect_ik_fit(env, metadata, stages, frame_ids=[0])
        fitted = result["frames"][0]["fits"]["no_regularization"]
        self.assertAlmostEqual(fitted["actuator_pose_rad"][0], 0.4)
        self.assertAlmostEqual(fitted["max_physical_joint_limit_violation_rad"], 0.2)

    def test_nonfinite_capture_is_rejected(self) -> None:
        env, metadata, stages = make_fixture()
        stages[0]["targets"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            inspect_ik_fit(env, metadata, stages, frame_ids=[0])


if __name__ == "__main__":
    unittest.main()
