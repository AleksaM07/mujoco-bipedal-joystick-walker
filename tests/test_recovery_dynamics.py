"""Analytic and native MuJoCo checks for the reference dynamics diagnostic."""

from __future__ import annotations

import json
import unittest

import mujoco
import numpy as np

from tools.recovery_dynamics import inspect_dynamics


def make_slider(
    actuator: str, joint_extra: str = "", option: str = ""
) -> mujoco.MjModel:
    """Make a 2 kg vertical slider with known gravity and inertial force."""
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><option gravity="0 0 -10">{option}</option>
        <worldbody><body><joint name="slide" type="slide" axis="0 0 1"
        {joint_extra}/><geom type="sphere" size="0.1" mass="2"/></body></worldbody>
        <actuator>{actuator}</actuator></mujoco>"""
    )


def inspect_one(
    model: mujoco.MjModel,
    position: float = 0,
    velocity: float = 0,
    acceleration: float = 0,
) -> dict:
    return inspect_dynamics(model, [[position]], [[velocity]], [[acceleration]])


class DynamicsTests(unittest.TestCase):
    def test_positive_negative_and_unit_gear(self) -> None:
        for gear in (1, 2, -2):
            with self.subTest(gear=gear):
                model = make_slider(f'<motor joint="slide" gear="{gear}"/>')
                result = inspect_one(model, acceleration=3)
                actuator = result["actuators"][0]
                self.assertAlmostEqual(
                    actuator["required_generalized_force"]["mean"], 26
                )
                self.assertAlmostEqual(
                    actuator["required_scalar_force"]["mean"], 26 / gear
                )
                self.assertAlmostEqual(actuator["required_ctrl"]["mean"], 26 / gear)
                self.assertLess(
                    actuator["generalized_force_shortfall_after_native_clamps"][
                        "abs_max"
                    ],
                    1e-10,
                )
                json.dumps(result, allow_nan=False)
                extreme = result["extreme_sample_force_decomposition"][
                    "largest_absolute_actuator_scalar_force"
                ]
                self.assertAlmostEqual(extreme["equation_error_abs_max"], 0.0)
                self.assertAlmostEqual(actuator["reference_acceleration"]["mean"], 3)

    def test_position_control_in_actuator_length_units_and_passive_damping(
        self,
    ) -> None:
        model = make_slider(
            '<position joint="slide" gear="-2" kp="100" kv="3"/>',
            joint_extra='damping="4"',
        )
        actuator = inspect_one(model, position=0.2, velocity=0.5)["actuators"][0]
        # qfrc = 20 + damping*qvel = 22; f = -11, l = -.4, ldot = -1.
        expected_ctrl = (-11 - (40 + 3)) / 100
        self.assertAlmostEqual(actuator["required_ctrl"]["mean"], expected_ctrl)
        self.assertLess(
            actuator["generalized_force_shortfall_after_native_clamps"]["abs_max"],
            1e-10,
        )

    def test_asymmetric_force_range_and_gear_conversion(self) -> None:
        model = make_slider(
            '<motor joint="slide" gear="-2" forcelimited="true" forcerange="-3 20"/>'
        )
        actuator = inspect_one(model)["actuators"][0]
        self.assertEqual(actuator["required_scalar_force"]["mean"], -10)
        self.assertEqual(actuator["force_limit"]["exceedance"]["max"], 7)
        self.assertEqual(
            actuator["generalized_force_shortfall_after_native_clamps"]["mean"], 14
        )

    def test_ignored_ranges_do_not_become_fake_limits(self) -> None:
        model = make_slider(
            '<motor joint="slide" forcelimited="false" forcerange="-1 1" '
            'ctrllimited="false" ctrlrange="-1 1"/>'
        )
        actuator = inspect_one(model)["actuators"][0]
        self.assertFalse(actuator["force_limit"]["enabled"])
        self.assertFalse(actuator["ctrl_limit"]["enabled"])
        self.assertEqual(
            actuator["generalized_force_shortfall_after_native_clamps"]["abs_max"], 0
        )

    def test_control_limit_separate_from_large_force_capacity(self) -> None:
        model = make_slider(
            '<position joint="slide" kp="10" ctrllimited="true" ctrlrange="-1 1" '
            'forcelimited="true" forcerange="-100 100"/>'
        )
        result = inspect_one(model)
        actuator = result["actuators"][0]
        self.assertEqual(result["any_actuator_force_limit_exceeded_fraction"], 0)
        self.assertEqual(result["any_actuator_ctrl_limit_exceeded_fraction"], 1)
        self.assertEqual(actuator["required_ctrl"]["mean"], 2)
        self.assertEqual(
            actuator["generalized_force_shortfall_after_native_clamps"]["mean"], 10
        )

    def test_global_control_clamp_disable_is_respected(self) -> None:
        model = make_slider(
            '<motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>',
            option='<flag clampctrl="disable"/>',
        )
        actuator = inspect_one(model)["actuators"][0]
        self.assertFalse(actuator["ctrl_limit"]["enabled"])
        self.assertEqual(
            actuator["generalized_force_shortfall_after_native_clamps"]["abs_max"], 0
        )

    def test_joint_force_limit_checked_after_transmission(self) -> None:
        model = make_slider(
            '<motor joint="slide" gear="2"/>',
            joint_extra='actuatorfrclimited="true" actuatorfrcrange="-5 5"',
        )
        actuator = inspect_one(model)["actuators"][0]
        self.assertEqual(
            actuator["joint_actuator_force_limit"]["exceedance"]["max"], 15
        )
        self.assertEqual(
            actuator["generalized_force_shortfall_after_native_clamps"]["mean"], 15
        )

    def test_free_joint_gravity_residual(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 -10"/><worldbody><body>'
            '<freejoint name="root"/>'
            '<geom type="sphere" size="0.1" mass="2"/></body></worldbody></mujoco>'
        )
        positions = np.asarray([model.qpos0])
        result = inspect_dynamics(model, positions, np.zeros((1, 6)), np.zeros((1, 6)))
        self.assertEqual(len(result["unactuated_dofs"]), 6)
        self.assertEqual(result["free_joint_residuals"][0]["force_norm_N"]["mean"], 20)
        self.assertEqual(result["free_joint_residuals"][0]["torque_norm_Nm"]["mean"], 0)
        json.dumps(result, allow_nan=False)

    def test_hinge_torque_and_negative_gear(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 0"/><worldbody><body>'
            '<joint name="hinge" axis="0 0 1"/>'
            '<geom type="sphere" pos="1 0 0" size="0.1" mass="2"/>'
            '</body></worldbody><actuator><motor joint="hinge" gear="-3"/>'
            "</actuator></mujoco>"
        )
        actuator = inspect_one(model, acceleration=3)["actuators"][0]
        inertia = 2 * (1 + 0.4 * 0.1**2)
        self.assertEqual(actuator["generalized_force_unit"], "Nm")
        self.assertAlmostEqual(
            actuator["required_generalized_force"]["mean"], 3 * inertia
        )
        self.assertAlmostEqual(actuator["required_scalar_force"]["mean"], -inertia)

    def test_free_fall_requires_no_external_root_force(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 -10"/><worldbody><body><freejoint/>'
            '<geom type="sphere" size="0.1" mass="2"/></body></worldbody></mujoco>'
        )
        result = inspect_dynamics(
            model, model.qpos0[None, :], np.zeros((1, 6)), [[0, 0, -10, 0, 0, 0]]
        )
        self.assertEqual(result["free_joint_residuals"][0]["force_norm_N"]["mean"], 0)

    def test_native_contacts_are_already_in_inverse_residual(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option gravity="0 0 -10"/><worldbody>'
            '<geom type="plane" size="1 1 0.1"/><body pos="0 0 0.099">'
            '<freejoint/><geom type="sphere" size="0.1" mass="2"/>'
            "</body></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_inverse(model, data)
        self.assertGreater(data.ncon, 0)
        self.assertGreater(abs(data.qfrc_constraint[2]), 0)
        result = inspect_dynamics(model, model.qpos0[None, :], [[0] * 6], [[0] * 6])
        required_z = result["unactuated_dofs"][2]["required_force"]["mean"]
        self.assertAlmostEqual(required_z, 20 - data.qfrc_constraint[2])

    def test_affine_gain_matches_native(self) -> None:
        model = make_slider(
            '<general joint="slide" gaintype="affine" gainprm="3 2 1" '
            'biastype="affine" biasprm="1 -4 -2"/>'
        )
        actuator = inspect_one(model, position=0.2, velocity=0.3)["actuators"][0]
        self.assertAlmostEqual(
            actuator["required_ctrl"]["mean"], (20 - (1 - 0.8 - 0.6)) / 3.7
        )
        self.assertLess(
            actuator["generalized_force_shortfall_after_native_clamps"]["abs_max"],
            1e-10,
        )

    def test_reject_ambiguous_stateful_or_singular_actuation(self) -> None:
        actuators = [
            '<motor joint="slide"/><motor joint="slide"/>',
            '<general joint="slide" dyntype="filter" dynprm="0.1"/>',
            '<motor joint="slide" gear="0"/>',
            '<general joint="slide" gainprm="0"/>',
        ]
        for actuator in actuators:
            with self.subTest(actuator=actuator), self.assertRaises(ValueError):
                inspect_one(make_slider(actuator))

    def test_reject_invalid_batches(self) -> None:
        model = make_slider('<motor joint="slide"/>')
        for positions, velocities, accelerations in (
            ([], [], []),
            ([[0]], [[0, 0]], [[0]]),
            ([[float("nan")]], [[0]], [[0]]),
        ):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                inspect_dynamics(model, positions, velocities, accelerations)


if __name__ == "__main__":
    unittest.main()
