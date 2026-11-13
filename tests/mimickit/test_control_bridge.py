"""Check native actuator semantics without depending on MimicKit or training."""

import mujoco
import numpy as np
import pytest

from skills.mujoco_control import hinge_actuator_forces, validate_hinge_actuators


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/>
      <worldbody><body pos="0 0 1"><freejoint/>
        <geom type="sphere" size="0.1" mass="2"/>
        <body><joint name="passive" axis="1 0 0" stiffness="30" damping="4"/>
          <joint name="active" axis="0 0 1" ref="0.2" stiffness="10" damping="2"/>
          <geom type="capsule" size="0.04 0.2" mass="1"/>
        </body>
      </body></worldbody>
      <actuator>
        <position joint="active" gear="-2" kp="45" kv="3"
          ctrlrange="-0.3 0.7" forcerange="-8 11"/>
        <position joint="active" gear="0.5" kp="12"
          ctrlrange="-0.8 0.9" forcerange="-4 5"/>
      </actuator>
    </mujoco>""")


@pytest.mark.parametrize("target", [-10, -0.1, 0.4, 10])
def test_forces_match_native_clipping_gear_bias_and_multiple_actuators(model, target):
    validate_hinge_actuators(model)
    data = mujoco.MjData(model)
    data.qpos[-2:] = [0.3, -0.12]
    data.qvel[-2:] = [-0.4, 1.7]
    data.ctrl[:] = [target, -target]
    mujoco.mj_forward(model, data)
    force, torque = hinge_actuator_forces(model, data.qpos, data.qvel, data.ctrl)
    np.testing.assert_allclose(force, data.actuator_force, atol=1e-12)
    np.testing.assert_allclose(torque, data.qfrc_actuator, atol=1e-12)
    assert torque[-2] == 0
    assert abs(data.qfrc_passive[-2]) > 1


def test_zero_passive_stiffness_does_not_disable_active_motor(model):
    model.jnt_stiffness[:] = 0
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.4
    mujoco.mj_forward(model, data)
    _, torque = hinge_actuator_forces(model, data.qpos, data.qvel, data.ctrl)
    assert abs(torque[-1]) > 1
    np.testing.assert_allclose(torque, data.qfrc_actuator)


@pytest.mark.parametrize("invalid", ["shape", "nan", "inf"])
def test_invalid_targets_rejected(model, invalid):
    targets = np.zeros(model.nu - 1 if invalid == "shape" else model.nu)
    if invalid != "shape":
        targets[0] = np.nan if invalid == "nan" else np.inf
    with pytest.raises(ValueError, match="finite vector"):
        hinge_actuator_forces(model, model.qpos0, np.zeros(model.nv), targets)


def test_activation_dynamics_rejected(model):
    model.actuator_dyntype[0] = mujoco.mjtDyn.mjDYN_FILTER
    with pytest.raises(ValueError, match="stateless"):
        validate_hinge_actuators(model)
