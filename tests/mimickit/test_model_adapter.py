"""Check the optional adapter against native physics with nontrivial joint frames."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from skills.mimickit_adapter import (
    build_hinge_kinematics,
    interpolate_quaternions,
    load_mimickit,
)
from tools.check_mimickit_model import actuator_mapping, compare_poses, sample_poses

SOURCE = Path(__file__).resolve().parents[2] / ".tmp_mimickit_navigation"


@pytest.fixture(scope="module")
def upstream():
    if not SOURCE.exists():
        pytest.skip(
            "Requires the pinned MimicKit checkout; see docs/MIMICKIT_PREFLIGHT.md"
        )
    return load_mimickit(SOURCE)


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/>
      <worldbody><body name="root" pos="1 2 3" euler="0.4 -0.2 0.3">
        <freejoint/><geom type="sphere" size="0.1" mass="2"/>
        <body name="child" pos="0.2 0.3 -0.4" euler="0.3 0.5 -0.2">
          <joint name="passive" axis="1 1 0" range="-1 1" ref="0.15"/>
          <joint name="active" axis="0 1 1" range="-0.5 0.8" ref="-0.2"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body name="tip" pos="0.1 -0.2 0.3" euler="0.2 0.4 0.7">
            <geom type="sphere" size="0.05" mass="0.2"/>
          </body>
        </body>
      </body></worldbody>
      <actuator><position name="drive" joint="active" kp="30"/></actuator>
    </mujoco>""")


def test_serial_axes_local_rotations_and_reference_offsets_match_mujoco(
    model, upstream
):
    adapter = build_hinge_kinematics(model, upstream)
    poses = sample_poses(model, 128, 22)
    assert compare_poses(model, adapter, poses)["passed"]
    poses[:, 3:7] *= -2  # Quaternion sign and magnitude do not change orientation.
    assert compare_poses(model, adapter, poses)["passed"]
    assert len(adapter.joint_ids) == 2
    assert actuator_mapping(model, adapter)[0]["kinematic_dof_index"] == 1


def test_oracle_detects_wrong_local_translation(model, upstream):
    adapter = build_hinge_kinematics(model, upstream)
    adapter.skeleton._local_translation[-1, 0] += 0.02
    result = compare_poses(model, adapter, sample_poses(model, 3, 7))
    assert not result["passed"]
    assert result["max_position_error_m"] > 0.019


@pytest.mark.parametrize("invalid", ["shape", "nan", "zero_quaternion"])
def test_invalid_states_fail_explicitly(model, upstream, invalid):
    adapter = build_hinge_kinematics(model, upstream)
    poses = sample_poses(model, 2, 7)
    if invalid == "shape":
        poses = poses[:, :-1]
    elif invalid == "nan":
        poses[0, 0] = np.nan
    else:
        poses[0, 3:7] = 0
    with pytest.raises(ValueError):
        adapter.forward(poses)


def test_nonzero_anchor_is_rejected(model, upstream):
    model.jnt_pos[1, 0] = 0.1
    with pytest.raises(ValueError, match="anchors"):
        build_hinge_kinematics(model, upstream)


def test_unsupported_joint_is_rejected(model, upstream):
    model.jnt_type[1] = mujoco.mjtJoint.mjJNT_SLIDE
    with pytest.raises(ValueError, match="Only hinge"):
        build_hinge_kinematics(model, upstream)


@pytest.mark.parametrize("angle", [0.0, 0.001, 0.01, 1.2, 3.0])
def test_interpolation_preserves_endpoints_and_constant_angular_speed(angle):
    import torch

    fractions = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    q0 = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(5, 4)
    q1 = torch.tensor(
        [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)], dtype=torch.float32
    ).expand(5, 4)
    result = interpolate_quaternions(q0, -q1, fractions)
    expected = torch.zeros_like(result)
    expected[:, 2] = torch.sin(fractions * angle / 2)
    expected[:, 3] = torch.cos(fractions * angle / 2)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)


def test_pinned_upstream_small_angle_fallback_changes_first_frame(upstream):
    from importlib import import_module

    import torch

    q0 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    q1 = torch.tensor([[0.0, 0.0, np.sin(0.0005), np.cos(0.0005)]], dtype=torch.float32)
    result = import_module("util.torch_util").slerp(q0, q1, torch.zeros(1))
    assert torch.linalg.norm(result - q0).item() > 1e-4
    torch.testing.assert_close(interpolate_quaternions(q0, q1, torch.zeros(1)), q0)
