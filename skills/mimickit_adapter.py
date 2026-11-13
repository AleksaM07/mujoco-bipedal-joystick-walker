"""Bridge MuJoCo hinge coordinates to MimicKit's kinematic model.

Virtual nodes represent serial hinges only in the reference skeleton. They do
not add physical bodies, actuators or inertia to the MuJoCo model. This is a
kinematics adapter, not an engine or policy/action adapter.
"""

import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

MIMICKIT_COMMIT = "fbe1a36e878f036d6bf5521defb8eb3b8ab4170d"


def interpolate_quaternions(q0: Any, q1: Any, fraction: Any) -> Any:
    """Interpolate xyzw rotations, preserving endpoints for small angles.

    Opt-in replacement for this pinned upstream's constant-midpoint fallback.
    Uses normalized linear interpolation near identity and shortest-arc SLERP
    elsewhere. No upstream files are changed.
    """
    import torch

    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    cosine = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(cosine < 0, -q1, q1)
    cosine = cosine.abs().clamp(max=1.0)
    fraction = fraction.unsqueeze(-1)
    linear = (1 - fraction) * q0 + fraction * q1
    theta = torch.acos(cosine)
    denominator = torch.sin(theta).clamp_min(1e-7)
    curved = (
        torch.sin((1 - fraction) * theta) * q0 + torch.sin(fraction * theta) * q1
    ) / denominator
    result = torch.where(cosine > 0.9995, linear, curved)
    return torch.nn.functional.normalize(result, dim=-1)


def load_mimickit(source: Path) -> Any:
    """Import the pinned, unmodified upstream kinematics implementation."""
    source = source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MIMICKIT_COMMIT:
        raise ValueError(f"Expected MimicKit {MIMICKIT_COMMIT}, found {revision}")
    changes = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if changes:
        raise ValueError("MimicKit checkout must be unmodified for this probe")
    package = source / "mimickit"
    if not (package / "anim/kin_char_model.py").is_file():
        raise FileNotFoundError(package / "anim/kin_char_model.py")
    sys.path.insert(0, str(package))
    module = import_module("anim.kin_char_model")
    if Path(module.__file__).resolve() != package / "anim/kin_char_model.py":
        raise ValueError("Another anim package is already imported")
    return module


@dataclass
class HingeKinematics:
    """Upstream skeleton plus explicit mappings to the original MuJoCo model."""

    skeleton: Any
    physical_body_nodes: np.ndarray
    joint_ids: np.ndarray
    qpos_indices: np.ndarray
    reference_angles: np.ndarray
    nq: int

    def encode_qpos(self, qpos: np.ndarray) -> tuple[Any, Any, Any]:
        """Convert batched MuJoCo wxyz root poses and hinge angles to Torch."""
        import torch

        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.nq or not len(qpos):
            raise ValueError(f"Expected nonempty qpos array with shape (N, {self.nq})")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains non-finite values")
        quaternion = qpos[:, 3:7]
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norm < 1e-12):
            raise ValueError("Root quaternion must be nonzero")
        quaternion = (quaternion / norm)[:, [1, 2, 3, 0]]
        angles = qpos[:, self.qpos_indices] - self.reference_angles
        return tuple(
            torch.as_tensor(value, dtype=torch.float32)
            for value in (qpos[:, :3], quaternion, angles)
        )

    def forward(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return physical body positions and xyzw orientations, excluding world."""
        root_pos, root_rot, angles = self.encode_qpos(qpos)
        rotations = self.skeleton.dof_to_rot(angles)
        positions, orientations = self.skeleton.forward_kinematics(
            root_pos, root_rot, rotations
        )
        return (
            positions[:, self.physical_body_nodes].numpy(),
            orientations[:, self.physical_body_nodes].numpy(),
        )


def build_hinge_kinematics(model: mujoco.MjModel, upstream: Any) -> HingeKinematics:
    """Build a serial-hinge skeleton without changing the physical model.

    Supports one free root followed by fixed bodies and origin-centred hinges.
    Unsupported translations, balls and nonzero hinge anchors fail explicitly.
    Compiled local transforms include MJCF Euler/default/compiler conventions.
    """
    import torch

    if (
        model.njnt < 1
        or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_bodyid[0] != 1
        or model.body_jntnum[1] != 1
        or np.any(model.body_parentid[2:] == 0)
    ):
        raise ValueError("Expected exactly one free-root character")
    if np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Only hinge joints are supported below the free root")
    if np.any(np.abs(model.jnt_pos[1:]) > 1e-12):
        raise ValueError("Nonzero hinge anchors are not supported")

    names: list[str] = []
    parents: list[int] = []
    translations: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    joints: list[Any] = []
    body_nodes: list[int] = []
    joint_ids: list[int] = []
    physical_nodes: dict[int, int] = {}
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    for body_id in range(1, model.nbody):
        body_name = model.body(body_id).name or f"body_{body_id}"
        parent_node = physical_nodes.get(int(model.body_parentid[body_id]), -1)
        start = int(model.body_jntadr[body_id])
        count = int(model.body_jntnum[body_id])
        body_joints = list(range(start, start + count)) if count else [None]
        for offset, joint_id in enumerate(body_joints):
            last = offset == len(body_joints) - 1
            names.append(body_name if last else f"__hinge_{body_id}_{offset}")
            parents.append(parent_node)
            translations.append(
                model.body_pos[body_id].copy() if offset == 0 else np.zeros(3)
            )
            rotations.append(
                model.body_quat[body_id][[1, 2, 3, 0]].copy()
                if offset == 0
                else identity.copy()
            )
            if body_id == 1:
                kind, axis, name = upstream.JointType.ROOT, None, body_name
            elif joint_id is None:
                kind, axis, name = upstream.JointType.FIXED, None, body_name
            else:
                kind = upstream.JointType.HINGE
                axis = torch.tensor(model.jnt_axis[joint_id], dtype=torch.float32)
                name = model.joint(joint_id).name
                joint_ids.append(joint_id)
            joints.append(upstream.Joint(name=name, joint_type=kind, axis=axis))
            parent_node = len(names) - 1
        physical_nodes[body_id] = parent_node
        body_nodes.append(parent_node)

    if len(set(names)) != len(names):
        raise ValueError("Physical body names collide with virtual hinge names")
    skeleton = upstream.KinCharModel(device="cpu")
    skeleton.init(names, parents, translations, rotations, joints)
    indices = model.jnt_qposadr[joint_ids].copy()
    return HingeKinematics(
        skeleton=skeleton,
        physical_body_nodes=np.asarray(body_nodes),
        joint_ids=np.asarray(joint_ids),
        qpos_indices=indices,
        reference_angles=model.qpos0[indices].copy(),
        nq=model.nq,
    )
