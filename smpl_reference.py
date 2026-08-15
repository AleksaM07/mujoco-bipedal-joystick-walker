from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bvh_reference import BvhReferenceBatch, LoopMode


SMPL_JOINT_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
}

SMPL_BONE_ORDER_NAMES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)

SMPL_MUJOCO_NAMES = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)

SMPL_TO_MUJOCO_JOINT_ORDER = np.array(
    [SMPL_BONE_ORDER_NAMES.index(name) for name in SMPL_MUJOCO_NAMES],
    dtype=np.int32,
)

SMPL_MUJOCO_JOINT_INDEX = {
    name: index for index, name in enumerate(SMPL_MUJOCO_NAMES)
}

SMPL_MUJOCO_PARENT_INDICES = np.array(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22],
    dtype=np.int32,
)

SMPL_MUJOCO_LOCAL_TRANSLATION = np.array(
    [
        [0.0000, 0.0000, 0.0000],
        [-0.0068, 0.0695, -0.0914],
        [-0.0045, 0.0343, -0.3752],
        [-0.0437, -0.0136, -0.3980],
        [0.1193, 0.0124, -0.0258],
        [-0.0043, -0.0677, -0.0905],
        [-0.0089, -0.0383, -0.3826],
        [-0.0423, 0.0158, -0.3984],
        [0.1193, -0.0124, -0.0258],
        [-0.0267, -0.0025, 0.1090],
        [0.0011, 0.0055, 0.1352],
        [0.0254, 0.0015, 0.0529],
        [-0.0429, -0.0028, 0.2139],
        [0.0513, 0.0052, 0.0650],
        [-0.0341, 0.0788, 0.1217],
        [-0.0089, 0.0910, 0.0305],
        [-0.0275, 0.2596, -0.0128],
        [-0.0012, 0.2492, 0.0090],
        [-0.0149, 0.0840, -0.0082],
        [-0.0386, -0.0818, 0.1188],
        [-0.0091, -0.0960, 0.0326],
        [-0.0214, -0.2537, -0.0133],
        [-0.0056, -0.2553, 0.0078],
        [-0.0103, -0.0846, -0.0061],
    ],
    dtype=np.float32,
)

SMPL_IK_TARGET_MAP = (
    ("metatarsal_midpoint_right", "R_Toe"),
    ("metatarsal_midpoint_left", "L_Toe"),
)

SMPL_TO_MUJOCO = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

SMPL_RETARGET_GAIN = {
    "abdomen_x": 0.75,
    "abdomen_y": 0.75,
    "abdomen_z": 0.75,
    "left_hip_z": 0.65,
    "right_hip_z": 0.65,
    "left_ankle_z": 0.85,
    "right_ankle_z": 0.85,
}


@dataclass(frozen=True)
class SmplMotionClip:
    qpos_targets: np.ndarray
    qvel_targets: np.ndarray
    root_pos_targets: np.ndarray
    root_quat_targets: np.ndarray
    root_vel_targets: np.ndarray
    root_angvel_targets: np.ndarray
    wrap_delta: np.ndarray
    frame_time: float
    source_path: Path
    loop_mode: LoopMode
    weight: float
    support_foot: str
    diagnostic_summary: str
    ik_target_positions: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.qpos_targets.shape[0])


def load_smpl_references(
    paths: tuple[str | Path, ...],
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    initial_root_pos: np.ndarray | None = None,
    initial_root_quat: np.ndarray | None = None,
) -> BvhReferenceBatch:
    initial_root_pos = (
        np.asarray(initial_root_pos, dtype=np.float32)
        if initial_root_pos is not None
        else np.zeros(3, dtype=np.float32)
    )
    initial_root_quat = (
        _normalize_quat(np.asarray(initial_root_quat, dtype=np.float32))
        if initial_root_quat is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    clips = [
        _load_smpl_clip(
            Path(path),
            actuator_joint_names,
            np.asarray(default_ctrl, dtype=np.float32),
            np.asarray(lower_limits, dtype=np.float32),
            np.asarray(upper_limits, dtype=np.float32),
            initial_root_pos,
            initial_root_quat,
        )
        for path in paths
    ]
    if not clips:
        raise ValueError("Potrebna je bar jedna validna SMPL motion referenca.")
    return _clips_to_batch(tuple(clips))


def _load_smpl_clip(
    path: Path,
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    initial_root_pos: np.ndarray,
    initial_root_quat: np.ndarray,
) -> SmplMotionClip:
    source_path = path
    bundle = np.load(source_path, allow_pickle=True)
    poses = np.asarray(bundle["poses"], dtype=np.float32)
    trans = np.asarray(bundle["trans"], dtype=np.float32)
    frame_rate = float(np.asarray(bundle["mocap_framerate"]).reshape(()))
    frame_time = 1.0 / max(frame_rate, 1e-6)
    frame_count = int(poses.shape[0])
    targets = np.tile(default_ctrl[None, :], (frame_count, 1)).astype(np.float32)
    smpl_global_positions = _smpl_global_joint_positions(poses, trans)
    chest_positions = smpl_global_positions[:, SMPL_MUJOCO_JOINT_INDEX["Chest"]]

    spine1 = _joint_axis_angles_to_mujoco(poses, "spine1")
    spine2 = _joint_axis_angles_to_mujoco(poses, "spine2")
    left_hip = _joint_axis_angles_to_mujoco(poses, "left_hip")
    right_hip = _joint_axis_angles_to_mujoco(poses, "right_hip")
    left_knee = _joint_axis_angles_to_mujoco(poses, "left_knee")
    right_knee = _joint_axis_angles_to_mujoco(poses, "right_knee")
    left_ankle = _joint_axis_angles_to_mujoco(poses, "left_ankle")
    right_ankle = _joint_axis_angles_to_mujoco(poses, "right_ankle")
    abdomen = 0.65 * spine1 + 0.35 * spine2

    absolute = {
        "abdomen_x": abdomen[:, 0],
        "abdomen_y": abdomen[:, 1],
        "abdomen_z": abdomen[:, 2],
        "left_hip_x": left_hip[:, 0],
        "left_hip_y": left_hip[:, 2],
        "left_hip_z": left_hip[:, 1],
        "right_hip_x": right_hip[:, 0],
        "right_hip_y": right_hip[:, 2],
        "right_hip_z": right_hip[:, 1],
        "left_knee_z": -left_knee[:, 1],
        "right_knee_z": -right_knee[:, 1],
        "left_ankle_y": left_ankle[:, 1],
        "right_ankle_y": right_ankle[:, 1],
        "left_ankle_z": left_ankle[:, 2],
        "right_ankle_z": right_ankle[:, 2],
    }
    bind_frame = _standing_like_frame_index(absolute)
    for joint_name, values in absolute.items():
        if joint_name not in actuator_joint_names:
            continue
        index = actuator_joint_names.index(joint_name)
        bind_offset = float(default_ctrl[index] - values[bind_frame])
        aligned = values + bind_offset
        gain = float(SMPL_RETARGET_GAIN.get(joint_name, 1.0))
        softened = default_ctrl[index] + gain * (aligned - default_ctrl[index])
        clipped = np.clip(softened, lower_limits[index], upper_limits[index])
        targets[:, index] = clipped.astype(np.float32)
    support_foot = _infer_support_foot_from_targets(absolute)
    diagnostic_summary = _build_clip_diagnostic_summary(
        targets,
        actuator_joint_names,
        lower_limits,
        upper_limits,
    )

    root_pos, root_quat, alignment_rot, raw_root_origin = _smpl_root_motion_targets(
        chest_positions,
        poses[:, :3],
        initial_root_pos,
        initial_root_quat,
    )
    ik_target_positions = _smpl_ik_target_positions(
        smpl_global_positions,
        alignment_rot,
        raw_root_origin,
        initial_root_pos,
    )
    qvel_targets = _target_velocities(targets, frame_time)
    root_vel_targets = _target_velocities(root_pos, frame_time)
    root_angvel_targets = _quat_angular_velocities(root_quat, frame_time)
    loop_mode = _resolve_loop_mode(
        qpos_targets=targets,
        qvel_targets=qvel_targets,
        root_pos_targets=root_pos,
        root_quat_targets=root_quat,
        root_vel_targets=root_vel_targets,
        root_angvel_targets=root_angvel_targets,
    )
    return SmplMotionClip(
        qpos_targets=targets,
        qvel_targets=qvel_targets,
        root_pos_targets=root_pos,
        root_quat_targets=root_quat,
        root_vel_targets=root_vel_targets,
        root_angvel_targets=root_angvel_targets,
        wrap_delta=_motion_wrap_delta(root_pos),
        frame_time=frame_time,
        source_path=source_path,
        loop_mode=loop_mode,
        weight=max(frame_count - 1, 1) * frame_time,
        support_foot=support_foot,
        diagnostic_summary=diagnostic_summary,
        ik_target_positions=ik_target_positions,
    )


def _clips_to_batch(clips: tuple[SmplMotionClip, ...]) -> BvhReferenceBatch:
    max_frames = max(clip.frame_count for clip in clips)
    action_size = clips[0].qpos_targets.shape[1]
    clip_count = len(clips)
    qpos_targets = np.zeros((clip_count, max_frames, action_size), dtype=np.float32)
    qvel_targets = np.zeros_like(qpos_targets)
    root_pos_targets = np.zeros((clip_count, max_frames, 3), dtype=np.float32)
    root_quat_targets = np.zeros((clip_count, max_frames, 4), dtype=np.float32)
    root_vel_targets = np.zeros((clip_count, max_frames, 3), dtype=np.float32)
    root_angvel_targets = np.zeros((clip_count, max_frames, 3), dtype=np.float32)
    wrap_deltas = np.zeros((clip_count, 3), dtype=np.float32)
    frame_times = np.zeros(clip_count, dtype=np.float32)
    frame_counts = np.zeros(clip_count, dtype=np.int32)
    loop_modes = np.zeros(clip_count, dtype=np.int32)
    weights = np.zeros(clip_count, dtype=np.float32)
    source_start_frames = np.zeros(clip_count, dtype=np.int32)
    source_end_frames = np.zeros(clip_count, dtype=np.int32)
    ik_target_count = clips[0].ik_target_positions.shape[1]
    ik_target_positions = np.zeros(
        (clip_count, max_frames, ik_target_count, 3),
        dtype=np.float32,
    )
    for index, clip in enumerate(clips):
        fc = clip.frame_count
        qpos_targets[index, :fc] = clip.qpos_targets
        qpos_targets[index, fc:] = clip.qpos_targets[-1]
        qvel_targets[index, :fc] = clip.qvel_targets
        root_pos_targets[index, :fc] = clip.root_pos_targets
        root_pos_targets[index, fc:] = clip.root_pos_targets[-1]
        root_quat_targets[index, :fc] = clip.root_quat_targets
        root_quat_targets[index, fc:] = clip.root_quat_targets[-1]
        root_vel_targets[index, :fc] = clip.root_vel_targets
        root_angvel_targets[index, :fc] = clip.root_angvel_targets
        ik_target_positions[index, :fc] = clip.ik_target_positions
        ik_target_positions[index, fc:] = clip.ik_target_positions[-1]
        wrap_deltas[index] = clip.wrap_delta
        frame_times[index] = clip.frame_time
        frame_counts[index] = fc
        loop_modes[index] = int(clip.loop_mode)
        weights[index] = max(float(clip.weight), 0.0)
        source_end_frames[index] = fc
    weight_sum = float(weights.sum())
    if weight_sum > 0.0:
        weights /= weight_sum
    else:
        weights[:] = 1.0 / float(clip_count)
    return BvhReferenceBatch(
        qpos_targets=qpos_targets,
        qvel_targets=qvel_targets,
        root_pos_targets=root_pos_targets,
        root_quat_targets=root_quat_targets,
        root_vel_targets=root_vel_targets,
        root_angvel_targets=root_angvel_targets,
        wrap_deltas=wrap_deltas,
        frame_times=frame_times,
        frame_counts=frame_counts,
        loop_modes=loop_modes,
        weights=weights,
        source_paths=tuple(clip.source_path for clip in clips),
        source_start_frames=source_start_frames,
        source_end_frames=source_end_frames,
        support_feet=tuple(clip.support_foot for clip in clips),
        diagnostic_summaries=tuple(clip.diagnostic_summary for clip in clips),
        ik_target_names=tuple(name for name, _ in SMPL_IK_TARGET_MAP),
        ik_target_positions=ik_target_positions,
    )


def _smpl_global_joint_positions(poses: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Compute SMPL global joint positions in MimicKit/MuJoCo joint order."""
    frame_count = int(poses.shape[0])
    body_axis_angles = np.concatenate(
        [
            poses[:, :66],
            np.zeros((frame_count, 6), dtype=np.float32),
        ],
        axis=1,
    )
    local_axis_angles = body_axis_angles.reshape(frame_count, 24, 3)[
        :,
        SMPL_TO_MUJOCO_JOINT_ORDER,
    ]
    local_rotations = _batch_axis_angle_to_matrix(
        local_axis_angles.reshape(-1, 3)
    ).reshape(frame_count, 24, 3, 3)
    global_rotations = np.zeros_like(local_rotations)
    global_positions = np.zeros((frame_count, 24, 3), dtype=np.float32)
    for joint_index, parent_index in enumerate(SMPL_MUJOCO_PARENT_INDICES):
        if parent_index < 0:
            global_rotations[:, joint_index] = local_rotations[:, joint_index]
            global_positions[:, joint_index] = SMPL_MUJOCO_LOCAL_TRANSLATION[joint_index]
            continue
        parent_rot = global_rotations[:, parent_index]
        local_offset = SMPL_MUJOCO_LOCAL_TRANSLATION[joint_index]
        global_rotations[:, joint_index] = (
            parent_rot @ local_rotations[:, joint_index]
        )
        global_positions[:, joint_index] = (
            global_positions[:, parent_index]
            + np.einsum("nij,j->ni", parent_rot, local_offset)
        )
    global_positions += trans[:, None, :]
    return global_positions.astype(np.float32)


def _smpl_ik_target_positions(
    smpl_global_positions: np.ndarray,
    alignment_rot: np.ndarray,
    raw_root_origin: np.ndarray,
    initial_root_pos: np.ndarray,
) -> np.ndarray:
    """Transform SMPL marker targets into the same world frame as root targets."""
    selected = np.stack(
        [
            smpl_global_positions[:, SMPL_MUJOCO_JOINT_INDEX[joint_name]]
            for _, joint_name in SMPL_IK_TARGET_MAP
        ],
        axis=1,
    )
    flat = selected.reshape(-1, 3)
    mujoco_positions = _smpl_positions_to_mujoco(flat).reshape(selected.shape)
    mujoco_positions -= raw_root_origin[None, None, :]
    mujoco_positions = (
        mujoco_positions @ alignment_rot.T[None, :, :] + initial_root_pos[None, None, :]
    )
    return mujoco_positions.astype(np.float32)


def _joint_axis_angles_to_mujoco(poses: np.ndarray, joint_name: str) -> np.ndarray:
    joint_index = SMPL_JOINT_INDEX[joint_name]
    axis_angles = poses[:, joint_index * 3 : (joint_index + 1) * 3]
    rotations = _batch_axis_angle_to_matrix(axis_angles)
    mujoco_rotations = SMPL_TO_MUJOCO[None, :, :] @ rotations @ SMPL_TO_MUJOCO.T[None, :, :]
    return _batch_matrix_to_euler_xyz(mujoco_rotations).astype(np.float32)


def _standing_like_frame_index(absolute: dict[str, np.ndarray]) -> int:
    left_knee = absolute["left_knee_z"]
    right_knee = absolute["right_knee_z"]
    knee_energy = np.square(left_knee) + np.square(right_knee)
    return int(np.argmin(knee_energy))


def _infer_support_foot_from_targets(absolute: dict[str, np.ndarray]) -> str:
    """Infer a dominant support foot from ankle/knee sagittal motion."""
    left_score = (
        np.abs(absolute["left_ankle_y"])
        + 0.5 * np.abs(absolute["left_ankle_z"])
        + 0.35 * np.abs(absolute["left_knee_z"])
    )
    right_score = (
        np.abs(absolute["right_ankle_y"])
        + 0.5 * np.abs(absolute["right_ankle_z"])
        + 0.35 * np.abs(absolute["right_knee_z"])
    )
    window = max(int(min(left_score.shape[0], 30)), 1)
    left_mean = float(np.mean(left_score[:window]))
    right_mean = float(np.mean(right_score[:window]))
    margin = abs(left_mean - right_mean)
    if margin < 0.025:
        return ""
    return "left_foot" if left_mean < right_mean else "right_foot"


def _build_clip_diagnostic_summary(
    targets: np.ndarray,
    actuator_joint_names: tuple[str, ...],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> str:
    """Summarize heavy retarget clipping for audit logs."""
    findings: list[tuple[float, str]] = []
    for index, joint_name in enumerate(actuator_joint_names):
        values = targets[:, index]
        lower = float(lower_limits[index])
        upper = float(upper_limits[index])
        if np.isfinite(lower):
            lower_fraction = float(np.mean(np.isclose(values, lower, atol=1e-4)))
            if lower_fraction >= 0.05:
                findings.append((lower_fraction, f"{joint_name}@lo={lower_fraction:.0%}"))
        if np.isfinite(upper):
            upper_fraction = float(np.mean(np.isclose(values, upper, atol=1e-4)))
            if upper_fraction >= 0.05:
                findings.append((upper_fraction, f"{joint_name}@hi={upper_fraction:.0%}"))
    if not findings:
        return ""
    findings.sort(key=lambda item: item[0], reverse=True)
    top = ", ".join(label for _, label in findings[:4])
    return f"retarget_clipping[{top}]"


def _smpl_root_motion_targets(
    root_source_positions: np.ndarray,
    pelvis_axis_angles: np.ndarray,
    initial_root_pos: np.ndarray,
    initial_root_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root_pos_local = _smpl_positions_to_mujoco(root_source_positions)
    raw_root_origin = root_pos_local[0].copy()
    root_pos_local -= raw_root_origin
    rotations = _batch_axis_angle_to_matrix(pelvis_axis_angles)
    mujoco_rotations = SMPL_TO_MUJOCO[None, :, :] @ rotations @ SMPL_TO_MUJOCO.T[None, :, :]
    raw_root_quat = _batch_matrix_to_quat(mujoco_rotations)
    raw_root_quat = _continuous_quat_sequence(raw_root_quat)
    raw_root_heading_quat = _continuous_quat_sequence(
        np.stack([_yaw_only_quat(quat) for quat in raw_root_quat], axis=0)
    )
    initial_root_quat = _normalize_quat(initial_root_quat)
    initial_root_heading_quat = _yaw_only_quat(initial_root_quat)
    initial_root_local_quat = _quat_mul(
        _quat_conjugate(initial_root_heading_quat),
        initial_root_quat,
    )
    alignment_quat = _quat_mul(
        initial_root_heading_quat,
        _quat_conjugate(raw_root_heading_quat[0]),
    )
    alignment_rot = _quat_to_matrix(alignment_quat)
    root_pos = root_pos_local @ alignment_rot.T
    root_pos += initial_root_pos[None, :]
    root_quat = np.stack(
        [
            _quat_mul(_quat_mul(alignment_quat, quat), initial_root_local_quat)
            for quat in raw_root_heading_quat
        ],
        axis=0,
    )
    return (
        root_pos.astype(np.float32),
        root_quat.astype(np.float32),
        alignment_rot.astype(np.float32),
        raw_root_origin.astype(np.float32),
    )


def _smpl_positions_to_mujoco(positions: np.ndarray) -> np.ndarray:
    scaled = positions.astype(np.float32)
    return np.stack([scaled[:, 2], -scaled[:, 0], scaled[:, 1]], axis=-1)


def _target_velocities(targets: np.ndarray, frame_time: float) -> np.ndarray:
    if targets.shape[0] < 2:
        return np.zeros_like(targets, dtype=np.float32)
    return np.gradient(targets, frame_time, axis=0).astype(np.float32)


def _quat_angular_velocities(quats: np.ndarray, frame_time: float) -> np.ndarray:
    if quats.shape[0] < 2:
        return np.zeros((quats.shape[0], 3), dtype=np.float32)
    velocities = np.zeros((quats.shape[0], 3), dtype=np.float32)
    dt = max(frame_time, 1e-6)
    velocities[0] = _quat_pair_angular_velocity(quats[0], quats[1], dt)
    for index in range(1, quats.shape[0] - 1):
        prev_velocity = _quat_pair_angular_velocity(quats[index - 1], quats[index], dt)
        next_velocity = _quat_pair_angular_velocity(quats[index], quats[index + 1], dt)
        velocities[index] = 0.5 * (prev_velocity + next_velocity)
    velocities[-1] = _quat_pair_angular_velocity(quats[-2], quats[-1], dt)
    return velocities


def _quat_pair_angular_velocity(quat0: np.ndarray, quat1: np.ndarray, frame_time: float) -> np.ndarray:
    delta = _quat_mul(_quat_conjugate(quat0), quat1)
    return _quat_to_expmap(delta) / max(frame_time, 1e-6)


def _motion_wrap_delta(root_pos: np.ndarray) -> np.ndarray:
    if root_pos.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)
    wrap_delta = root_pos[-1] - root_pos[0]
    wrap_delta[2] = 0.0
    return wrap_delta.astype(np.float32)


def _resolve_loop_mode(
    qpos_targets: np.ndarray,
    qvel_targets: np.ndarray,
    root_pos_targets: np.ndarray,
    root_quat_targets: np.ndarray,
    root_vel_targets: np.ndarray,
    root_angvel_targets: np.ndarray,
) -> LoopMode:
    if qpos_targets.shape[0] < 2:
        return LoopMode.CLAMP
    qpos_gap = float(np.max(np.abs(qpos_targets[0] - qpos_targets[-1])))
    qvel_gap = float(np.max(np.abs(qvel_targets[0] - qvel_targets[-1])))
    root_height_gap = float(np.abs(root_pos_targets[0, 2] - root_pos_targets[-1, 2]))
    root_quat_gap = float(np.linalg.norm(root_quat_targets[0] - root_quat_targets[-1]))
    root_vel_gap = float(np.linalg.norm(root_vel_targets[0] - root_vel_targets[-1]))
    root_angvel_gap = float(np.linalg.norm(root_angvel_targets[0] - root_angvel_targets[-1]))
    if (
        qpos_gap < 0.15
        and qvel_gap < 1.5
        and root_height_gap < 0.08
        and root_quat_gap < 0.2
        and root_vel_gap < 1.0
        and root_angvel_gap < 2.0
    ):
        return LoopMode.WRAP
    return LoopMode.CLAMP


def _batch_axis_angle_to_matrix(axis_angles: np.ndarray) -> np.ndarray:
    axis_angles = np.asarray(axis_angles, dtype=np.float32)
    thetas = np.linalg.norm(axis_angles, axis=1, keepdims=True)
    safe_thetas = np.maximum(thetas, 1e-8)
    axes = axis_angles / safe_thetas
    x = axes[:, 0:1]
    y = axes[:, 1:2]
    z = axes[:, 2:3]
    c = np.cos(thetas)
    s = np.sin(thetas)
    one_c = 1.0 - c
    matrices = np.zeros((axis_angles.shape[0], 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = (c + x * x * one_c)[:, 0]
    matrices[:, 0, 1] = (x * y * one_c - z * s)[:, 0]
    matrices[:, 0, 2] = (x * z * one_c + y * s)[:, 0]
    matrices[:, 1, 0] = (y * x * one_c + z * s)[:, 0]
    matrices[:, 1, 1] = (c + y * y * one_c)[:, 0]
    matrices[:, 1, 2] = (y * z * one_c - x * s)[:, 0]
    matrices[:, 2, 0] = (z * x * one_c - y * s)[:, 0]
    matrices[:, 2, 1] = (z * y * one_c + x * s)[:, 0]
    matrices[:, 2, 2] = (c + z * z * one_c)[:, 0]
    identity_mask = (thetas[:, 0] < 1e-8) | (~np.isfinite(thetas[:, 0]))
    matrices[identity_mask] = np.eye(3, dtype=np.float32)
    return matrices


def _batch_matrix_to_euler_xyz(matrices: np.ndarray) -> np.ndarray:
    sy = np.clip(matrices[:, 0, 2], -1.0, 1.0)
    y = np.arcsin(sy)
    cy = np.cos(y)
    x = np.where(
        np.abs(cy) > 1e-6,
        np.arctan2(-matrices[:, 1, 2], matrices[:, 2, 2]),
        np.arctan2(matrices[:, 2, 1], matrices[:, 1, 1]),
    )
    z = np.where(
        np.abs(cy) > 1e-6,
        np.arctan2(-matrices[:, 0, 1], matrices[:, 0, 0]),
        0.0,
    )
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _batch_matrix_to_quat(matrices: np.ndarray) -> np.ndarray:
    quats = np.zeros((matrices.shape[0], 4), dtype=np.float32)
    for index, matrix in enumerate(matrices):
        quats[index] = _matrix_to_quat_single(matrix)
    return quats


def _matrix_to_quat_single(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        diag = np.diag(matrix)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-8))
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-8))
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-8))
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    return _normalize_quat(np.array([w, x, y, z], dtype=np.float32))


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8 or not np.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _quat_mul(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = quat_a
    wb, xb, yb, zb = quat_b
    return _normalize_quat(
        np.array(
            [
                wa * wb - xa * xb - ya * yb - za * zb,
                wa * xb + xa * wb + ya * zb - za * yb,
                wa * yb - xa * zb + ya * wb + za * xb,
                wa * zb + xa * yb - ya * xb + za * wb,
            ],
            dtype=np.float32,
        )
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_to_expmap(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = quat[1:] / sin_half
    return (axis * angle).astype(np.float32)


def _yaw_only_quat(quat: np.ndarray) -> np.ndarray:
    rotation = _quat_to_matrix(_normalize_quat(quat))
    forward = np.array([rotation[0, 0], rotation[1, 0], 0.0], dtype=np.float32)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8 or not np.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    forward /= norm
    yaw = float(np.arctan2(forward[1], forward[0]))
    half_yaw = 0.5 * yaw
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float32)


def _continuous_quat_sequence(quats: np.ndarray) -> np.ndarray:
    result = quats.copy()
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result
