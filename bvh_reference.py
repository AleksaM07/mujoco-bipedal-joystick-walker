import argparse
import csv
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from config import (
    BVH_INDEX_PATTERN,
    BVH_ROOT,
    BVH_TIER1_EXCLUDE,
    BVH_TIER2_HINTS,
    BVH_UNEVEN_HINTS,
    PROJECT_ROOT,
    read_reference_gait_list,
    resolve_project_path,
)

# REF: MIMICKIT-MOTION-LIBRARY
# TYPE: REFERENCE_CODE_DERIVED
"""BVH-to-motion-reference helpers.

The runtime path follows MimicKit's motion-library shape: load many clips, keep
per-clip fps/loop/frame counts, derive velocities, and expose one padded static
tensor batch for JAX/MJX.

Absolute joint targets are built in DeepMimic/MimicKit spirit: each frame is an
absolute character-DOF pose. CMU BVH is converted to MuJoCo-frame hinge angles,
then mapped by role onto this humanoid's actuators (sagittal flexion → hip_z /
knee_z). Marina's step segmentation supplies clip boundaries; it is not a
skeleton retargeter.
"""


class LoopMode(IntEnum):
    """MimicKit-compatible loop mode values."""

    CLAMP = 0
    WRAP = 1


@dataclass(frozen=True)
class MotionSegment:
    """Frame slice for one usable motion clip."""

    start_frame: int
    end_frame: int
    support_foot: str = ""


@dataclass(frozen=True)
class ParsedJoint:
    """BVH hierarchy node needed for lightweight FK and root extraction."""

    name: str
    parent: str | None
    children: tuple[str, ...]
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_indices: tuple[int, ...]


@dataclass(frozen=True)
class ParsedBvh:
    """BVH hierarchy plus dense motion channels."""

    joints: dict[str, ParsedJoint]
    channels: tuple[tuple[str, str], ...]
    motion: np.ndarray
    frames: int
    frame_time: float
    root_name: str

    def slice(self, segment: MotionSegment) -> "ParsedBvh":
        start = max(0, min(segment.start_frame, self.frames - 1))
        end = max(start + 1, min(segment.end_frame, self.frames))
        return ParsedBvh(
            joints=self.joints,
            channels=self.channels,
            motion=self.motion[start:end],
            frames=end - start,
            frame_time=self.frame_time,
            root_name=self.root_name,
        )


@dataclass(frozen=True)
class MotionClip:
    """Retargeted motion clip in the shape the JAX env can consume."""

    qpos_targets: np.ndarray
    qvel_targets: np.ndarray
    root_pos_targets: np.ndarray
    root_quat_targets: np.ndarray
    root_vel_targets: np.ndarray
    root_angvel_targets: np.ndarray
    wrap_delta: np.ndarray
    frame_time: float
    source_path: Path
    source_start_frame: int
    source_end_frame: int
    support_foot: str
    loop_mode: LoopMode = LoopMode.CLAMP
    weight: float = 1.0

    @property
    def frame_count(self) -> int:
        return int(self.qpos_targets.shape[0])


@dataclass(frozen=True)
class BvhReferenceBatch:
    """Padded motion clips ready for JAX/MJX static-shape reference tracking."""

    qpos_targets: np.ndarray
    qvel_targets: np.ndarray
    root_pos_targets: np.ndarray
    root_quat_targets: np.ndarray
    root_vel_targets: np.ndarray
    root_angvel_targets: np.ndarray
    wrap_deltas: np.ndarray
    frame_times: np.ndarray
    frame_counts: np.ndarray
    loop_modes: np.ndarray
    weights: np.ndarray
    source_paths: tuple[Path, ...]
    source_start_frames: np.ndarray
    source_end_frames: np.ndarray
    support_feet: tuple[str, ...]
    diagnostic_summaries: tuple[str, ...] = ()
    ik_target_names: tuple[str, ...] = ()
    ik_target_positions: np.ndarray | None = None


class BvhMotionLibrary:
    """Small MimicKit-style motion library specialized for this BVH source."""

    def __init__(self, clips: tuple[MotionClip, ...]) -> None:
        if not clips:
            raise ValueError("Potrebna je bar jedna validna BVH motion referenca.")
        self.clips = clips

    @classmethod
    def from_bvh_paths(
        cls,
        paths: tuple[str | Path, ...],
        actuator_joint_names: tuple[str, ...],
        default_ctrl: np.ndarray,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        initial_root_pos: np.ndarray,
        initial_root_quat: np.ndarray,
    ) -> "BvhMotionLibrary":
        clips: list[MotionClip] = []
        for source_path in expand_motion_paths(paths):
            bvh = _parse_bvh(source_path)
            segments = _motion_segments_for_bvh(source_path, bvh)
            candidate_clips = [
                _retarget_segment(
                    source_path=source_path,
                    bvh=bvh,
                    segment=segment,
                    actuator_joint_names=actuator_joint_names,
                    default_ctrl=default_ctrl,
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                    initial_root_pos=initial_root_pos,
                    initial_root_quat=initial_root_quat,
                )
                for segment in segments
            ]
            clips.extend(_select_motion_clips_for_bvh(candidate_clips))
        return cls(tuple(clips))

    def to_reference_batch(self) -> BvhReferenceBatch:
        max_frames = max(clip.frame_count for clip in self.clips)
        action_size = self.clips[0].qpos_targets.shape[1]
        clip_count = len(self.clips)

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

        for index, clip in enumerate(self.clips):
            frame_count = clip.frame_count
            qpos_targets[index, :frame_count] = clip.qpos_targets
            qpos_targets[index, frame_count:] = clip.qpos_targets[-1]
            qvel_targets[index, :frame_count] = clip.qvel_targets

            root_pos_targets[index, :frame_count] = clip.root_pos_targets
            root_pos_targets[index, frame_count:] = clip.root_pos_targets[-1]
            root_quat_targets[index, :frame_count] = clip.root_quat_targets
            root_quat_targets[index, frame_count:] = clip.root_quat_targets[-1]
            root_vel_targets[index, :frame_count] = clip.root_vel_targets
            root_angvel_targets[index, :frame_count] = clip.root_angvel_targets
            wrap_deltas[index] = clip.wrap_delta

            frame_times[index] = clip.frame_time
            frame_counts[index] = frame_count
            loop_modes[index] = int(clip.loop_mode)
            weights[index] = max(float(clip.weight), 0.0)
            source_start_frames[index] = clip.source_start_frame
            source_end_frames[index] = clip.source_end_frame

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
            source_paths=tuple(clip.source_path for clip in self.clips),
            source_start_frames=source_start_frames,
            source_end_frames=source_end_frames,
            support_feet=tuple(clip.support_foot for clip in self.clips),
        )


def load_bvh_references(
    paths: tuple[str | Path, ...],
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    initial_root_pos: np.ndarray | None = None,
    initial_root_quat: np.ndarray | None = None,
) -> BvhReferenceBatch:
    """Load BVH/list paths into one MimicKit-style padded reference batch."""
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
    library = BvhMotionLibrary.from_bvh_paths(
        paths=paths,
        actuator_joint_names=actuator_joint_names,
        default_ctrl=np.asarray(default_ctrl, dtype=np.float32),
        lower_limits=np.asarray(lower_limits, dtype=np.float32),
        upper_limits=np.asarray(upper_limits, dtype=np.float32),
        initial_root_pos=initial_root_pos,
        initial_root_quat=initial_root_quat,
    )
    return library.to_reference_batch()


def expand_motion_paths(paths: tuple[str | Path, ...]) -> tuple[Path, ...]:
    """Expand BVH paths and one-path-per-line list files."""
    expanded: list[Path] = []
    for raw_path in paths:
        path = resolve_project_path(raw_path)
        if path.suffix.lower() == ".txt":
            expanded.extend(resolve_project_path(item) for item in read_reference_gait_list(path))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("Potrebna je bar jedna BVH putanja ili lista putanja.")
    return tuple(expanded)


def _retarget_segment(
    source_path: Path,
    bvh: ParsedBvh,
    segment: MotionSegment,
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    initial_root_pos: np.ndarray,
    initial_root_quat: np.ndarray,
) -> MotionClip:
    segment_bvh = bvh.slice(segment)
    targets = np.tile(default_ctrl, (segment_bvh.frames, 1)).astype(np.float32)

    def assign_target(joint_name: str, values: np.ndarray) -> None:
        if joint_name not in actuator_joint_names:
            return
        index = actuator_joint_names.index(joint_name)
        targets[:, index] = np.clip(values, lower_limits[index], upper_limits[index])

    # REF: MIMICKIT-MOTION-LIBRARY
    # TYPE: REFERENCE_CODE_DERIVED
    # MimicKit/DeepMimic motions store absolute character DOFs, not deltas from
    # frame 0. Until GMR is wired, build absolute hinge angles from BVH local
    # rotations (MuJoCo frame), then map by ROLE onto this XML's actuators.
    #
    # Empirically, after Y-up→Z-up conversion, CMU walking flexion lives on
    # MuJoCo hinge-Y for both XYZ and ZYX BVH channel orders. This biomechanics
    # humanoid puts sagittal flexion on *_z hinges (hip_z, knee_z), so we map
    # flexion→hip_z/knee_z rather than world-axis-i → joint_*i.
    pelvis = _joint_hinge_angles(segment_bvh, ("Hips",), allow_missing=True)
    lowerback = _joint_hinge_angles(
        segment_bvh,
        ("lowerback", "Spine", "Spine1"),
        allow_missing=True,
    )
    chest = _joint_hinge_angles(
        segment_bvh,
        ("Chest", "Spine1", "Spine2"),
        allow_missing=True,
    )
    abdomen = 0.7 * lowerback + 0.3 * chest
    if not np.any(abdomen):
        abdomen = 0.5 * pelvis

    left_hip = _joint_hinge_angles(
        segment_bvh,
        ("LeftUpLeg", "LeftHip"),
        allow_missing=True,
    )
    right_hip = _joint_hinge_angles(
        segment_bvh,
        ("RightUpLeg", "RightHip"),
        allow_missing=True,
    )
    left_knee = _joint_hinge_angles(
        segment_bvh,
        ("LeftLeg", "LeftKnee"),
        allow_missing=True,
    )
    right_knee = _joint_hinge_angles(
        segment_bvh,
        ("RightLeg", "RightKnee"),
        allow_missing=True,
    )
    left_ankle = _joint_hinge_angles(
        segment_bvh,
        ("LeftFoot", "LeftAnkle"),
        allow_missing=True,
    )
    right_ankle = _joint_hinge_angles(
        segment_bvh,
        ("RightFoot", "RightAnkle"),
        allow_missing=True,
    )

    # Absolute DOF series in this character's actuator semantics.
    # Indices of hinge_angles: 0=X, 1=Y(flexion), 2=Z.
    absolute = {
        "abdomen_x": abdomen[:, 0],
        "abdomen_y": abdomen[:, 1],
        "abdomen_z": abdomen[:, 2],
        "pelvis_x": pelvis[:, 0],
        "pelvis_y": pelvis[:, 1],
        "pelvis_z": pelvis[:, 2],
        "left_hip_x": left_hip[:, 0],
        "left_hip_y": left_hip[:, 2],
        "left_hip_z": left_hip[:, 1],
        "right_hip_x": right_hip[:, 0],
        "right_hip_y": right_hip[:, 2],
        "right_hip_z": right_hip[:, 1],
        # BVH knee flexion is >=0 when bent; this XML knee_z is <=0 when bent.
        "left_knee_z": -left_knee[:, 1],
        "right_knee_z": -right_knee[:, 1],
        "left_ankle_y": left_ankle[:, 1],
        "right_ankle_y": right_ankle[:, 1],
        "left_ankle_z": left_ankle[:, 2],
        "right_ankle_z": right_ankle[:, 2],
    }

    # Bind-pose offset only: pick a standing-like frame (Marina foot-speed idea)
    # so absolute walking amplitudes stay, but rest pose matches default_ctrl.
    bind_frame = _standing_like_frame_index(
        segment_bvh,
        left_knee_flex=left_knee[:, 1],
        right_knee_flex=right_knee[:, 1],
    )
    for joint_name, values in absolute.items():
        if joint_name not in actuator_joint_names:
            continue
        index = actuator_joint_names.index(joint_name)
        bind_offset = float(default_ctrl[index] - values[bind_frame])
        assign_target(joint_name, values + bind_offset)

    root_pos, root_quat = _root_motion_targets(
        segment_bvh,
        initial_root_pos,
        initial_root_quat,
    )
    qvel_targets = _target_velocities(targets, segment_bvh.frame_time)
    root_vel_targets = _target_velocities(root_pos, segment_bvh.frame_time)
    root_angvel_targets = _quat_angular_velocities(root_quat, segment_bvh.frame_time)
    loop_mode = _resolve_loop_mode(
        qpos_targets=targets,
        qvel_targets=qvel_targets,
        root_pos_targets=root_pos,
        root_quat_targets=root_quat,
        root_vel_targets=root_vel_targets,
        root_angvel_targets=root_angvel_targets,
    )
    return MotionClip(
        qpos_targets=targets,
        qvel_targets=qvel_targets,
        root_pos_targets=root_pos,
        root_quat_targets=root_quat,
        root_vel_targets=root_vel_targets,
        root_angvel_targets=root_angvel_targets,
        wrap_delta=_motion_wrap_delta(root_pos),
        frame_time=segment_bvh.frame_time,
        source_path=source_path,
        source_start_frame=segment.start_frame,
        source_end_frame=segment.end_frame,
        support_foot=segment.support_foot,
        loop_mode=loop_mode,
        weight=max(segment_bvh.frames - 1, 1) * segment_bvh.frame_time,
    )


def _target_velocities(targets: np.ndarray, frame_time: float) -> np.ndarray:
    """Calculate frame velocities in MimicKit's derived-data spirit."""
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


def _quat_pair_angular_velocity(
    quat0: np.ndarray,
    quat1: np.ndarray,
    frame_time: float,
) -> np.ndarray:
    """Angular velocity from one quaternion interval in MuJoCo wxyz order."""
    delta = _quat_mul(_quat_conjugate(quat0), quat1)
    return _quat_to_expmap(delta) / max(frame_time, 1e-6)


def _motion_wrap_delta(root_pos: np.ndarray) -> np.ndarray:
    """Per-loop root translation offset in MimicKit's locomotion style."""
    if root_pos.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)
    wrap_delta = np.asarray(root_pos[-1] - root_pos[0], dtype=np.float32)
    wrap_delta[2] = 0.0
    return wrap_delta


def _resolve_loop_mode(
    qpos_targets: np.ndarray,
    qvel_targets: np.ndarray,
    root_pos_targets: np.ndarray,
    root_quat_targets: np.ndarray,
    root_vel_targets: np.ndarray,
    root_angvel_targets: np.ndarray,
) -> LoopMode:
    """Mark a clip WRAP only when its seam is reasonably continuous."""
    if qpos_targets.shape[0] < 3:
        return LoopMode.CLAMP

    pose_rms = float(np.sqrt(np.mean(np.square(qpos_targets[-1] - qpos_targets[0]))))
    pose_max = float(np.max(np.abs(qpos_targets[-1] - qpos_targets[0])))
    vel_rms = float(np.sqrt(np.mean(np.square(qvel_targets[-1] - qvel_targets[0]))))
    root_vel_err = float(np.linalg.norm(root_vel_targets[-1] - root_vel_targets[0]))
    root_angvel_err = float(
        np.linalg.norm(root_angvel_targets[-1] - root_angvel_targets[0])
    )
    root_height_err = float(abs(root_pos_targets[-1, 2] - root_pos_targets[0, 2]))
    root_rot_err = float(
        np.linalg.norm(
            _quat_to_expmap(
                _quat_mul(
                    _quat_conjugate(root_quat_targets[0]),
                    root_quat_targets[-1],
                )
            )
        )
    )

    seam_ok = (
        pose_rms < 0.18
        and pose_max < 0.5
        and vel_rms < 2.5
        and root_vel_err < 1.5
        and root_angvel_err < 3.0
        and root_height_err < 0.06
        and root_rot_err < 0.35
    )
    return LoopMode.WRAP if seam_ok else LoopMode.CLAMP


def _select_motion_clips_for_bvh(
    candidate_clips: list[MotionClip],
    max_wrap: int = 4,
    max_clamp: int = 2,
) -> tuple[MotionClip, ...]:
    """Rank per-source candidate clips in a more MimicKit-like locomotion spirit."""
    if not candidate_clips:
        return ()

    wrap_clips = [clip for clip in candidate_clips if clip.loop_mode == LoopMode.WRAP]
    clamp_clips = [clip for clip in candidate_clips if clip.loop_mode == LoopMode.CLAMP]

    if wrap_clips:
        ranked_wrap = sorted(
            wrap_clips,
            key=_motion_clip_rank_key,
            reverse=True,
        )
        return tuple(ranked_wrap[:max_wrap])

    ranked_clamp = sorted(
        clamp_clips,
        key=_motion_clip_rank_key,
        reverse=True,
    )
    return tuple(ranked_clamp[:max_clamp])


def _motion_clip_rank_key(clip: MotionClip) -> tuple[float, float, float, float]:
    """Prefer forward-moving, smooth, sufficiently long clips."""
    duration = max((clip.frame_count - 1) * clip.frame_time, clip.frame_time)
    horizontal_delta = clip.root_pos_targets[-1, :2] - clip.root_pos_targets[0, :2]
    travel_distance = float(np.linalg.norm(horizontal_delta))
    mean_speed = travel_distance / max(duration, 1e-6)
    seam_score = -_loop_seam_error(clip)
    height_stability = -float(
        np.max(np.abs(clip.root_pos_targets[:, 2] - clip.root_pos_targets[0, 2]))
    )
    return (
        seam_score,
        mean_speed,
        float(duration),
        height_stability,
    )


def _loop_seam_error(clip: MotionClip) -> float:
    """Aggregate seam mismatch into one score for ranking candidates."""
    pose_err = float(np.sqrt(np.mean(np.square(clip.qpos_targets[-1] - clip.qpos_targets[0]))))
    vel_err = float(np.sqrt(np.mean(np.square(clip.qvel_targets[-1] - clip.qvel_targets[0]))))
    root_vel_err = float(np.linalg.norm(clip.root_vel_targets[-1] - clip.root_vel_targets[0]))
    root_angvel_err = float(
        np.linalg.norm(clip.root_angvel_targets[-1] - clip.root_angvel_targets[0])
    )
    root_height_err = float(abs(clip.root_pos_targets[-1, 2] - clip.root_pos_targets[0, 2]))
    root_rot_err = float(
        np.linalg.norm(
            _quat_to_expmap(
                _quat_mul(
                    _quat_conjugate(clip.root_quat_targets[0]),
                    clip.root_quat_targets[-1],
                )
            )
        )
    )
    return (
        pose_err
        + 0.5 * vel_err
        + 0.5 * root_vel_err
        + 0.25 * root_angvel_err
        + 2.0 * root_height_err
        + root_rot_err
    )


def _joint_hinge_angles(
    bvh: ParsedBvh,
    joint_names: tuple[str, ...],
    allow_missing: bool = False,
) -> np.ndarray:
    """Absolute MuJoCo-frame hinge angles (X,Y,Z) for one BVH joint.

    This mirrors MimicKit/DeepMimic absolute DOF series more closely than
    frame-0 relative exp-maps: each frame is an absolute local rotation,
    converted into the MuJoCo frame, then projected onto cardinal hinge axes.
    """
    rotations = _joint_mujoco_rotations(bvh, joint_names, allow_missing=allow_missing)
    angles = np.zeros((bvh.frames, 3), dtype=np.float32)
    axes = np.eye(3, dtype=np.float32)
    for frame_index, rotation in enumerate(rotations):
        for axis_index, axis in enumerate(axes):
            angles[frame_index, axis_index] = _hinge_angle_from_matrix(rotation, axis)
    return _unwrap_angle_series(angles)


def _joint_mujoco_rotations(
    bvh: ParsedBvh,
    joint_names: tuple[str, ...],
    allow_missing: bool = False,
) -> np.ndarray:
    """Local joint rotations in the MuJoCo frame, shape (frames, 3, 3)."""
    joint_name = _first_existing_joint(bvh, joint_names)
    if joint_name is None:
        if allow_missing:
            return np.repeat(np.eye(3, dtype=np.float32)[None, :, :], bvh.frames, axis=0)
        joined = ", ".join(joint_names)
        raise ValueError(f"BVH nema nijedan trazeni joint: {joined}.")

    joint = bvh.joints[joint_name]
    rotations = np.zeros((bvh.frames, 3, 3), dtype=np.float32)
    for frame_index in range(bvh.frames):
        rotation = np.eye(3, dtype=np.float32)
        channel_values = bvh.motion[frame_index, list(joint.channel_indices)]
        for channel, value in zip(joint.channels, channel_values, strict=True):
            if channel.endswith("rotation"):
                rotation = rotation @ _axis_rotation(channel[0], float(value))
        rotations[frame_index] = _bvh_rotation_to_mujoco(rotation)
    return rotations


def _hinge_angle_from_matrix(rotation: np.ndarray, axis: np.ndarray) -> float:
    """Extract signed rotation about a known hinge axis from a rotation matrix.

    Same geometric idea MimicKit uses when converting joint rotations to hinge
    DOFs: recover θ from R ≈ Rot(axis, θ).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return 0.0
    axis = axis / axis_norm
    skew = 0.5 * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    sin_theta = float(np.dot(axis, skew))
    cos_theta = 0.5 * (float(np.trace(rotation)) - 1.0)
    return float(np.arctan2(sin_theta, cos_theta))


def _unwrap_angle_series(angles: np.ndarray) -> np.ndarray:
    """Remove 2π jumps so absolute hinge series stay continuous over a clip."""
    unwrapped = np.unwrap(angles, axis=0)
    return unwrapped.astype(np.float32)


def _standing_like_frame_index(
    bvh: ParsedBvh,
    left_knee_flex: np.ndarray,
    right_knee_flex: np.ndarray,
) -> int:
    """Pick a bind/rest frame using Marina-style low foot speed when possible.

    REF: MARINA-BVH-STEP-SEGMENTATION
    TYPE: PROJECT_COLLABORATOR_DERIVED
    """
    left_name = _first_existing_joint(bvh, ("LeftFoot", "LeftAnkle"))
    right_name = _first_existing_joint(bvh, ("RightFoot", "RightAnkle"))
    if left_name is not None and right_name is not None and bvh.frames >= 3:
        positions = _global_joint_positions(bvh)
        left_speed = _foot_speed(positions[left_name], bvh.frame_time)
        right_speed = _foot_speed(positions[right_name], bvh.frame_time)
        both_slow = left_speed + right_speed
        # Prefer double-support-ish frames with relatively straight knees.
        knee_bend = np.abs(left_knee_flex) + np.abs(right_knee_flex)
        score = both_slow + 0.25 * knee_bend
        return int(np.argmin(score))

    knee_bend = np.abs(left_knee_flex) + np.abs(right_knee_flex)
    if np.any(np.isfinite(knee_bend)):
        return int(np.argmin(knee_bend))
    return 0


def _parse_bvh(path: Path) -> ParsedBvh:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    joints: dict[str, ParsedJoint] = {}
    children_by_parent: defaultdict[str, list[str]] = defaultdict(list)
    stack: list[str | None] = []
    current_joint: str | None = None
    root_name: str | None = None
    channels: list[tuple[str, str]] = []
    channel_counter = 0
    motion_index = None

    for index, stripped in enumerate(lines):
        if stripped == "MOTION":
            motion_index = index
            break
        parts = stripped.split()
        if not parts:
            continue
        if parts[0] in {"ROOT", "JOINT"}:
            name = parts[1]
            parent = next((item for item in reversed(stack) if item is not None), None)
            if parts[0] == "ROOT":
                parent = None
                root_name = name
            if parent is not None:
                children_by_parent[parent].append(name)
            joints[name] = ParsedJoint(
                name=name,
                parent=parent,
                children=(),
                offset=np.zeros(3, dtype=np.float32),
                channels=(),
                channel_indices=(),
            )
            stack.append(name)
            current_joint = name
        elif stripped == "End Site":
            stack.append(None)
            current_joint = None
        elif parts[0] == "OFFSET" and current_joint is not None:
            joint = joints[current_joint]
            joints[current_joint] = ParsedJoint(
                name=joint.name,
                parent=joint.parent,
                children=joint.children,
                offset=np.asarray([float(value) for value in parts[1:4]], dtype=np.float32),
                channels=joint.channels,
                channel_indices=joint.channel_indices,
            )
        elif parts[0] == "CHANNELS" and current_joint is not None:
            count = int(parts[1])
            joint_channels = tuple(parts[2 : 2 + count])
            joint_indices = tuple(range(channel_counter, channel_counter + count))
            for channel_name in joint_channels:
                channels.append((current_joint, channel_name))
            joint = joints[current_joint]
            joints[current_joint] = ParsedJoint(
                name=joint.name,
                parent=joint.parent,
                children=joint.children,
                offset=joint.offset,
                channels=joint_channels,
                channel_indices=joint_indices,
            )
            channel_counter += count
        elif stripped == "}":
            if stack:
                stack.pop()
            current_joint = next((item for item in reversed(stack) if item is not None), None)

    if motion_index is None:
        raise ValueError(f"{path} nema MOTION sekciju.")
    if root_name is None:
        raise ValueError(f"{path} nema ROOT joint.")

    for name, joint in tuple(joints.items()):
        joints[name] = ParsedJoint(
            name=joint.name,
            parent=joint.parent,
            children=tuple(children_by_parent[name]),
            offset=joint.offset,
            channels=joint.channels,
            channel_indices=joint.channel_indices,
        )

    frames = int(lines[motion_index + 1].split(":", 1)[1])
    frame_time = float(lines[motion_index + 2].split(":", 1)[1])
    motion_lines = lines[motion_index + 3 : motion_index + 3 + frames]
    motion = np.asarray(
        [[float(value) for value in line.split()] for line in motion_lines if line],
        dtype=np.float32,
    )
    if motion.shape != (frames, len(channels)):
        raise ValueError(
            f"{path} ima motion shape {motion.shape}, "
            f"ali channels={len(channels)} i frames={frames}."
        )

    return ParsedBvh(
        joints=joints,
        channels=tuple(channels),
        motion=motion,
        frames=frames,
        frame_time=frame_time,
        root_name=root_name,
    )


def _motion_segments_for_bvh(path: Path, bvh: ParsedBvh) -> tuple[MotionSegment, ...]:
    """Build a MimicKit-style candidate library instead of trusting one cut source.

    Marina's CSV step boundaries are useful hints, but a locomotion motion
    library needs multiple candidate windows per source clip: full-stride
    cycles, longer same-foot cycles, and occasionally a broad fallback window.
    """
    candidates: list[MotionSegment] = []
    csv_segments = _read_step_segments(path.parent / "steps.csv", bvh.frames)
    detected = _detect_step_segments(bvh)

    if csv_segments:
        candidates.extend(csv_segments)
        candidates.extend(_promote_stride_cycles(csv_segments, bvh.frames))
        candidates.extend(_long_cycle_segments(csv_segments, bvh.frames))

    if detected:
        candidates.extend(detected)
        candidates.extend(_promote_stride_cycles(detected, bvh.frames))
        candidates.extend(_long_cycle_segments(detected, bvh.frames))

    # MimicKit-style fallback: keep a broad locomotion window too, so clip
    # ranking can still salvage useful motion when event segmentation is weak.
    full_clip = _valid_segment(0, bvh.frames, bvh.frames, "")
    if full_clip is not None:
        candidates.append(full_clip)

    deduped = _dedupe_segments(candidates)
    if deduped:
        return deduped
    return (MotionSegment(0, bvh.frames, ""),)


def _promote_stride_cycles(
    segments: tuple[MotionSegment, ...],
    frame_count: int,
) -> tuple[MotionSegment, ...]:
    """Prefer full gait cycles over half-steps when segmentation allows it.

    MimicKit-style locomotion looping works much better on full stride cycles
    (same-foot strike to next same-foot strike) than on individual half-steps.
    """
    if len(segments) < 3:
        return ()
    cycles: list[MotionSegment] = []
    for index in range(len(segments) - 2):
        first = segments[index]
        third = segments[index + 2]
        if (
            first.support_foot
            and third.support_foot
            and first.support_foot == third.support_foot
        ):
            cycle = _valid_segment(
                first.start_frame,
                third.end_frame,
                frame_count,
                first.support_foot,
            )
            if cycle is not None:
                cycles.append(cycle)
    return tuple(cycles)


def _long_cycle_segments(
    segments: tuple[MotionSegment, ...],
    frame_count: int,
) -> tuple[MotionSegment, ...]:
    """Build longer same-foot locomotion cycles when available.

    These are often more loopable than single-stride cycles because seam pose
    and root velocity line up better over a more complete gait phrase.
    """
    if len(segments) < 5:
        return ()
    cycles: list[MotionSegment] = []
    for index in range(len(segments) - 4):
        first = segments[index]
        fifth = segments[index + 4]
        if (
            first.support_foot
            and fifth.support_foot
            and first.support_foot == fifth.support_foot
        ):
            cycle = _valid_segment(
                first.start_frame,
                fifth.end_frame,
                frame_count,
                first.support_foot,
            )
            if cycle is not None:
                cycles.append(cycle)
    return tuple(cycles)


def _dedupe_segments(segments: list[MotionSegment]) -> tuple[MotionSegment, ...]:
    """Keep unique frame windows only."""
    unique: dict[tuple[int, int, str], MotionSegment] = {}
    for segment in segments:
        key = (segment.start_frame, segment.end_frame, segment.support_foot)
        unique[key] = segment
    return tuple(
        sorted(
            unique.values(),
            key=lambda segment: (segment.start_frame, segment.end_frame, segment.support_foot),
        )
    )


def _read_step_segments(path: Path, frame_count: int) -> tuple[MotionSegment, ...]:
    if not path.exists():
        return ()
    segments: list[MotionSegment] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                start = int(row["start_frame"])
                end = int(row["end_frame"]) + 1
            except (KeyError, TypeError, ValueError):
                continue
            segment = _valid_segment(
                start,
                end,
                frame_count,
                support_foot=str(row.get("support_foot", "")),
            )
            if segment is not None:
                segments.append(segment)
    return tuple(segments)


def _detect_step_segments(bvh: ParsedBvh) -> tuple[MotionSegment, ...]:
    left_name = _first_existing_joint(bvh, ("LeftFoot", "LeftAnkle"))
    right_name = _first_existing_joint(bvh, ("RightFoot", "RightAnkle"))
    if left_name is None or right_name is None:
        return ()

    positions = _global_joint_positions(bvh)
    left_speed = _foot_speed(positions[left_name], bvh.frame_time)
    right_speed = _foot_speed(positions[right_name], bvh.frame_time)
    left_contact = _median_filter_bool(left_speed < 5.0, 7)
    right_contact = _median_filter_bool(right_speed < 5.0, 7)
    left_strikes = _heel_strikes(left_contact)
    right_strikes = _heel_strikes(right_contact)

    events = [(int(frame), "L") for frame in left_strikes]
    events.extend((int(frame), "R") for frame in right_strikes)
    events.sort(key=lambda item: item[0])

    segments: list[MotionSegment] = []
    for index in range(len(events) - 1):
        start, support = events[index]
        end, _ = events[index + 1]
        segment = _valid_segment(start, end, bvh.frames, support)
        if segment is not None:
            segments.append(segment)
    return tuple(segments)


def _valid_segment(
    start: int,
    end: int,
    frame_count: int,
    support_foot: str,
) -> MotionSegment | None:
    start = max(0, min(start, frame_count - 1))
    end = max(start + 1, min(end, frame_count))
    length = end - start
    if length < 12 or length > 240:
        return None
    return MotionSegment(start, end, support_foot)


def _global_joint_positions(bvh: ParsedBvh) -> dict[str, np.ndarray]:
    positions = {
        name: np.zeros((bvh.frames, 3), dtype=np.float32)
        for name in bvh.joints
    }

    def traverse(
        joint_name: str,
        frame: int,
        parent_position: np.ndarray,
        parent_rotation: np.ndarray,
    ) -> None:
        joint = bvh.joints[joint_name]
        channel_values = bvh.motion[frame, list(joint.channel_indices)]
        translation = np.zeros(3, dtype=np.float32)
        rotation = np.eye(3, dtype=np.float32)
        for channel, value in zip(joint.channels, channel_values, strict=True):
            if channel == "Xposition":
                translation[0] = value
            elif channel == "Yposition":
                translation[1] = value
            elif channel == "Zposition":
                translation[2] = value
            elif channel.endswith("rotation"):
                rotation = rotation @ _axis_rotation(channel[0], float(value))

        if joint.parent is None:
            global_position = translation
            global_rotation = rotation
        else:
            global_position = parent_position + parent_rotation @ joint.offset
            global_rotation = parent_rotation @ rotation

        positions[joint_name][frame] = global_position
        for child in joint.children:
            traverse(child, frame, global_position, global_rotation)

    for frame in range(bvh.frames):
        traverse(bvh.root_name, frame, np.zeros(3), np.eye(3))
    return positions


def _root_motion_targets(
    bvh: ParsedBvh,
    initial_root_pos: np.ndarray,
    initial_root_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return root targets aligned to the MuJoCo character's initial frame."""
    root = bvh.joints[bvh.root_name]
    root_values = bvh.motion[:, list(root.channel_indices)]
    root_translation = np.zeros((bvh.frames, 3), dtype=np.float32)
    root_rotation = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], bvh.frames, axis=0)

    for frame_index in range(bvh.frames):
        rotation = np.eye(3, dtype=np.float32)
        for channel_index, channel in enumerate(root.channels):
            value = float(root_values[frame_index, channel_index])
            if channel == "Xposition":
                root_translation[frame_index, 0] = value
            elif channel == "Yposition":
                root_translation[frame_index, 1] = value
            elif channel == "Zposition":
                root_translation[frame_index, 2] = value
            elif channel.endswith("rotation"):
                rotation = rotation @ _axis_rotation(channel[0], value)
        root_rotation[frame_index] = rotation

    root_pos_local = _bvh_positions_to_mujoco(root_translation)
    root_pos_local -= root_pos_local[0]

    raw_root_quat = np.stack(
        [
            _matrix_to_quat(_bvh_rotation_to_mujoco(rotation))
            for rotation in root_rotation
        ],
        axis=0,
    )
    raw_root_quat = _continuous_quat_sequence(raw_root_quat)
    # Keep only heading on the free root. In CMU BVH, root pitch/roll often
    # encode upper-body lean that this MuJoCo model already expresses through
    # pelvis/abdomen joints. Copying full root tilt into the free joint makes
    # reset states fail immediately even when joint targets are otherwise good.
    raw_root_heading_quat = _continuous_quat_sequence(
        np.stack([_yaw_only_quat(quat) for quat in raw_root_quat], axis=0)
    )

    initial_root_quat = _normalize_quat(initial_root_quat)
    initial_root_heading_quat = _yaw_only_quat(initial_root_quat)
    initial_root_local_quat = _quat_mul(
        _quat_conjugate(initial_root_heading_quat),
        initial_root_quat,
    )
    alignment_quat = _heading_alignment_quat(
        initial_root_quat,
        raw_root_heading_quat[0],
    )
    alignment_rot = _quat_to_matrix(alignment_quat)
    root_pos = root_pos_local @ alignment_rot.T
    root_pos += initial_root_pos[None, :]
    root_quat = np.stack(
        [
            _quat_mul(
                _quat_mul(alignment_quat, quat),
                initial_root_local_quat,
            )
            for quat in raw_root_heading_quat
        ],
        axis=0,
    )
    return root_pos.astype(np.float32), root_quat.astype(np.float32)


def _heading_alignment_quat(
    target_root_quat: np.ndarray,
    source_root_quat: np.ndarray,
) -> np.ndarray:
    """Align only the heading/yaw component of the source root orientation."""
    target_heading = _yaw_only_quat(target_root_quat)
    source_heading = _yaw_only_quat(source_root_quat)
    return _quat_mul(target_heading, _quat_conjugate(source_heading))


def _yaw_only_quat(quat: np.ndarray) -> np.ndarray:
    """Quaternion containing only the projected heading of the given pose."""
    rotation = _quat_to_matrix(_normalize_quat(quat))
    forward = np.array([rotation[0, 0], rotation[1, 0], 0.0], dtype=np.float32)
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    forward /= forward_norm
    yaw = float(np.arctan2(forward[1], forward[0]))
    half_yaw = 0.5 * yaw
    return np.array(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float32,
    )


def _bvh_positions_to_mujoco(positions: np.ndarray) -> np.ndarray:
    # BVH is Y-up centimeters for these CMU-style files. MuJoCo is Z-up meters.
    scaled = positions.astype(np.float32) * 0.01
    return np.stack([scaled[:, 2], -scaled[:, 0], scaled[:, 1]], axis=-1)


def _bvh_rotation_to_mujoco(rotation: np.ndarray) -> np.ndarray:
    conversion = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return conversion @ rotation @ conversion.T


def _foot_speed(position: np.ndarray, frame_time: float) -> np.ndarray:
    velocity = np.gradient(position, frame_time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    return _median_filter_1d(speed, 5)


def _median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if values.shape[0] < kernel_size:
        return values
    radius = kernel_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + kernel_size]) for index in range(values.shape[0])],
        dtype=values.dtype,
    )


def _median_filter_bool(values: np.ndarray, kernel_size: int) -> np.ndarray:
    filtered = _median_filter_1d(values.astype(np.float32), kernel_size)
    return filtered >= 0.5


def _heel_strikes(contact: np.ndarray) -> np.ndarray:
    return np.where(np.diff(contact.astype(np.int32)) == 1)[0] + 1


def _first_existing_joint(bvh: ParsedBvh, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in bvh.joints), None)


def _axis_rotation(axis: str, angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    if axis == "X":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float32,
        )
    if axis == "Y":
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float32,
        )
    if axis == "Z":
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    raise ValueError(f"Nepoznata rotaciona osa: {axis}")


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float32,
        )
    else:
        diag_index = int(np.argmax(np.diag(matrix)))
        if diag_index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float32,
            )
        elif diag_index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float32,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float32,
            )
    return _normalize_quat(quat)


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _continuous_quat_sequence(quats: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so neighboring frames stay on the same hemisphere."""
    result = np.asarray(quats, dtype=np.float32).copy()
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] = -result[index]
    return result


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return _normalize_quat(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float32,
        )
    )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_to_expmap(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector_norm = float(np.linalg.norm(quat[1:]))
    if vector_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(quat[0]))
    return (quat[1:] / vector_norm * angle).astype(np.float32)


def build_walk_tiers_main() -> None:
    """Build BVH walking tier lists from the CMU index text files."""
    args = parse_walk_tier_args()
    if not BVH_ROOT.exists():
        raise FileNotFoundError(f"BVH folder not found: {BVH_ROOT}")

    descriptions = read_bvh_descriptions()
    existing_bvh = sorted(BVH_ROOT.rglob("*.bvh"))
    if not existing_bvh:
        raise ValueError(f"No BVH files found under {BVH_ROOT}")

    buckets = build_tier_buckets(existing_bvh, descriptions)

    if not args.dry_run:
        for filename, entries in buckets.items():
            write_reference_list(BVH_ROOT / filename, entries)
        write_walk_tier_summary(BVH_ROOT / "walk_tiers_summary.md", buckets)

    action = "checked" if args.dry_run else "wrote"
    print(f"{action} BVH walking tier lists")
    for filename, entries in buckets.items():
        print(f"{filename}: {len(entries)}")

    if args.check_duplicates:
        duplicate_groups = find_duplicate_bvh_files(existing_bvh)
        print_duplicate_report(duplicate_groups)


def parse_walk_tier_args() -> argparse.Namespace:
    """Parse BVH walking tier CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build BVH walking tiers and optionally audit exact duplicates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify files and print counts without writing tier lists.",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Print exact duplicate BVH files grouped by SHA-256 hash.",
    )
    return parser.parse_args()


def build_tier_buckets(
    bvh_paths: list[Path],
    descriptions: dict[str, str],
) -> dict[str, list[tuple[str, str]]]:
    """Classify BVH paths into curriculum tiers."""
    buckets = {
        "tier1_forward_walk.txt": [],
        "tier2_walk_variations.txt": [],
        "tier3_style_or_complex_walks.txt": [],
        "uneven_terrain_walks.txt": [],
    }

    for bvh_path in bvh_paths:
        description = descriptions.get(bvh_path.stem, "")
        bucket = classify_bvh_description(description)
        relative_path = bvh_path.relative_to(PROJECT_ROOT).as_posix()
        buckets[bucket].append((relative_path, description))
    return buckets


def read_bvh_descriptions() -> dict[str, str]:
    """Read motion id descriptions from every bundled CMU text index."""
    descriptions: dict[str, str] = {}
    for index_path in BVH_ROOT.rglob("cmu-mocap-index-text.txt"):
        for line in index_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            match = BVH_INDEX_PATTERN.match(line)
            if not match:
                continue
            motion_id, description = match.groups()
            descriptions.setdefault(motion_id, description.strip())
    return descriptions


def classify_bvh_description(description: str) -> str:
    """Classify a walking clip into curriculum tiers."""
    normalized = description.lower()
    words = set(re.findall(r"[a-z]+", normalized))

    if words & BVH_UNEVEN_HINTS:
        return "uneven_terrain_walks.txt"
    if is_tier1_forward_walk(normalized, words):
        return "tier1_forward_walk.txt"
    if words & BVH_TIER2_HINTS:
        return "tier2_walk_variations.txt"
    return "tier3_style_or_complex_walks.txt"


def is_tier1_forward_walk(description: str, words: set[str]) -> bool:
    """Return True for plain forward walking references."""
    if words & BVH_TIER1_EXCLUDE:
        return False
    return description in {"walk", "normal walk"} or "normal walk" in description


def write_reference_list(path: Path, entries: list[tuple[str, str]]) -> None:
    """Write one path-per-line list with descriptions as comments."""
    lines = [
        "# One BVH path per non-comment line.",
        "# Description is kept above each path for review.",
    ]
    for relative_path, description in entries:
        lines.append(f"# {description}")
        lines.append(relative_path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_walk_tier_summary(
    path: Path,
    buckets: dict[str, list[tuple[str, str]]],
) -> None:
    """Write a short human-readable summary of the tiers."""
    lines = [
        "# BVH Walking Tiers",
        "",
        "Generated from `cmu-mocap-index-text.txt` descriptions.",
        "",
        "Recommended curriculum:",
        "",
        "- Start with `tier1_forward_walk.txt` only.",
        "- After stable walking, resume with tier1 + tier2.",
        "- Keep tier3 and uneven terrain for later robustness experiments.",
        "- Do not switch tiers automatically inside the env; use separate runs.",
        "",
    ]
    for filename, entries in buckets.items():
        lines.append(f"## {filename}")
        lines.append("")
        lines.append(f"Count: {len(entries)}")
        lines.append("")
        for relative_path, description in entries[:20]:
            lines.append(f"- `{relative_path}` - {description}")
        if len(entries) > 20:
            lines.append(f"- ... {len(entries) - 20} more")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def find_duplicate_bvh_files(paths: list[Path]) -> list[list[Path]]:
    """Find exact duplicate BVH files by hashing file contents."""
    paths_by_hash: defaultdict[str, list[Path]] = defaultdict(list)
    for path in paths:
        paths_by_hash[file_sha256(path)].append(path)
    return [group for group in paths_by_hash.values() if len(group) > 1]


def file_sha256(path: Path) -> str:
    """Return a SHA-256 hash for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_duplicate_report(duplicate_groups: list[list[Path]]) -> None:
    """Print duplicate BVH groups in repo-relative form."""
    if not duplicate_groups:
        print("duplicate BVH files: none")
        return

    duplicate_file_count = sum(len(group) for group in duplicate_groups)
    extra_copy_count = duplicate_file_count - len(duplicate_groups)
    print(
        "duplicate BVH files: "
        f"{len(duplicate_groups)} groups, {extra_copy_count} removable copies"
    )
    for group_index, group in enumerate(duplicate_groups, start=1):
        print(f"duplicate group {group_index}:")
        for path in group:
            print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    build_walk_tiers_main()
