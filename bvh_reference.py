from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BvhReference:
    """Retargetovana BVH referenca u redosledu MuJoCo aktuatora."""

    qpos_targets: np.ndarray
    qvel_targets: np.ndarray
    root_height_offsets: np.ndarray
    root_forward_velocity_factors: np.ndarray
    frame_time: float
    source_path: Path


@dataclass(frozen=True)
class BvhReferenceBatch:
    """Vise retargetovanih BVH referenci padovanih na istu duzinu."""

    qpos_targets: np.ndarray
    qvel_targets: np.ndarray
    root_height_offsets: np.ndarray
    root_forward_velocity_factors: np.ndarray
    frame_times: np.ndarray
    frame_counts: np.ndarray
    source_paths: tuple[Path, ...]


def load_bvh_references(
    paths: tuple[str | Path, ...],
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> BvhReferenceBatch:
    """Ucita vise BVH referenci i pad-uje ih u jedan staticki tensor."""
    if not paths:
        raise ValueError("Potrebna je bar jedna BVH putanja.")

    references = [
        load_bvh_reference(
            path,
            actuator_joint_names,
            default_ctrl,
            lower_limits,
            upper_limits,
        )
        for path in paths
    ]
    max_frames = max(reference.qpos_targets.shape[0] for reference in references)
    action_size = references[0].qpos_targets.shape[1]
    targets = np.zeros((len(references), max_frames, action_size), dtype=np.float32)
    velocities = np.zeros(
        (len(references), max_frames, action_size),
        dtype=np.float32,
    )
    root_height_offsets = np.zeros((len(references), max_frames), dtype=np.float32)
    root_forward_velocity_factors = np.ones(
        (len(references), max_frames),
        dtype=np.float32,
    )
    frame_times = np.zeros(len(references), dtype=np.float32)
    frame_counts = np.zeros(len(references), dtype=np.int32)

    for index, reference in enumerate(references):
        frame_count = reference.qpos_targets.shape[0]
        targets[index, :frame_count] = reference.qpos_targets
        targets[index, frame_count:] = reference.qpos_targets[-1]
        velocities[index, :frame_count] = reference.qvel_targets
        velocities[index, frame_count:] = 0.0
        root_height_offsets[index, :frame_count] = reference.root_height_offsets
        root_height_offsets[index, frame_count:] = reference.root_height_offsets[-1]
        root_forward_velocity_factors[
            index,
            :frame_count,
        ] = reference.root_forward_velocity_factors
        root_forward_velocity_factors[index, frame_count:] = 1.0
        frame_times[index] = reference.frame_time
        frame_counts[index] = frame_count

    return BvhReferenceBatch(
        qpos_targets=targets,
        qvel_targets=velocities,
        root_height_offsets=root_height_offsets,
        root_forward_velocity_factors=root_forward_velocity_factors,
        frame_times=frame_times,
        frame_counts=frame_counts,
        source_paths=tuple(reference.source_path for reference in references),
    )


def load_bvh_reference(
    path: str | Path,
    actuator_joint_names: tuple[str, ...],
    default_ctrl: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> BvhReference:
    """Ucita CMU-style BVH i retargetuje osnovni hod na nas model."""
    source_path = Path(path).expanduser()
    bvh = _parse_bvh(source_path)
    targets = np.tile(np.asarray(default_ctrl, dtype=np.float32), (bvh.frames, 1))

    def assign_target(joint_name: str, values: np.ndarray) -> None:
        if joint_name not in actuator_joint_names:
            return
        index = actuator_joint_names.index(joint_name)
        targets[:, index] = np.clip(values, lower_limits[index], upper_limits[index])

    # CMU BVH koristi pozitivnu X rotaciju kolena za fleksiju, dok nas model
    # koristi negativan knee_z. Zato kolena mapiramo posebno kao fleksiju.
    assign_target(
        "left_hip_x",
        default_ctrl[actuator_joint_names.index("left_hip_x")]
        + 0.55 * _centered_rotation(bvh, "LeftHip", "Xrotation"),
    )
    assign_target(
        "right_hip_x",
        default_ctrl[actuator_joint_names.index("right_hip_x")]
        + 0.55 * _centered_rotation(bvh, "RightHip", "Xrotation"),
    )
    assign_target(
        "left_knee_z",
        default_ctrl[actuator_joint_names.index("left_knee_z")]
        - 0.75 * _positive_flexion(bvh, "LeftKnee", "Xrotation"),
    )
    assign_target(
        "right_knee_z",
        default_ctrl[actuator_joint_names.index("right_knee_z")]
        - 0.75 * _positive_flexion(bvh, "RightKnee", "Xrotation"),
    )
    assign_target(
        "left_ankle_y",
        default_ctrl[actuator_joint_names.index("left_ankle_y")]
        + 0.35 * _centered_rotation(bvh, "LeftAnkle", "Xrotation"),
    )
    assign_target(
        "right_ankle_y",
        default_ctrl[actuator_joint_names.index("right_ankle_y")]
        + 0.35 * _centered_rotation(bvh, "RightAnkle", "Xrotation"),
    )

    return BvhReference(
        qpos_targets=targets.astype(np.float32),
        qvel_targets=_target_velocities(targets, bvh.frame_time),
        root_height_offsets=_root_height_offsets(bvh),
        root_forward_velocity_factors=_root_forward_velocity_factors(bvh),
        frame_time=bvh.frame_time,
        source_path=source_path,
    )


def _target_velocities(targets: np.ndarray, frame_time: float) -> np.ndarray:
    """Izracuna reference joint brzine iz retargetovanih poza."""
    if targets.shape[0] < 2:
        return np.zeros_like(targets, dtype=np.float32)
    return np.gradient(targets, frame_time, axis=0).astype(np.float32)


@dataclass(frozen=True)
class ParsedBvh:
    """Minimalni BVH podaci potrebni za retargeting."""

    channels: tuple[tuple[str, str], ...]
    motion: np.ndarray
    frames: int
    frame_time: float


def _parse_bvh(path: Path) -> ParsedBvh:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    channels: list[tuple[str, str]] = []
    stack: list[str | None] = []
    motion_index = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "MOTION":
            motion_index = index
            break

        parts = stripped.split()
        if not parts:
            continue
        if parts[0] in {"ROOT", "JOINT"}:
            stack.append(parts[1])
        elif parts[0] == "End":
            stack.append(None)
        elif parts[0] == "}":
            if stack:
                stack.pop()
        elif parts[0] == "CHANNELS" and stack:
            joint_name = stack[-1]
            if joint_name is None:
                continue
            for channel_name in parts[2:]:
                channels.append((joint_name, channel_name))

    if motion_index is None:
        raise ValueError(f"{path} nema MOTION sekciju.")

    frames = int(lines[motion_index + 1].split(":", 1)[1])
    frame_time = float(lines[motion_index + 2].split(":", 1)[1])
    motion_lines = lines[motion_index + 3 : motion_index + 3 + frames]
    motion = np.asarray(
        [[float(value) for value in line.split()] for line in motion_lines],
        dtype=np.float32,
    )

    if motion.shape != (frames, len(channels)):
        raise ValueError(
            f"{path} ima motion shape {motion.shape}, "
            f"ali channels={len(channels)} i frames={frames}."
        )

    return ParsedBvh(
        channels=tuple(channels),
        motion=motion,
        frames=frames,
        frame_time=frame_time,
    )


def _rotation_degrees(
    bvh: ParsedBvh,
    joint_name: str,
    channel_name: str,
) -> np.ndarray:
    try:
        channel_index = bvh.channels.index((joint_name, channel_name))
    except ValueError as exc:
        raise ValueError(
            f"BVH nema kanal {joint_name}.{channel_name}."
        ) from exc
    return bvh.motion[:, channel_index]


def _centered_rotation(
    bvh: ParsedBvh,
    joint_name: str,
    channel_name: str,
) -> np.ndarray:
    degrees = _rotation_degrees(bvh, joint_name, channel_name)
    centered = degrees - np.mean(degrees)
    return np.deg2rad(centered)


def _positive_flexion(
    bvh: ParsedBvh,
    joint_name: str,
    channel_name: str,
) -> np.ndarray:
    degrees = _rotation_degrees(bvh, joint_name, channel_name)
    baseline = np.percentile(degrees, 5.0)
    return np.deg2rad(np.maximum(degrees - baseline, 0.0))


def _position_channel(
    bvh: ParsedBvh,
    joint_name: str,
    channel_name: str,
) -> np.ndarray:
    """Vrati BVH root translaciju za dati kanal."""
    try:
        channel_index = bvh.channels.index((joint_name, channel_name))
    except ValueError as exc:
        raise ValueError(
            f"BVH nema kanal {joint_name}.{channel_name}."
        ) from exc
    return bvh.motion[:, channel_index]


def _root_height_offsets(bvh: ParsedBvh) -> np.ndarray:
    """Root height oscilacija iz CMU Hips.Y, u relativnoj humanoid skali."""
    root_y = _position_channel(bvh, "Hips", "Yposition")
    centered = root_y - np.median(root_y)
    scale = _cmu_unit_to_model_meter_scale(bvh)
    return (centered * scale).astype(np.float32)


def _root_forward_velocity_factors(bvh: ParsedBvh) -> np.ndarray:
    """Normalizovan forward-speed profil iz CMU Hips.Z translacije."""
    root_z = _position_channel(bvh, "Hips", "Zposition")
    forward_velocity = np.gradient(root_z, bvh.frame_time).astype(np.float32)
    positive_velocity = forward_velocity[forward_velocity > 0.0]
    if positive_velocity.size == 0:
        return np.ones(bvh.frames, dtype=np.float32)
    nominal_velocity = float(np.mean(positive_velocity))
    if nominal_velocity < 1e-6:
        return np.ones(bvh.frames, dtype=np.float32)
    factors = forward_velocity / nominal_velocity
    return np.clip(factors, 0.0, 2.5).astype(np.float32)


def _cmu_unit_to_model_meter_scale(bvh: ParsedBvh) -> float:
    """Proceni m/BVH-unit iz root visine; CMU fajlovi nisu u MuJoCo metrima."""
    root_y = _position_channel(bvh, "Hips", "Yposition")
    median_height = float(np.median(root_y))
    if median_height <= 1e-6:
        return 0.0254
    # Nas root je oko pelvis/torso visine, ne puna ljudska visina.
    return 1.5 / median_height
