"""Visualize retargeted reference playback against MuJoCo physics.

The blue model is the simulated character driven by reference PD targets.  The
orange skeleton is the kinematic reference in the same XML space.  If the orange
motion looks plausible while the blue model falls or drifts, the reference is
loaded but not dynamically feasible for the current actuator/contact setup.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomechanics_env import BiomechanicsJoystickEnv
from config import (
    DEFAULT_SMPL_REFERENCE_FILES,
    expand_reference_gait_files,
)


BODY_LINKS = (
    ("thorax", "head"),
    ("thorax", "abdomen"),
    ("abdomen", "pelvis"),
    ("pelvis", "left_thigh"),
    ("left_thigh", "left_shank"),
    ("left_shank", "left_foot"),
    ("pelvis", "right_thigh"),
    ("right_thigh", "right_shank"),
    ("right_shank", "right_foot"),
    ("thorax", "left_upper_arm"),
    ("left_upper_arm", "left_forearm"),
    ("left_forearm", "left_hand"),
    ("thorax", "right_upper_arm"),
    ("right_upper_arm", "right_forearm"),
    ("right_forearm", "right_hand"),
)


KEY_SITES = (
    "metatarsal_midpoint_left",
    "metatarsal_midpoint_right",
    "calcaneous_left",
    "calcaneous_right",
    "head_vertex",
)


@dataclass
class ReferenceFrame:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    frame_index: int


class ViewerControls:
    def __init__(self) -> None:
        self.paused = False
        self.reset_requested = False

    def key_callback(self, keycode: int) -> None:
        if keycode == 32:  # Space
            self.paused = not self.paused
        elif keycode in (82, 114):  # R/r
            self.reset_requested = True


def project_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def make_env(args: argparse.Namespace) -> BiomechanicsJoystickEnv:
    reference_files = expand_reference_gait_files(
        args.reference_gait_file
        or [project_path(path) for path in DEFAULT_SMPL_REFERENCE_FILES],
        None,
    )
    config_overrides = {
        "impl": "jax",
        "physics_backend": "mjx_jax",
        "warp_num_worlds": 1,
        "command_profile": "forward",
        "reference_gait": "smpl",
        "reference_gait_file": reference_files,
        "reference_target_observation": False,
        "reference_action_mode": "residual",
        "reference_residual_scale": args.reference_residual_scale,
        "reference_replay_target_step": 1,
        "deepmimic_reward_mode": "pure",
        "deepmimic_key_bodies": (
            "metatarsal_midpoint_right",
            "metatarsal_midpoint_left",
        ),
        "pose_termination": False,
        "reset_sample_attempts": 1,
        "reset_projection_levels": (1.0,),
        "action_smoothing": 1.0,
    }
    if args.xml_path is not None:
        config_overrides["xml_path"] = str(args.xml_path)
    return BiomechanicsJoystickEnv(
        env_version=args.env_version,
        config_overrides=config_overrides,
    )


def enable_body_floor_collision_debug(model: mujoco.MjModel) -> None:
    """Make every body geom collide with the floor, but not with other bodies.

    The training XML intentionally keeps only the feet collidable.  That is
    efficient for locomotion, but it makes falls look confusing because the
    torso can pass through the visual floor while the soles remain constrained.
    This debug-only bitmask keeps body-body pairs disabled and enables body-floor
    pairs so failed playback is easier to read in the viewer.
    """
    floor_id = model.geom("floor").id
    model.geom_contype[floor_id] = 1
    model.geom_conaffinity[floor_id] = 2
    for geom_id in range(model.ngeom):
        if geom_id == floor_id:
            continue
        model.geom_contype[geom_id] = 2
        model.geom_conaffinity[geom_id] = 1


def host_array(value: object) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def sample_reference(
    env: BiomechanicsJoystickEnv,
    clip_id: int,
    time_s: float,
    ctrl_time_s: float | None = None,
) -> ReferenceFrame:
    frame_times = host_array(env._bvh_reference_frame_times)
    frame_counts = host_array(env._bvh_reference_frame_counts).astype(np.int32)
    qpos_targets = host_array(env._bvh_reference_qpos_targets)
    qvel_targets = host_array(env._bvh_reference_qvel_targets)
    root_pos_targets = np.asarray(env._bvh_reference_sim_root_pos_targets_np)
    root_quat_targets = np.asarray(env._bvh_reference_sim_root_quat_targets_np)
    root_vel_targets = np.asarray(env._bvh_reference_sim_root_vel_targets_np)
    root_angvel_targets = np.asarray(env._bvh_reference_sim_root_angvel_targets_np)

    frame_count = int(frame_counts[clip_id])
    frame_time = float(frame_times[clip_id])
    frame_index = int(np.clip(round(time_s / max(frame_time, 1e-6)), 0, frame_count - 1))

    full_qpos = np.asarray(env._init_q_np, dtype=np.float64).copy()
    full_qvel = np.zeros(env.mj_model.nv, dtype=np.float64)
    full_qpos[:3] = root_pos_targets[clip_id, frame_index]
    full_qpos[3:7] = root_quat_targets[clip_id, frame_index]
    full_qpos[env._actuator_qpos_indices_np] = qpos_targets[clip_id, frame_index]
    full_qvel[:3] = root_vel_targets[clip_id, frame_index]
    full_qvel[3:6] = root_angvel_targets[clip_id, frame_index]
    full_qvel[env._actuator_dof_indices_np] = qvel_targets[clip_id, frame_index]
    if ctrl_time_s is None:
        ctrl_frame_index = frame_index
    else:
        ctrl_frame_index = int(
            np.clip(round(ctrl_time_s / max(frame_time, 1e-6)), 0, frame_count - 1)
        )
    ctrl = qpos_targets[clip_id, ctrl_frame_index].astype(np.float64)
    return ReferenceFrame(qpos=full_qpos, qvel=full_qvel, ctrl=ctrl, frame_index=frame_index)


def set_data_from_reference(data: mujoco.MjData, frame: ReferenceFrame) -> None:
    data.qpos[:] = frame.qpos
    data.qvel[:] = frame.qvel
    data.ctrl[:] = frame.ctrl


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        pos.astype(np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32),
    )
    scene.ngeom += 1


def add_capsule(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        start.astype(np.float64),
        end.astype(np.float64),
    )
    geom.rgba[:] = rgba
    scene.ngeom += 1


def draw_reference_ghost(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    ref_data: mujoco.MjData,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    orange = np.array([1.0, 0.55, 0.08, 0.78], dtype=np.float32)
    yellow = np.array([1.0, 0.95, 0.15, 0.95], dtype=np.float32)

    for start_name, end_name in BODY_LINKS:
        try:
            start = ref_data.xpos[model.body(start_name).id]
            end = ref_data.xpos[model.body(end_name).id]
        except KeyError:
            continue
        add_capsule(scene, start, end, 0.012, orange)

    for body_id in range(1, model.nbody):
        add_sphere(scene, ref_data.xpos[body_id], 0.025, orange)

    for site_name in KEY_SITES:
        try:
            add_sphere(scene, ref_data.site_xpos[model.site(site_name).id], 0.045, yellow)
        except KeyError:
            continue


def copy_sim_to_view(
    model: mujoco.MjModel,
    sim_data: mujoco.MjData,
    view_data: mujoco.MjData,
    offset: np.ndarray,
) -> None:
    view_data.qpos[:] = sim_data.qpos
    view_data.qvel[:] = sim_data.qvel
    view_data.ctrl[:] = sim_data.ctrl
    view_data.qpos[:3] += offset
    mujoco.mj_forward(model, view_data)


def print_status(
    model: mujoco.MjModel,
    sim_data: mujoco.MjData,
    ref_data: mujoco.MjData,
    reference_offset: np.ndarray,
    frame_index: int,
) -> None:
    left_site = model.site("metatarsal_midpoint_left").id
    right_site = model.site("metatarsal_midpoint_right").id
    sim_left = sim_data.site_xpos[left_site]
    sim_right = sim_data.site_xpos[right_site]
    ref_left = ref_data.site_xpos[left_site] - reference_offset
    ref_right = ref_data.site_xpos[right_site] - reference_offset
    left_err = np.linalg.norm(sim_left - ref_left)
    right_err = np.linalg.norm(sim_right - ref_right)
    print(
        "frame={:04d} root_z={:.3f} ref_root_z={:.3f} "
        "left_foot_err={:.3f} right_foot_err={:.3f}".format(
            frame_index,
            float(sim_data.qpos[2]),
            float(ref_data.qpos[2]),
            float(left_err),
            float(right_err),
        ),
        flush=True,
    )


def run_viewer(args: argparse.Namespace) -> None:
    env = make_env(args)
    model = env.mj_model
    if args.debug_body_floor_collision:
        enable_body_floor_collision_debug(model)
    clip_count = int(host_array(env._bvh_reference_clip_count))
    clip_id = int(np.clip(args.clip_id, 0, clip_count - 1))
    frame_times = host_array(env._bvh_reference_frame_times)
    start_time = args.start_frame * float(frame_times[clip_id])

    sim_data = mujoco.MjData(model)
    ref_data = mujoco.MjData(model)
    view_data = mujoco.MjData(model)
    controls = ViewerControls()

    dynamic_offset = np.array([0.0, 0.75, 0.0], dtype=np.float64)
    reference_offset = np.array([0.0, -0.75, 0.0], dtype=np.float64)
    substeps = max(int(round(env.dt / model.opt.timestep)), 1)

    motion_time = start_time
    initial_frame = sample_reference(env, clip_id, motion_time)
    set_data_from_reference(sim_data, initial_frame)
    mujoco.mj_forward(model, sim_data)

    print(
        "Blue body = physics playback, orange/yellow = kinematic reference. "
        "Space pauses, R resets.",
        flush=True,
    )
    print(f"clip_id={clip_id} start_frame={args.start_frame} substeps={substeps}", flush=True)

    with mujoco.viewer.launch_passive(
        model,
        view_data,
        key_callback=controls.key_callback,
    ) as viewer:
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -18
        status_counter = 0

        while viewer.is_running():
            if controls.reset_requested:
                motion_time = start_time
                reset_frame = sample_reference(env, clip_id, motion_time)
                set_data_from_reference(sim_data, reset_frame)
                mujoco.mj_forward(model, sim_data)
                controls.reset_requested = False

            ctrl_time = motion_time + args.target_step * env.dt
            frame = sample_reference(env, clip_id, motion_time, ctrl_time)
            ref_data.qpos[:] = frame.qpos
            ref_data.qvel[:] = frame.qvel
            ref_data.ctrl[:] = frame.ctrl
            ref_data.qpos[:3] += reference_offset
            mujoco.mj_forward(model, ref_data)

            if not controls.paused:
                sim_data.ctrl[:] = frame.ctrl
                mujoco.mj_step(model, sim_data, nstep=substeps)
                motion_time += env.dt * args.speed

            copy_sim_to_view(model, sim_data, view_data, dynamic_offset)
            draw_reference_ghost(viewer, model, ref_data)
            viewer.cam.lookat[:] = np.array(
                [view_data.qpos[0], 0.0, max(view_data.qpos[2], 0.9)],
                dtype=np.float64,
            )
            viewer.sync()

            status_counter += 1
            if status_counter % 50 == 0:
                print_status(
                    model,
                    sim_data,
                    ref_data,
                    reference_offset,
                    frame.frame_index,
                )
            time.sleep(env.dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show retargeted SMPL reference ghost next to physics playback.",
    )
    parser.add_argument("--clip-id", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--env-version", choices=["standard", "hardcore"], default="standard")
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument(
        "--reference-gait-file",
        action="append",
        default=None,
        help="SMPL .npz reference file. Can be repeated. Defaults to the 4 training clips.",
    )
    parser.add_argument("--reference-residual-scale", type=float, default=0.0)
    parser.add_argument(
        "--target-step",
        type=int,
        default=1,
        help="Future reference step used as the PD target, matching training default.",
    )
    parser.add_argument(
        "--debug-body-floor-collision",
        action="store_true",
        help="Viewer-only: make all body geoms collide with the floor for readable falls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_viewer(parse_args())
