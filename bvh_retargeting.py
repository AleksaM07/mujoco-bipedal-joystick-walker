"""Host-only BVH retargeting ported from cd6557c, independent of RL rewards.

Uses the actual main-branch MuJoCo skeleton. No reference-state reset,
DeepMimic reward, residual actions, or PPO settings are imported here.
"""

from dataclasses import replace
from itertools import product
from typing import Any

import mujoco
import numpy as np

from biomechanics_model import LEG_ACTUATED_JOINTS, TRUNK_ACTUATED_JOINTS
from bvh_reference import BvhReferenceBatch, LoopMode, load_bvh_references
from config import resolve_project_path

ARM_ACTUATED_JOINTS = (
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",
    "left_elbow_z",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z",
    "right_elbow_z",
)


class BvhReferenceRetargeter:
    """Adapt the modern motion pipeline to a supplied MuJoCo model."""

    FOOT_CONTACT_PRELOAD = 0.005
    FOOT_CONTACT_HEIGHT = 0.05
    MIN_STANDING_HEIGHT_RATIO = 0.6
    REFERENCE_RETARGET_ROOT_XY_SCALE = 0.35
    REFERENCE_ROOT_HEIGHT_MAX_SPEED = 0.75

    def __init__(
        self,
        model: mujoco.MjModel,
        initial_qpos: np.ndarray,
        config: Any,
    ) -> None:
        self._mj_model = model
        self._init_q_np = np.asarray(initial_qpos, dtype=np.float64).copy()
        self._config = config
        joint_ids = model.actuator_trnid[:, 0]
        self._actuator_joint_names = tuple(
            model.joint(int(joint_id)).name for joint_id in joint_ids
        )
        self._actuator_qpos_indices_np = model.jnt_qposadr[joint_ids]
        self._actuator_dof_indices_np = model.jnt_dofadr[joint_ids]
        self._default_ctrl = self._init_q_np[self._actuator_qpos_indices_np]
        joint_ranges = model.jnt_range[joint_ids]
        limited = model.actuator_ctrllimited.astype(bool)
        self._actuator_ctrl_lower_limits_np = np.where(
            limited,
            np.maximum(joint_ranges[:, 0], model.actuator_ctrlrange[:, 0]),
            joint_ranges[:, 0],
        )
        self._actuator_ctrl_upper_limits_np = np.where(
            limited,
            np.minimum(joint_ranges[:, 1], model.actuator_ctrlrange[:, 1]),
            joint_ranges[:, 1],
        )
        self._foot_geom_ids_np = np.array(
            [model.geom(name).id for name in ("left_foot_sole", "right_foot_sole")]
        )
        self._torso_body_id = model.body("thorax").id

    def load(self, paths: tuple[str, ...]) -> BvhReferenceBatch:
        """Load, segment, retarget, floor-align and differentiate BVH clips."""
        references = load_bvh_references(
            tuple(resolve_project_path(path) for path in paths),
            self._actuator_joint_names,
            self._default_ctrl,
            self._actuator_ctrl_lower_limits_np,
            self._actuator_ctrl_upper_limits_np,
            initial_root_pos=self._init_q_np[:3],
            initial_root_quat=self._init_q_np[3:7],
        )
        return self.retarget(references)

    def retarget(self, references: BvhReferenceBatch) -> BvhReferenceBatch:
        """Run the reference-only stages from the latest experimental branch."""
        references = self._filter_reference_clips(references)
        references = self._apply_reference_loop_mode_override(references)
        motion_speed_scale = float(
            self._config.get("reference_motion_speed_scale", 1.0)
        )
        if not np.isfinite(motion_speed_scale) or motion_speed_scale <= 0.0:
            raise ValueError(
                "reference_motion_speed_scale must be finite and positive."
            )
        reference_frame_times = (
            np.asarray(references.frame_times, dtype=np.float32) / motion_speed_scale
        )
        root_angvel_targets = (
            np.asarray(
                references.root_angvel_targets,
                dtype=np.float32,
            )
            * motion_speed_scale
        )
        (
            sim_root_pos_targets,
            sim_root_vel_targets,
            sim_wrap_deltas,
        ) = self._align_reference_roots_to_sim_floor(
            references.qpos_targets,
            references.root_pos_targets,
            references.root_quat_targets,
            reference_frame_times,
            references.frame_counts,
            references.loop_modes,
        )
        qpos_targets = np.asarray(references.qpos_targets, dtype=np.float32)
        qvel_targets = np.asarray(references.qvel_targets, dtype=np.float32)
        ik_target_positions = getattr(references, "ik_target_positions", None)
        if ik_target_positions is not None:
            root_delta = sim_root_pos_targets - references.root_pos_targets
            aligned_ik_targets = (
                np.asarray(ik_target_positions, dtype=np.float32)
                + root_delta[:, :, None, :]
            )
            aligned_ik_targets = self._normalize_ik_foot_target_heights(
                aligned_ik_targets,
                tuple(getattr(references, "ik_target_names", ())),
                references.frame_counts,
            )
            stability_before_ik = bool(
                self._config.get("reference_stability_before_ik", False)
            )
            if stability_before_ik:
                # Applying the stability blend after IK destroys the foot
                # positions that IK just fitted, especially swing clearance at
                # low sagittal alpha.  Use the stable pose as the IK seed so
                # feet remain constrained while the rest of the body stays
                # close to the safer prior.
                qpos_targets = self._apply_reference_stability_prior(qpos_targets)
            ik_seed_targets = qpos_targets.copy()
            unscaled_root_targets = sim_root_pos_targets.copy()
            sim_root_pos_targets = self._scale_reference_root_xy_motion(
                sim_root_pos_targets,
                references.frame_counts,
            )
            root_scale_delta = sim_root_pos_targets - unscaled_root_targets
            aligned_ik_targets += root_scale_delta[:, :, None, :]
            ik_seed_root_targets = sim_root_pos_targets.copy()
            qpos_targets, sim_root_pos_targets = (
                self._retarget_reference_qpos_to_ik_targets(
                    qpos_targets,
                    aligned_ik_targets,
                    tuple(getattr(references, "ik_target_names", ())),
                    sim_root_pos_targets,
                    references.root_quat_targets,
                    references.frame_counts,
                )
            )
            if stability_before_ik:
                ik_correction_alpha = float(
                    self._config.get("reference_ik_correction_alpha", 1.0)
                )
                qpos_targets = ik_seed_targets + np.clip(
                    ik_correction_alpha,
                    0.0,
                    1.0,
                ) * (qpos_targets - ik_seed_targets)
                sim_root_pos_targets = ik_seed_root_targets + np.clip(
                    ik_correction_alpha,
                    0.0,
                    1.0,
                ) * (sim_root_pos_targets - ik_seed_root_targets)
            if not stability_before_ik:
                qpos_targets = self._apply_reference_stability_prior(qpos_targets)
            qvel_targets = self._reference_qvel_from_qpos_targets(
                qpos_targets,
                reference_frame_times,
                references.frame_counts,
                references.loop_modes,
            )
            (
                sim_root_pos_targets,
                sim_root_vel_targets,
                sim_wrap_deltas,
            ) = self._align_reference_roots_to_sim_floor(
                qpos_targets,
                sim_root_pos_targets,
                references.root_quat_targets,
                reference_frame_times,
                references.frame_counts,
                references.loop_modes,
            )
        else:
            qpos_targets = self._apply_reference_stability_prior(qpos_targets)
            qvel_targets = self._reference_qvel_from_qpos_targets(
                qpos_targets,
                reference_frame_times,
                references.frame_counts,
                references.loop_modes,
            )
            sim_root_pos_targets = self._scale_reference_root_xy_motion(
                sim_root_pos_targets,
                references.frame_counts,
            )
            (
                sim_root_pos_targets,
                sim_root_vel_targets,
                sim_wrap_deltas,
            ) = self._align_reference_roots_to_sim_floor(
                qpos_targets,
                sim_root_pos_targets,
                references.root_quat_targets,
                reference_frame_times,
                references.frame_counts,
                references.loop_modes,
            )
        return replace(
            references,
            qpos_targets=qpos_targets,
            qvel_targets=qvel_targets,
            root_pos_targets=sim_root_pos_targets,
            root_vel_targets=sim_root_vel_targets,
            root_angvel_targets=root_angvel_targets,
            wrap_deltas=sim_wrap_deltas,
            frame_times=reference_frame_times,
        )

    def _retarget_reference_qpos_to_ik_targets(
        self,
        qpos_targets: np.ndarray,
        ik_target_positions: np.ndarray,
        ik_target_names: tuple[str, ...],
        root_pos_targets: np.ndarray,
        root_quat_targets: np.ndarray,
        frame_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit reference actuator poses to SMPL marker targets in this XML space.

        This is the missing DeepMimic/MimicKit-style retarget step: SMPL marker
        positions are solved through the actual MuJoCo skeleton and joint limits
        once during reference loading, instead of expecting raw SMPL local angles
        to be valid controls for a different XML.
        """
        marker_specs = self._resolve_host_ik_markers(ik_target_names)
        if not marker_specs:
            return (
                np.asarray(qpos_targets, dtype=np.float32),
                np.asarray(root_pos_targets, dtype=np.float32),
            )

        fitted = np.asarray(qpos_targets, dtype=np.float32).copy()
        fitted_root = np.asarray(root_pos_targets, dtype=np.float32).copy()
        seed_targets = np.asarray(qpos_targets, dtype=np.float64)
        ik_target_positions = np.asarray(ik_target_positions, dtype=np.float64)
        root_pos_targets = np.asarray(root_pos_targets, dtype=np.float64)
        root_quat_targets = np.asarray(root_quat_targets, dtype=np.float64)
        frame_counts = np.asarray(frame_counts, dtype=np.int32)

        data = mujoco.MjData(self._mj_model)
        action_size = seed_targets.shape[-1]
        lower = self._actuator_ctrl_lower_limits_np.astype(np.float64)
        upper = self._actuator_ctrl_upper_limits_np.astype(np.float64)
        ik_update_mask = np.array(
            [
                joint_name in LEG_ACTUATED_JOINTS or joint_name in ARM_ACTUATED_JOINTS
                for joint_name in self._actuator_joint_names
            ],
            dtype=np.float64,
        )
        pose_reg = float(self._config.get("reference_ik_pose_regularization", 0.20))
        smooth_reg = float(self._config.get("reference_ik_smooth_regularization", 0.08))
        configured_root_smooth_reg = self._config.get(
            "reference_ik_root_smooth_regularization",
            None,
        )
        root_smooth_reg = (
            smooth_reg
            if configured_root_smooth_reg is None
            else float(configured_root_smooth_reg)
        )
        com_weight = float(self._config.get("reference_ik_com_weight", 0.0))
        support_height_scale = max(
            float(self._config.get("reference_ik_support_height_scale", 0.03)),
            1e-6,
        )
        optimize_root_xy = bool(
            self._config.get("reference_ik_optimize_root_xy", False)
        )
        root_reg = float(self._config.get("reference_ik_root_regularization", 0.20))
        swing_foot_weight_scale = np.clip(
            float(self._config.get("reference_ik_swing_foot_weight_scale", 1.0)),
            0.0,
            1.0,
        )
        damping = 1e-5
        max_update = 0.12
        max_iters = 10
        balance_update_mask = np.array(
            [
                joint_name in TRUNK_ACTUATED_JOINTS or joint_name in LEG_ACTUATED_JOINTS
                for joint_name in self._actuator_joint_names
            ],
            dtype=np.float64,
        )

        neutral_data = mujoco.MjData(self._mj_model)
        neutral_data.qpos[:] = self._init_q_np
        neutral_data.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, neutral_data)
        neutral_foot_positions = np.asarray(
            [
                self._host_marker_position(neutral_data, marker)
                for marker in marker_specs
                if bool(marker["is_foot"])
            ],
            dtype=np.float64,
        )
        neutral_com_offset = np.zeros(2, dtype=np.float64)
        if len(neutral_foot_positions) > 0:
            neutral_com = self._host_whole_body_com(neutral_data)
            neutral_com_offset = neutral_com[:2] - np.mean(
                neutral_foot_positions[:, :2],
                axis=0,
            )

        for clip_id in range(seed_targets.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            previous_qpos: np.ndarray | None = None
            previous_root_xy: np.ndarray | None = None
            for frame_id in range(frame_count):
                seed = seed_targets[clip_id, frame_id].copy()
                root_seed = root_pos_targets[clip_id, frame_id].copy()
                foot_targets = np.asarray(
                    [
                        ik_target_positions[
                            clip_id,
                            frame_id,
                            int(marker["target_index"]),
                        ]
                        for marker in marker_specs
                        if bool(marker["is_foot"])
                    ],
                    dtype=np.float64,
                )
                desired_com_xy: np.ndarray | None = None
                support_weight_by_target: dict[int, float] = {}
                if com_weight > 0.0 and len(foot_targets) > 0:
                    relative_heights = foot_targets[:, 2] - np.min(foot_targets[:, 2])
                    support_weights = np.exp(-relative_heights / support_height_scale)
                    support_weights /= max(float(np.sum(support_weights)), 1e-9)
                    support_index = 0
                    for marker in marker_specs:
                        if not bool(marker["is_foot"]):
                            continue
                        support_weight_by_target[int(marker["target_index"])] = float(
                            support_weights[support_index]
                        )
                        support_index += 1
                    desired_support_xy = np.sum(
                        support_weights[:, None] * foot_targets[:, :2],
                        axis=0,
                    )
                    desired_com_xy = desired_support_xy + neutral_com_offset
                qpos = (
                    seed.copy()
                    if previous_qpos is None
                    else (0.75 * seed + 0.25 * previous_qpos)
                )
                root_xy = (
                    root_seed[:2].copy()
                    if previous_root_xy is None
                    else 0.75 * root_seed[:2] + 0.25 * previous_root_xy
                )
                qpos = np.clip(qpos, lower, upper)
                for _ in range(max_iters):
                    full_qpos = self._init_q_np.copy()
                    full_qpos[:3] = root_seed
                    full_qpos[:2] = root_xy
                    full_qpos[3:7] = root_quat_targets[clip_id, frame_id]
                    full_qpos[self._actuator_qpos_indices_np] = qpos
                    data.qpos[:] = full_qpos
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(self._mj_model, data)

                    residuals: list[np.ndarray] = []
                    jacobian_rows: list[np.ndarray] = []
                    for marker in marker_specs:
                        target_pos = ik_target_positions[
                            clip_id,
                            frame_id,
                            int(marker["target_index"]),
                        ].copy()
                        if marker["is_foot"]:
                            target_pos[2] = max(
                                target_pos[2],
                                float(self.FOOT_CONTACT_HEIGHT),
                            )
                        current_pos = self._host_marker_position(data, marker)
                        marker_weight = float(marker["weight"])
                        if bool(marker["is_foot"]) and support_weight_by_target:
                            support_weight = support_weight_by_target[
                                int(marker["target_index"])
                            ]
                            maximum_support_weight = max(
                                support_weight_by_target.values()
                            )
                            stance_fraction = support_weight / max(
                                maximum_support_weight,
                                1e-9,
                            )
                            marker_weight *= (
                                swing_foot_weight_scale
                                + (1.0 - swing_foot_weight_scale) * stance_fraction
                            )
                        sqrt_weight = np.sqrt(marker_weight)
                        residuals.append(sqrt_weight * (target_pos - current_pos))
                        marker_jac = self._host_marker_jacobian(data, marker)
                        marker_jac = marker_jac * ik_update_mask[None, :]
                        if optimize_root_xy:
                            root_jacobian = np.zeros((3, 2), dtype=np.float64)
                            root_jacobian[:2] = np.eye(2, dtype=np.float64)
                            marker_jac = np.hstack((root_jacobian, marker_jac))
                        jacobian_rows.append(sqrt_weight * marker_jac)

                    if desired_com_xy is not None:
                        current_com = self._host_whole_body_com(data)
                        com_jacobian = self._host_whole_body_com_jacobian(data)
                        sqrt_com_weight = np.sqrt(com_weight)
                        residuals.append(
                            sqrt_com_weight * (desired_com_xy - current_com[:2])
                        )
                        com_jacobian = com_jacobian[:2] * balance_update_mask[None, :]
                        if optimize_root_xy:
                            com_jacobian = np.hstack(
                                (np.eye(2, dtype=np.float64), com_jacobian)
                            )
                        jacobian_rows.append(sqrt_com_weight * com_jacobian)

                    sqrt_pose_reg = np.sqrt(pose_reg)
                    residuals.append(sqrt_pose_reg * (seed - qpos))
                    pose_jacobian = np.eye(action_size, dtype=np.float64)
                    if optimize_root_xy:
                        pose_jacobian = np.hstack(
                            (
                                np.zeros((action_size, 2), dtype=np.float64),
                                pose_jacobian,
                            )
                        )
                    jacobian_rows.append(sqrt_pose_reg * pose_jacobian)
                    if optimize_root_xy and root_reg > 0.0:
                        sqrt_root_reg = np.sqrt(root_reg)
                        residuals.append(sqrt_root_reg * (root_seed[:2] - root_xy))
                        root_reg_jacobian = np.hstack(
                            (
                                np.eye(2, dtype=np.float64),
                                np.zeros((2, action_size), dtype=np.float64),
                            )
                        )
                        jacobian_rows.append(sqrt_root_reg * root_reg_jacobian)
                    if previous_qpos is not None:
                        sqrt_smooth_reg = np.sqrt(smooth_reg)
                        residuals.append(sqrt_smooth_reg * (previous_qpos - qpos))
                        smooth_jacobian = np.eye(action_size, dtype=np.float64)
                        if optimize_root_xy:
                            smooth_jacobian = np.hstack(
                                (
                                    np.zeros((action_size, 2), dtype=np.float64),
                                    smooth_jacobian,
                                )
                            )
                        jacobian_rows.append(sqrt_smooth_reg * smooth_jacobian)
                    if optimize_root_xy and previous_root_xy is not None:
                        sqrt_smooth_reg = np.sqrt(root_smooth_reg)
                        residuals.append(sqrt_smooth_reg * (previous_root_xy - root_xy))
                        root_smooth_jacobian = np.hstack(
                            (
                                np.eye(2, dtype=np.float64),
                                np.zeros((2, action_size), dtype=np.float64),
                            )
                        )
                        jacobian_rows.append(sqrt_smooth_reg * root_smooth_jacobian)

                    residual = np.concatenate(residuals)
                    jacobian = np.vstack(jacobian_rows)
                    system = jacobian.T @ jacobian
                    rhs = jacobian.T @ residual
                    system += damping * np.eye(system.shape[0])
                    try:
                        update = np.linalg.solve(system, rhs)
                    except np.linalg.LinAlgError:
                        update = np.linalg.lstsq(system, rhs, rcond=None)[0]
                    if optimize_root_xy:
                        root_update = np.clip(update[:2], -0.03, 0.03)
                        joint_update = np.clip(
                            update[2:],
                            -max_update,
                            max_update,
                        )
                        root_xy += root_update
                    else:
                        joint_update = np.clip(update, -max_update, max_update)
                    qpos = np.clip(qpos + joint_update, lower, upper)
                    if float(np.linalg.norm(update)) < 1e-4:
                        break

                fitted[clip_id, frame_id] = qpos.astype(np.float32)
                fitted_root[clip_id, frame_id, :2] = root_xy.astype(np.float32)
                previous_qpos = qpos.copy()
                previous_root_xy = root_xy.copy()
            fitted[clip_id, frame_count:] = fitted[clip_id, frame_count - 1]
            fitted_root[clip_id, frame_count:] = fitted_root[
                clip_id,
                frame_count - 1,
            ]

        return fitted, fitted_root

    def _normalize_ik_foot_target_heights(
        self,
        ik_target_positions: np.ndarray,
        ik_target_names: tuple[str, ...],
        frame_counts: np.ndarray,
    ) -> np.ndarray:
        """Shift SMPL foot marker Z so each frame has a stance contact target."""
        foot_indices = [
            index
            for index, marker_name in enumerate(ik_target_names)
            if "metatarsal" in marker_name or "foot" in marker_name
        ]
        if not foot_indices:
            return ik_target_positions

        normalized = np.asarray(ik_target_positions, dtype=np.float32).copy()
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        stance_site_height = self._standing_marker_height(
            ik_target_names[foot_indices[0]]
        )
        for clip_id in range(normalized.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            foot_z = normalized[clip_id, :frame_count, foot_indices, 2]
            per_frame_offset = stance_site_height - np.min(foot_z, axis=1)
            normalized[clip_id, :frame_count, foot_indices, 2] += per_frame_offset[
                :, None
            ]
            normalized[clip_id, frame_count:] = normalized[clip_id, frame_count - 1]
        return normalized

    def _standing_marker_height(self, marker_name: str) -> float:
        """Return marker height in the simulator's neutral floor-aligned pose."""
        data = mujoco.MjData(self._mj_model)
        data.qpos[:] = self._init_q_np
        data.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, data)
        try:
            return float(data.site_xpos[self._mj_model.site(marker_name).id, 2])
        except KeyError:
            try:
                return float(data.xpos[self._mj_model.body(marker_name).id, 2])
            except KeyError:
                return float(self.FOOT_CONTACT_HEIGHT)

    def _apply_reference_stability_prior(
        self,
        qpos_targets: np.ndarray,
    ) -> np.ndarray:
        """Blend raw SMPL targets toward a dynamically safer locomotion pose.

        Full-amplitude SMPL joint targets can place this XML's COM far outside
        the support foot.  Keep the gait signal, but damp the raw pose enough
        that feed-forward reference playback starts from load-bearing contacts.
        """
        qpos_targets = np.asarray(qpos_targets, dtype=np.float32)
        neutral = np.asarray(self._default_ctrl, dtype=np.float32)
        sagittal_motion_joints = {
            "left_hip_x",
            "right_hip_x",
            "left_hip_z",
            "right_hip_z",
            "left_knee_z",
            "right_knee_z",
            "left_ankle_y",
            "right_ankle_y",
        }
        alpha = np.array(
            [
                float(
                    self._config.get(
                        "reference_stability_sagittal_alpha",
                        0.40,
                    )
                )
                if joint_name in sagittal_motion_joints
                else float(
                    self._config.get(
                        "reference_stability_arm_alpha",
                        0.75,
                    )
                )
                if joint_name in ARM_ACTUATED_JOINTS
                else float(
                    self._config.get(
                        "reference_stability_other_alpha",
                        0.18,
                    )
                )
                for joint_name in self._actuator_joint_names
            ],
            dtype=np.float32,
        )
        return neutral[None, None, :] + alpha[None, None, :] * (
            qpos_targets - neutral[None, None, :]
        )

    def _resolve_host_ik_markers(
        self,
        marker_names: tuple[str, ...],
    ) -> list[dict[str, object]]:
        """Resolve marker names to host MuJoCo site/body ids for IK."""
        specs: list[dict[str, object]] = []
        for target_index, marker_name in enumerate(marker_names):
            is_foot = "metatarsal" in marker_name or "foot" in marker_name
            weight = 4.0 if is_foot else 1.2
            if "pelvis" in marker_name:
                weight = 2.0
            elif "hand" in marker_name:
                weight = 0.15
            elif "head" in marker_name:
                weight = 0.7
            try:
                specs.append(
                    {
                        "name": marker_name,
                        "kind": "site",
                        "id": self._mj_model.site(marker_name).id,
                        "weight": weight,
                        "is_foot": is_foot,
                        "target_index": target_index,
                    }
                )
                continue
            except KeyError:
                pass
            try:
                specs.append(
                    {
                        "name": marker_name,
                        "kind": "body",
                        "id": self._mj_model.body(marker_name).id,
                        "weight": weight,
                        "is_foot": is_foot,
                        "target_index": target_index,
                    }
                )
            except KeyError:
                continue
        return specs

    def _host_whole_body_com(self, data: mujoco.MjData) -> np.ndarray:
        """Return the mass-weighted host-model COM in world coordinates."""
        body_masses = np.asarray(self._mj_model.body_mass[1:], dtype=np.float64)
        total_mass = max(float(np.sum(body_masses)), 1e-9)
        return (
            np.sum(
                np.asarray(data.xipos[1:], dtype=np.float64) * body_masses[:, None],
                axis=0,
            )
            / total_mass
        )

    def _host_whole_body_com_jacobian(self, data: mujoco.MjData) -> np.ndarray:
        """Return whole-body COM Jacobian columns for actuated hinge DOFs."""
        jacobian = np.zeros((3, self._mj_model.nv), dtype=np.float64)
        mujoco.mj_jacSubtreeCom(
            self._mj_model,
            data,
            jacobian,
            self._torso_body_id,
        )
        return jacobian[:, self._actuator_dof_indices_np]

    def _host_marker_position(
        self,
        data: mujoco.MjData,
        marker: dict[str, object],
    ) -> np.ndarray:
        """Return one host marker position for a resolved IK spec."""
        marker_id = int(marker["id"])
        if marker["kind"] == "site":
            return np.asarray(data.site_xpos[marker_id], dtype=np.float64)
        return np.asarray(data.xpos[marker_id], dtype=np.float64)

    def _host_marker_jacobian(
        self,
        data: mujoco.MjData,
        marker: dict[str, object],
    ) -> np.ndarray:
        """Return marker position Jacobian columns for actuated hinge DOFs."""
        jacp = np.zeros((3, self._mj_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._mj_model.nv), dtype=np.float64)
        marker_id = int(marker["id"])
        if marker["kind"] == "site":
            mujoco.mj_jacSite(self._mj_model, data, jacp, jacr, marker_id)
        else:
            mujoco.mj_jacBody(self._mj_model, data, jacp, jacr, marker_id)
        return jacp[:, self._actuator_dof_indices_np]

    def _reference_qvel_from_qpos_targets(
        self,
        qpos_targets: np.ndarray,
        frame_times: np.ndarray,
        frame_counts: np.ndarray,
        loop_modes: np.ndarray,
    ) -> np.ndarray:
        """Differentiate actuator targets without a velocity jump at wrap seams."""
        qpos_targets = np.asarray(qpos_targets, dtype=np.float32)
        frame_times = np.asarray(frame_times, dtype=np.float32)
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        loop_modes = np.asarray(loop_modes, dtype=np.int32)
        qvel_targets = np.zeros_like(qpos_targets, dtype=np.float32)
        for clip_id in range(qpos_targets.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 1:
                continue
            dt = max(float(frame_times[clip_id]), 1e-6)
            qvel_targets[clip_id, :frame_count] = np.gradient(
                qpos_targets[clip_id, :frame_count],
                dt,
                axis=0,
            ).astype(np.float32)
            if loop_modes[clip_id] == int(LoopMode.WRAP) and frame_count > 2:
                seam_velocity = (
                    qpos_targets[clip_id, 1] - qpos_targets[clip_id, frame_count - 2]
                ) / (2.0 * dt)
                qvel_targets[clip_id, 0] = seam_velocity
                qvel_targets[clip_id, frame_count - 1] = seam_velocity
            qvel_targets[clip_id, frame_count:] = qvel_targets[
                clip_id,
                frame_count - 1,
            ]
        return qvel_targets

    def _scale_reference_root_xy_motion(
        self,
        root_pos_targets: np.ndarray,
        frame_counts: np.ndarray,
    ) -> np.ndarray:
        """Keep SMPL root travel consistent with the damped retargeted legs.

        After IK we intentionally blend joint targets toward a stable locomotion
        prior.  Leaving the original full-speed SMPL chest/root translation in
        place creates a treadmill reference: the ghost root moves forward while
        the physically simulated feet remain planted.  Scale only horizontal
        displacement from each clip's first frame; Z is recomputed by the normal
        floor-alignment pass.
        """
        scaled = np.asarray(root_pos_targets, dtype=np.float32).copy()
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        root_xy_scale = float(
            self._config.get(
                "reference_root_xy_scale",
                self.REFERENCE_RETARGET_ROOT_XY_SCALE,
            )
        )
        configured_x_scale = self._config.get("reference_root_x_scale", None)
        configured_y_scale = self._config.get("reference_root_y_scale", None)
        root_x_scale = (
            root_xy_scale if configured_x_scale is None else float(configured_x_scale)
        )
        root_y_scale = (
            root_xy_scale if configured_y_scale is None else float(configured_y_scale)
        )
        root_scale = np.array([root_x_scale, root_y_scale], dtype=np.float32)
        for clip_id in range(scaled.shape[0]):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            origin_xy = scaled[clip_id, 0, :2].copy()
            scaled[clip_id, :frame_count, :2] = origin_xy + root_scale * (
                scaled[clip_id, :frame_count, :2] - origin_xy
            )
            scaled[clip_id, frame_count:, :2] = scaled[clip_id, frame_count - 1, :2]
        return scaled

    def _align_reference_roots_to_sim_floor(
        self,
        qpos_targets: np.ndarray,
        root_pos_targets: np.ndarray,
        root_quat_targets: np.ndarray,
        frame_times: np.ndarray,
        frame_counts: np.ndarray,
        loop_modes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bake reference root Z into the same floor-aligned MuJoCo space as reset.

        SMPL/BVH root translations can be globally plausible while still placing
        this XML's foot geoms above or below the MuJoCo floor. MimicKit avoids
        that by storing motions in the simulator character space. This pass does
        the lightweight equivalent for our current retargeted actuator targets:
        run host MuJoCo FK per frame, shift the root so the lowest foot center is
        at the same contact-ready height used by reset, then derive root velocity
        from the baked root trajectory.
        """
        aligned_root_pos = np.asarray(root_pos_targets, dtype=np.float32).copy()
        root_quat_targets = np.asarray(root_quat_targets, dtype=np.float32)
        qpos_targets = np.asarray(qpos_targets, dtype=np.float32)
        frame_counts = np.asarray(frame_counts, dtype=np.int32)
        frame_times = np.asarray(frame_times, dtype=np.float32)
        loop_modes = np.asarray(loop_modes, dtype=np.int32)

        data = mujoco.MjData(self._mj_model)
        standing_root_height = float(np.asarray(self._mj_model.qpos0[2]))
        minimum_reset_height = max(
            standing_root_height * 0.82,
            standing_root_height * float(self.MIN_STANDING_HEIGHT_RATIO) + 0.05,
        )
        desired_lowest_height = -float(self.FOOT_CONTACT_PRELOAD)

        clip_count, max_frames, _ = qpos_targets.shape
        for clip_id in range(clip_count):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            for frame_id in range(max_frames):
                source_frame = min(frame_id, frame_count - 1)
                full_qpos = self._init_q_np.copy()
                full_qpos[:3] = aligned_root_pos[clip_id, source_frame]
                full_qpos[3:7] = root_quat_targets[clip_id, source_frame]
                full_qpos[self._actuator_qpos_indices_np] = qpos_targets[
                    clip_id,
                    source_frame,
                ]
                data.qpos[:] = full_qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(self._mj_model, data)

                foot_heights = np.array(
                    [
                        self._geom_min_z(data, int(geom_id))
                        for geom_id in self._foot_geom_ids_np
                    ],
                    dtype=np.float32,
                )
                z_offset = desired_lowest_height - float(np.min(foot_heights))
                aligned_z = max(float(full_qpos[2]) + z_offset, minimum_reset_height)
                aligned_root_pos[clip_id, frame_id] = aligned_root_pos[
                    clip_id,
                    source_frame,
                ]
                aligned_root_pos[clip_id, frame_id, 2] = aligned_z

        aligned_root_vel = np.zeros_like(aligned_root_pos, dtype=np.float32)
        wrap_deltas = np.zeros((clip_count, 3), dtype=np.float32)
        for clip_id in range(clip_count):
            frame_count = int(frame_counts[clip_id])
            if frame_count <= 0:
                continue
            dt = max(float(frame_times[clip_id]), 1e-6)
            max_root_z_step = float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED) * dt
            root_z = aligned_root_pos[clip_id, :frame_count, 2].copy()
            for frame_id in range(1, frame_count):
                delta_z = np.clip(
                    root_z[frame_id] - root_z[frame_id - 1],
                    -max_root_z_step,
                    max_root_z_step,
                )
                root_z[frame_id] = root_z[frame_id - 1] + delta_z
            for frame_id in range(frame_count - 2, -1, -1):
                delta_z = np.clip(
                    root_z[frame_id] - root_z[frame_id + 1],
                    -max_root_z_step,
                    max_root_z_step,
                )
                root_z[frame_id] = root_z[frame_id + 1] + delta_z
            root_z = np.maximum(root_z, minimum_reset_height)
            aligned_root_pos[clip_id, :frame_count, 2] = root_z
            aligned_root_pos[clip_id, frame_count:, 2] = root_z[-1]
            if bool(self._config.get("reference_lock_stance_feet", True)):
                aligned_root_pos[clip_id] = self._lock_clip_root_xy_to_stance_feet(
                    qpos_targets[clip_id],
                    aligned_root_pos[clip_id],
                    root_quat_targets[clip_id],
                    frame_count,
                )
            if frame_count > 1:
                aligned_root_vel[clip_id, :frame_count] = np.gradient(
                    aligned_root_pos[clip_id, :frame_count],
                    dt,
                    axis=0,
                ).astype(np.float32)
                aligned_root_vel[clip_id, :frame_count, 2] = np.clip(
                    aligned_root_vel[clip_id, :frame_count, 2],
                    -float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED),
                    float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED),
                )
                aligned_root_vel[clip_id, frame_count:] = aligned_root_vel[
                    clip_id,
                    frame_count - 1,
                ]
            if loop_modes[clip_id] == int(LoopMode.WRAP):
                wrap_deltas[clip_id] = (
                    aligned_root_pos[clip_id, frame_count - 1]
                    - aligned_root_pos[clip_id, 0]
                )
                wrap_deltas[clip_id, 2] = 0.0
                if frame_count > 2:
                    previous_position = (
                        aligned_root_pos[clip_id, frame_count - 2]
                        - wrap_deltas[clip_id]
                    )
                    seam_velocity = (
                        aligned_root_pos[clip_id, 1] - previous_position
                    ) / (2.0 * dt)
                    seam_velocity[2] = np.clip(
                        seam_velocity[2],
                        -float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED),
                        float(self.REFERENCE_ROOT_HEIGHT_MAX_SPEED),
                    )
                    aligned_root_vel[clip_id, 0] = seam_velocity
                    aligned_root_vel[clip_id, frame_count - 1] = seam_velocity
                    aligned_root_vel[clip_id, frame_count:] = seam_velocity

        return aligned_root_pos, aligned_root_vel, wrap_deltas

    def _lock_clip_root_xy_to_stance_feet(
        self,
        qpos_targets: np.ndarray,
        root_pos_targets: np.ndarray,
        root_quat_targets: np.ndarray,
        frame_count: int,
    ) -> np.ndarray:
        """Reduce kinematic foot skating by anchoring the lowest contact foot."""
        locked_root = np.asarray(root_pos_targets, dtype=np.float32).copy()
        if frame_count <= 1:
            return locked_root

        data = mujoco.MjData(self._mj_model)
        contact_height = float(self._config.get("reference_foot_lock_height", 0.035))
        xy_offset = np.zeros(2, dtype=np.float32)
        active_foot: int | None = None
        active_anchor = np.zeros(2, dtype=np.float32)

        for frame_id in range(frame_count):
            full_qpos = self._init_q_np.copy()
            full_qpos[:3] = locked_root[frame_id]
            full_qpos[:2] += xy_offset
            full_qpos[3:7] = root_quat_targets[frame_id]
            full_qpos[self._actuator_qpos_indices_np] = qpos_targets[frame_id]
            data.qpos[:] = full_qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(self._mj_model, data)

            foot_min_z = np.array(
                [
                    self._geom_min_z(data, int(geom_id))
                    for geom_id in self._foot_geom_ids_np
                ],
                dtype=np.float32,
            )
            contact_feet = foot_min_z <= contact_height
            if not np.any(contact_feet):
                active_foot = None
                locked_root[frame_id, :2] += xy_offset
                continue

            foot_xy = np.asarray(
                data.geom_xpos[self._foot_geom_ids_np, :2],
                dtype=np.float32,
            )
            if active_foot is None or not bool(contact_feet[active_foot]):
                active_foot = int(np.argmin(np.where(contact_feet, foot_min_z, np.inf)))
                active_anchor = foot_xy[active_foot].copy()
            else:
                xy_offset += active_anchor - foot_xy[active_foot]
                full_qpos[:2] = locked_root[frame_id, :2] + xy_offset
                data.qpos[:] = full_qpos
                mujoco.mj_forward(self._mj_model, data)
                foot_xy = np.asarray(
                    data.geom_xpos[self._foot_geom_ids_np, :2],
                    dtype=np.float32,
                )
                active_anchor = foot_xy[active_foot].copy()

            locked_root[frame_id, :2] = full_qpos[:2]

        locked_root[frame_count:] = locked_root[frame_count - 1]
        return locked_root

    def _geom_min_z(self, data: mujoco.MjData, geom_id: int) -> float:
        """Proceni donju Z tacku za foot geom, uz ispravan capsule tretman."""
        geom_pos = data.geom_xpos[geom_id]
        geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self._mj_model.geom_size[geom_id]
        geom_type = self._mj_model.geom_type[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            min_z = np.inf
            for signs in product((-1.0, 1.0), repeat=3):
                corner = geom_pos + geom_xmat @ (geom_size * np.array(signs))
                min_z = min(min_z, corner[2])
            return float(min_z)

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_pos[2] - geom_size[0])

        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = geom_size[0]
            half_length = geom_size[1]
            axis = geom_xmat[:, 2]
            endpoint_a = geom_pos + axis * half_length
            endpoint_b = geom_pos - axis * half_length
            return float(min(endpoint_a[2], endpoint_b[2]) - radius)

        # Conservative fallback for simple geoms that may appear in generated feet.
        return float(geom_pos[2] - np.max(geom_size))

    def _filter_reference_clips(
        self,
        references: BvhReferenceBatch,
    ) -> BvhReferenceBatch:
        """Filter out pathological short clips from the training reference batch."""
        min_motion_length = float(self._config.get("reference_min_motion_length", 0.0))
        if min_motion_length <= 0.0:
            return references

        frame_counts = np.asarray(references.frame_counts, dtype=np.int32)
        motion_lengths = np.asarray(
            references.frame_times, dtype=np.float32
        ) * np.maximum(
            frame_counts - 1,
            1,
        )
        keep = motion_lengths >= min_motion_length
        if not np.any(keep):
            return references

        weights = np.asarray(references.weights[keep], dtype=np.float32)
        weight_sum = float(weights.sum())
        if weight_sum > 0.0:
            weights /= weight_sum
        else:
            weights[:] = 1.0 / float(weights.shape[0])

        keep_indices = np.flatnonzero(keep)
        diagnostic_summaries = (
            tuple(references.diagnostic_summaries[index] for index in keep_indices)
            if len(references.diagnostic_summaries) == len(frame_counts)
            else (references.diagnostic_summaries)
        )
        ik_target_positions = (
            references.ik_target_positions[keep]
            if references.ik_target_positions is not None
            else None
        )
        return replace(
            references,
            qpos_targets=references.qpos_targets[keep],
            qvel_targets=references.qvel_targets[keep],
            root_pos_targets=references.root_pos_targets[keep],
            root_quat_targets=references.root_quat_targets[keep],
            root_vel_targets=references.root_vel_targets[keep],
            root_angvel_targets=references.root_angvel_targets[keep],
            wrap_deltas=references.wrap_deltas[keep],
            frame_times=references.frame_times[keep],
            frame_counts=references.frame_counts[keep],
            loop_modes=references.loop_modes[keep],
            weights=weights,
            source_paths=tuple(
                references.source_paths[index] for index in keep_indices
            ),
            source_start_frames=references.source_start_frames[keep],
            source_end_frames=references.source_end_frames[keep],
            support_feet=tuple(
                references.support_feet[index] for index in keep_indices
            ),
            diagnostic_summaries=diagnostic_summaries,
            ik_target_positions=ik_target_positions,
        )

    def _apply_reference_loop_mode_override(
        self,
        references: BvhReferenceBatch,
    ) -> BvhReferenceBatch:
        """Optionally force finite/looping playback for inspected references."""
        mode = str(self._config.get("reference_loop_mode", "auto")).lower()
        if mode == "auto":
            return references
        if mode not in ("wrap", "clamp"):
            raise ValueError(
                "reference_loop_mode mora biti 'auto', 'wrap' ili 'clamp', "
                f"dobijeno: {mode!r}."
            )

        loop_value = int(LoopMode.WRAP if mode == "wrap" else LoopMode.CLAMP)
        loop_modes = np.full_like(
            np.asarray(references.loop_modes, dtype=np.int32),
            loop_value,
        )
        return replace(references, loop_modes=loop_modes)
