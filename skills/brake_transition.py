"""Specialized bridge from FINAL_WALKING into FINAL_STANDING's capture region."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from biomechanics_env import BiomechanicsJoystickEnv
from config import resolve_project_path
from skills.walk_stop import WalkStopEnv


class BrakeTransitionEnv(WalkStopEnv):
    """Train only the short braking maneuver, not walking or long standing.

    Resets sample shortly before the reference starts its stop blend. Unlike the
    legacy walk-stop reward, physical terms are additive and remain informative
    while the humanoid is still moving quickly.
    """

    CAPTURE_DWELL_SECONDS = 0.20

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        minimum = float(self._config.get("brake_lead_seconds_min", 0.2))
        maximum = float(self._config.get("brake_lead_seconds_max", 0.8))
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("brake_lead_seconds_min must be finite and nonnegative.")
        if not np.isfinite(maximum) or maximum < minimum:
            raise ValueError(
                "brake_lead_seconds_max must be finite and at least the minimum."
            )
        self._brake_lead_seconds_min = minimum
        self._brake_lead_seconds_max = maximum
        self._capture_dwell_steps = max(
            1, round(self.CAPTURE_DWELL_SECONDS / self.dt)
        )
        self._brake_state_bank = self._load_state_bank(
            self._config.get("brake_state_bank_file")
        )

    def _load_state_bank(self, raw_path: str | None) -> dict[str, jax.Array] | None:
        """Load and validate real FINAL_WALKING states used for resets."""
        if raw_path is None:
            return None
        path = resolve_project_path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Brake state bank is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "qpos",
                "qvel",
                "ctrl",
                "last_action",
                "motion_time",
                "clip_id",
                "dt",
                "xml_sha256",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Brake state bank missing arrays: {sorted(missing)}")
            values = {name: np.asarray(archive[name]) for name in required}

        count = values["qpos"].shape[0]
        expected_shapes = {
            "qpos": (count, self._mjx_model.nq),
            "qvel": (count, self._mjx_model.nv),
            "ctrl": (count, self.action_size),
            "last_action": (count, self.action_size),
            "motion_time": (count,),
            "clip_id": (count,),
        }
        for name, expected_shape in expected_shapes.items():
            if values[name].shape != expected_shape:
                raise ValueError(
                    f"Brake state bank {name} has shape {values[name].shape}, "
                    f"expected {expected_shape}."
                )
        if not np.isclose(float(values["dt"]), self.dt):
            raise ValueError("Brake state bank control dt does not match the env.")
        xml_hash = hashlib.sha256(Path(self.xml_path).read_bytes()).hexdigest()
        if str(values["xml_sha256"]) != xml_hash:
            raise ValueError("Brake state bank XML hash does not match the env XML.")
        if not all(np.isfinite(values[name]).all() for name in expected_shapes):
            raise ValueError("Brake state bank contains non-finite values.")

        clip_ids = values["clip_id"].astype(np.int32)
        durations = np.asarray(self._bvh_reference_motion_lengths)[clip_ids]
        stop_times = durations * self._transition_stop_phase
        phase_times = np.mod(values["motion_time"], durations)
        first_stop_delta = np.mod(stop_times - phase_times, durations)
        blend_seconds = self._transition_blend_steps * self.dt
        extra_cycles = np.maximum(
            np.ceil((blend_seconds - first_stop_delta) / durations), 0.0
        )
        lead_seconds = first_stop_delta + extra_cycles * durations - blend_seconds
        eligible = (
            (lead_seconds >= self._brake_lead_seconds_min)
            & (lead_seconds <= self._brake_lead_seconds_max)
        )
        if not np.any(eligible):
            raise ValueError("Brake state bank has no states in the configured lead window.")
        return {
            name: jp.asarray(values[name][eligible]) for name in expected_shapes
        }

    def _reset_from_state_bank(self, state: mjx_env.State) -> mjx_env.State:
        """Replace an ideal BVH reset with a real frozen-walker physical state."""
        if self._brake_state_bank is None:
            return state
        info = dict(state.info)
        info["rng"], bank_key = jax.random.split(info["rng"])
        count = self._brake_state_bank["qpos"].shape[0]
        index = jax.random.randint(bank_key, shape=(), minval=0, maxval=count)
        data = self._fresh_reset_data().replace(
            qpos=self._brake_state_bank["qpos"][index],
            qvel=self._brake_state_bank["qvel"][index],
            ctrl=self._brake_state_bank["ctrl"][index],
        )
        data = mjx.forward(self._mjx_model, data)
        info.update(
            command=jp.zeros(3, dtype=jp.float32),
            last_action=self._brake_state_bank["last_action"][index],
            motion_step=jp.array(0, dtype=jp.int32),
            command_step=jp.array(0, dtype=jp.int32),
            bvh_reference_clip_id=self._brake_state_bank["clip_id"][index].astype(
                jp.int32
            ),
            bvh_reference_time_offset=self._brake_state_bank["motion_time"][index],
            reference_fallback_standing=jp.array(False),
            init_motion_count=jp.array(1.0, dtype=jp.float32),
            init_fallback_count=jp.array(0.0, dtype=jp.float32),
            last_foot_xy=self._foot_xy(data),
        )
        state = state.replace(data=data, info=info, obs=self._get_obs(data, info))
        return self._initialize_transition(state)

    def _sample_bvh_motion_time(
        self, rng: jax.Array, clip_id: jax.Array
    ) -> jax.Array:
        """Place every reset shortly before the calibrated double-support stop."""
        duration = self._bvh_reference_motion_lengths[clip_id]
        lead_seconds = jax.random.uniform(
            rng,
            shape=(),
            minval=self._brake_lead_seconds_min,
            maxval=self._brake_lead_seconds_max,
        )
        blend_seconds = self._transition_blend_steps * self.dt
        stop_phase_time = duration * self._transition_stop_phase
        return jp.mod(stop_phase_time - blend_seconds - lead_seconds, duration)

    def _capture_terms(
        self, data: mjx.Data, info: dict
    ) -> dict[str, jax.Array]:
        feet = self._foot_xy(data)
        foot_delta = feet[1] - feet[0]
        denominator = jp.maximum(jp.dot(foot_delta, foot_delta), 1e-8)
        center_of_mass = self._center_of_mass(data)
        com_xy = center_of_mass[:2]
        com_velocity = self._center_of_mass_velocity(data)[:2]

        def distance_to_support(point: jax.Array) -> jax.Array:
            fraction = jp.clip(
                jp.dot(point - feet[0], foot_delta) / denominator, 0.0, 1.0
            )
            closest = feet[0] + fraction * foot_delta
            return jp.maximum(jp.linalg.norm(point - closest) - 0.06, 0.0)

        gravity = float(np.linalg.norm(self.mj_model.opt.gravity))
        omega = jp.sqrt(gravity / jp.maximum(center_of_mass[2], 0.25))
        capture_point = com_xy + com_velocity / jp.maximum(omega, 1e-6)
        local_angvel = self._local_root_angvel(data)
        return {
            "capture_distance": distance_to_support(capture_point),
            "support_distance": distance_to_support(com_xy),
            "angular_speed": jp.linalg.norm(local_angvel[jp.array([0, 2])]),
            "any_foot_contact": jp.any(self._foot_contact(data, info) > 0),
        }

    def _capture_ready(self, data: mjx.Data, info: dict) -> jax.Array:
        terms = self._capture_terms(data, info)
        return (
            (self._transition_alpha(info) >= 1.0)
            & (jp.linalg.norm(data.qvel[:2]) < self.thresholds.max_speed)
            & (terms["capture_distance"] < 0.08)
            & (self._torso_up(data) >= self.thresholds.min_torso_up)
            & (data.qpos[2] / self._init_q[2] >= self.thresholds.min_height_ratio)
            & terms["any_foot_contact"]
            & (~self._get_done(data, info))
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict,
    ) -> jax.Array:
        """Use broad additive shaping that has a gradient before capture."""
        imitation = BiomechanicsJoystickEnv._get_reward(
            self, data, action, previous_action, info
        )
        imitation_quality = jp.clip(imitation / self.REWARD_MAX, 0.0, 1.0)
        alpha = self._transition_alpha(info)
        terms = self._capture_terms(data, info)
        speed = jp.linalg.norm(data.qvel[:2])
        walking_reference = BiomechanicsJoystickEnv._query_bvh_reference(
            self, info, 0
        )
        walking_speed = jp.linalg.norm(walking_reference["sim_root_vel"][:2])
        desired_speed = (1.0 - alpha) * walking_speed
        speed_score = jp.exp(-jp.square((speed - desired_speed) / 0.55))
        stop_progress = jp.clip(
            1.0 - speed / jp.maximum(walking_speed, 0.5), 0.0, 1.0
        )
        capture_score = jp.exp(-jp.square(terms["capture_distance"] / 0.25))
        support_score = jp.exp(-jp.square(terms["support_distance"] / 0.16))
        angular_score = jp.exp(-jp.square(terms["angular_speed"] / 1.5))
        torso_score = jp.clip((self._torso_up(data) - 0.45) / 0.50, 0.0, 1.0)
        height_score = jp.clip(
            (data.qpos[2] / self._init_q[2] - 0.55) / 0.35, 0.0, 1.0
        )
        contact_score = terms["any_foot_contact"].astype(jp.float32)

        early_score = (
            0.45 * imitation_quality
            + 0.30 * speed_score
            + 0.15 * torso_score
            + 0.10 * height_score
        )
        braking_score = (
            0.20 * speed_score
            + 0.15 * stop_progress
            + 0.20 * capture_score
            + 0.10 * support_score
            + 0.10 * angular_score
            + 0.10 * torso_score
            + 0.10 * height_score
            + 0.05 * contact_score
        )
        score = (1.0 - alpha) * early_score + alpha * braking_score
        score += 0.25 * self._capture_ready(data, info).astype(jp.float32)
        reward = self.REWARD_MAX * score
        reward -= self.ACTION_RATE_COST_SCALE * jp.sum(
            jp.square(action - previous_action)
        )
        reward -= 0.05 * jp.clip(self._get_foot_slip_cost(data, info), 0.0, 3.0)
        terminal_penalty = jp.array(-self.REWARD_MAX, dtype=reward.dtype)
        return jp.where(
            self._get_done(data, info),
            terminal_penalty,
            jp.clip(jp.nan_to_num(reward, nan=0.0), 0.0, self.REWARD_MAX),
        )

    def _brake_metrics(self, state: mjx_env.State) -> dict[str, jax.Array]:
        terms = self._capture_terms(state.data, state.info)
        capture_steps = state.info.get(
            "brake_capture_steps", jp.array(0, dtype=jp.int32)
        )
        return {
            "brake_speed": jp.linalg.norm(state.data.qvel[:2]),
            "brake_capture_distance": terms["capture_distance"],
            "brake_support_distance": terms["support_distance"],
            "brake_angular_speed": terms["angular_speed"],
            "brake_capture_ready": self._capture_ready(
                state.data, state.info
            ).astype(jp.float32),
            "brake_capture_steps": capture_steps.astype(jp.float32),
            "brake_handoff_ready": (
                capture_steps >= self._capture_dwell_steps
            ).astype(jp.float32),
            "brake_state_bank_active": jp.array(
                self._brake_state_bank is not None, dtype=jp.float32
            ),
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        state = self._reset_from_state_bank(state)
        info = {**state.info, "brake_capture_steps": jp.array(0, dtype=jp.int32)}
        state = state.replace(info=info, obs=self._get_obs(state.data, info))
        return state.replace(metrics={**state.metrics, **self._brake_metrics(state)})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        ready = self._capture_ready(state.data, state.info)
        previous = state.info.get(
            "brake_capture_steps", jp.array(0, dtype=jp.int32)
        )
        capture_steps = jp.where(ready, previous + 1, 0)
        info = {**state.info, "brake_capture_steps": capture_steps}
        state = state.replace(info=info, obs=self._get_obs(state.data, info))
        return state.replace(metrics={**state.metrics, **self._brake_metrics(state)})
