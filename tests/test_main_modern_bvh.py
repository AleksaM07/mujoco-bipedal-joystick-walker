"""Regression tests for the deliberately narrow main + modern BVH merge."""

import ast
import subprocess
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jp
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv
from bvh_reference import LoopMode
from bvh_retargeting import BvhReferenceRetargeter
from config import EnvConfig
from train import make_environment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = "33eaa1f"


@lru_cache
def main_source(filename: str) -> str:
    """Read the immutable main baseline without changing the working tree."""
    return subprocess.check_output(
        ["git", "show", f"{MAIN_SOURCE}:{filename}"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    )


def environment_methods(source: str) -> dict[str, ast.FunctionDef]:
    """Return AST methods of the locomotion class."""
    env_class = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "BiomechanicsJoystickEnv"
    )
    return {
        node.name: node for node in env_class.body if isinstance(node, ast.FunctionDef)
    }


def test_main_reward_control_and_model_are_preserved() -> None:
    baseline = environment_methods(main_source("biomechanics_env.py"))
    current = environment_methods(
        (PROJECT_ROOT / "biomechanics_env.py").read_text(encoding="utf-8")
    )
    reference_adaptation = {
        "__init__",
        "_configure_bvh_reference",
        "reset",
        "step",
        "_get_bvh_reference_phase_angle",
        "_get_bvh_reference_gait_target",
        "_get_bvh_reference_velocity_target",
    }
    for name, method in baseline.items():
        if name not in reference_adaptation:
            assert ast.dump(current[name]) == ast.dump(method), name
    # Step differs only by the new motion clock. All main action/control and
    # reward ordering must otherwise match exactly.
    current_step = current["step"]
    current_step.body = [
        node
        for node in current_step.body
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "motion_step"
        )
    ]
    assert ast.dump(current_step) == ast.dump(baseline["step"])
    for node in ast.walk(current["reset"]):
        if isinstance(node, ast.Dict):
            kept = [
                (key, value)
                for key, value in zip(node.keys, node.values)
                if not (isinstance(key, ast.Constant) and key.value == "motion_step")
            ]
            node.keys = [key for key, _ in kept]
            node.values = [value for _, value in kept]
    assert ast.dump(current["reset"]) == ast.dump(baseline["reset"])
    assert ast.dump(
        ast.parse((PROJECT_ROOT / "biomechanics_model.py").read_text(encoding="utf-8"))
    ) == ast.dump(ast.parse(main_source("biomechanics_model.py")))


def test_ppo_defaults_are_identical_to_main() -> None:
    def ppo_function(source: str) -> ast.FunctionDef:
        return next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "default_biomechanics_ppo_config"
        )

    assert ast.dump(
        ppo_function((PROJECT_ROOT / "config.py").read_text(encoding="utf-8"))
    ) == ast.dump(ppo_function(main_source("config.py")))


def sampling_env(loop_mode: LoopMode = LoopMode.WRAP) -> BiomechanicsJoystickEnv:
    """Make a reference-sampling fixture without compiling physics."""
    env = object.__new__(BiomechanicsJoystickEnv)
    env._config = SimpleNamespace(reference_pipeline="modern")
    env._ctrl_dt = 0.02
    env._bvh_reference_frame_times = jp.array([0.04])
    env._bvh_reference_frame_counts = jp.array([4])
    env._bvh_reference_loop_modes = jp.array([int(loop_mode)])
    env._bvh_reference_qpos_targets = jp.array([[[0.0], [1.0], [2.0], [3.0]]])
    env._bvh_reference_qvel_targets = jp.ones((1, 4, 1)) * 25.0
    return env


def test_motion_clock_does_not_restart_with_command_clock() -> None:
    env = sampling_env()
    info = {
        "bvh_reference_clip_id": jp.array(0),
        "motion_step": jp.array(5),
        "step": jp.array(501),
    }
    before = env._get_bvh_reference_gait_target(info)
    info["step"] = jp.array(0)
    np.testing.assert_allclose(env._get_bvh_reference_gait_target(info), before)
    np.testing.assert_allclose(before, [2.5])


def test_clamp_holds_pose_and_stops_velocity() -> None:
    env = sampling_env(LoopMode.CLAMP)
    info = {"bvh_reference_clip_id": jp.array(0), "motion_step": jp.array(100)}
    np.testing.assert_allclose(env._get_bvh_reference_gait_target(info), [3.0])
    np.testing.assert_allclose(env._get_bvh_reference_velocity_target(info), [0.0])


def test_velocities_follow_the_actual_motion_speed() -> None:
    retargeter = object.__new__(BvhReferenceRetargeter)
    targets = np.array([[[0.0], [0.1], [0.2], [0.3]]], dtype=np.float32)
    counts = np.array([4])
    loops = np.array([int(LoopMode.CLAMP)])
    normal = retargeter._reference_qvel_from_qpos_targets(
        targets, np.array([0.1]), counts, loops
    )
    slow = retargeter._reference_qvel_from_qpos_targets(
        targets, np.array([0.4]), counts, loops
    )
    np.testing.assert_allclose(normal, 1.0, atol=1e-6)
    np.testing.assert_allclose(slow, normal * 0.25, atol=1e-6)


def test_modern_bvh_loads_on_the_unchanged_main_model() -> None:
    env = make_environment(
        EnvConfig(
            reference_gait="bvh",
            reference_pipeline="modern",
            reference_gait_file=(
                "BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"
            ),
            reference_motion_speed_scale=1.0,
        ),
        enable_erfi=False,
    )
    reference = env._bvh_reference_batch
    retargeter = BvhReferenceRetargeter(
        env.mj_model, np.asarray(env._init_q), env._config
    )
    assert "trainfast_v17" in env._xml_path.name
    assert reference.qpos_targets.shape[-1] == 18
    assert np.isfinite(reference.qpos_targets).all()
    assert np.isfinite(reference.qvel_targets).all()
    assert (reference.frame_times > 0).all()
    assert (
        reference.qpos_targets >= (retargeter._actuator_ctrl_lower_limits_np - 1e-6)
    ).all()
    assert (
        reference.qpos_targets <= (retargeter._actuator_ctrl_upper_limits_np + 1e-6)
    ).all()
    for clip_id, count in enumerate(reference.frame_counts):
        gradient = np.gradient(
            reference.qpos_targets[clip_id, :count],
            reference.frame_times[clip_id],
            axis=0,
        )
        np.testing.assert_allclose(
            reference.qvel_targets[clip_id, 1 : count - 1], gradient[1:-1], atol=1e-5
        )


def test_legacy_bvh_loader_is_frozen() -> None:
    current = ast.parse(
        (PROJECT_ROOT / "bvh_reference_main.py").read_text(encoding="utf-8")
    )
    current.body.pop(0)  # Only the provenance docstring is new.
    assert ast.dump(current) == ast.dump(ast.parse(main_source("bvh_reference.py")))
