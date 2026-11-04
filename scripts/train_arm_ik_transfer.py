"""Transfer the frozen 26-DOF walker to a full elbow-and-wrist IK reference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT, TrainConfig
from evaluate import find_run_config, restore_saved_env_config
from train import configure_stdout_encoding, run_training


DEFAULT_SOURCE = (
    PROJECT_ROOT / "prezentacija/2nd_success_stiff_arms/model_000010035200"
)


def main() -> None:
    """Fine-tune only the saved walk's arm reference and arm reward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, choices=[32, 64, 128, 256], default=64)
    parser.add_argument("--arm-reward-weight", type=float, default=1.0)
    parser.add_argument("--run-tag", default="arms_ik_v3_elbow_wrist_10m")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive.")
    if not 0.0 < args.arm_reward_weight <= 4.0:
        parser.error("--arm-reward-weight must be in (0, 4].")

    configure_stdout_encoding()
    jax.config.update("jax_default_device", jax.devices("gpu")[0])
    source = args.source.resolve(strict=True)
    saved = find_run_config(source)
    env = restore_saved_env_config(saved)
    if not env.arm_actuators:
        raise ValueError("Arm IK transfer requires the archived 26-DOF arms model.")
    env.deepmimic_reward_mode = "pure"
    env.command_profile = "walk"
    env.arm_reference_mode = "cmu_elbow_wrist_ik"
    env.deepmimic_arm_reward_weight = args.arm_reward_weight
    # The previous transfer attenuated every arm target by 25%, even after IK.
    # Keep the locomotion stability prior unchanged but preserve the full arm
    # motion now that the reference constrains the complete kinematic chain.
    env.reference_stability_arm_alpha = 1.0

    training = TrainConfig(**saved["train"])
    training.num_timesteps = args.timesteps
    training.num_envs = args.num_envs
    training.num_eval_envs = 4
    training.num_evals = 11
    training.learning_rate = 5e-5
    training.resume_from = str(source)
    training.checkpoint_out = None
    training.run_tag = args.run_tag
    result = run_training(env, training, PROJECT_ROOT / "runs")
    print(f"arm IK transfer complete: {result}", flush=True)


if __name__ == "__main__":
    main()
