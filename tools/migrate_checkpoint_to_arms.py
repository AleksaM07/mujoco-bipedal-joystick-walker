"""Widen the verified 18-DOF walking checkpoint to the 26-DOF arms model.

The migration preserves every locomotion input, hidden weight, locomotion action
output, value-network weight, and running-normalizer statistic. New arm-specific
inputs are ignored initially, while new arm action means start at zero residual.
This gives PPO a genuine warm start instead of silently restarting from random
weights when the action and observation dimensions change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from brax.training import networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from ml_collections import config_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CHECKPOINT = (
    PROJECT_ROOT / "prezentacija/1st_success/final_model_000025681920"
)
DEFAULT_SOURCE_RUN_CONFIG = PROJECT_ROOT / "prezentacija/1st_success/config.json"
DEFAULT_TARGET_XML = (
    PROJECT_ROOT
    / "generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/arms_joystick_v1_warmstart"


@dataclass(frozen=True)
class ObservationLayout:
    """Dimensions needed to describe this project's DeepMimic observation."""

    action_size: int
    nq: int = 40
    nv: int = 39
    body_count: int = 17
    key_body_count: int = 2
    target_steps: int = 4
    gait_phase_size: int = 2

    @property
    def target_frame_size(self) -> int:
        return 2 * self.action_size + 7 + 3 * self.key_body_count

    @property
    def state_size(self) -> int:
        return (
            12
            + self.gait_phase_size
            + self.target_steps * self.target_frame_size
            + (self.nq - 7)
            + (self.nv - 6)
            + self.action_size
        )

    @property
    def gym_privileged_size(self) -> int:
        return self.body_count * (10 + 6 + 6)

    @property
    def privileged_size(self) -> int:
        return (
            self.state_size
            + 3
            + 6
            + self.gym_privileged_size
            + self.action_size
            + 6
            + 2
            + self.action_size
        )


def append_range_mapping(
    old_indices: list[int],
    new_indices: list[int],
    old_start: int,
    new_start: int,
    size: int,
) -> None:
    """Append one contiguous old-to-new index mapping."""
    old_indices.extend(range(old_start, old_start + size))
    new_indices.extend(range(new_start, new_start + size))


def state_index_mapping(
    source: ObservationLayout,
    target: ObservationLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Map all source policy fields into their semantic target positions."""
    if target.action_size < source.action_size:
        raise ValueError("Target action size must not be smaller than the source.")
    fixed_prefix_size = 12 + source.gait_phase_size
    if source.gait_phase_size != target.gait_phase_size:
        raise ValueError("Source and target gait-phase layouts differ.")
    if source.target_steps != target.target_steps:
        raise ValueError("Source and target reference lookahead counts differ.")
    if source.key_body_count != target.key_body_count:
        raise ValueError("Source and target key-body layouts differ.")
    if (source.nq, source.nv) != (target.nq, target.nv):
        raise ValueError("Source and target generalized-coordinate layouts differ.")

    old_indices: list[int] = []
    new_indices: list[int] = []
    append_range_mapping(
        old_indices,
        new_indices,
        old_start=0,
        new_start=0,
        size=fixed_prefix_size,
    )

    old_target_start = fixed_prefix_size
    new_target_start = fixed_prefix_size
    reference_tail_size = 7 + 3 * source.key_body_count
    for step in range(source.target_steps):
        old_frame = old_target_start + step * source.target_frame_size
        new_frame = new_target_start + step * target.target_frame_size
        append_range_mapping(
            old_indices,
            new_indices,
            old_frame,
            new_frame,
            source.action_size,
        )
        append_range_mapping(
            old_indices,
            new_indices,
            old_frame + source.action_size,
            new_frame + target.action_size,
            source.action_size,
        )
        append_range_mapping(
            old_indices,
            new_indices,
            old_frame + 2 * source.action_size,
            new_frame + 2 * target.action_size,
            reference_tail_size,
        )

    old_joint_start = old_target_start + source.target_steps * source.target_frame_size
    new_joint_start = new_target_start + target.target_steps * target.target_frame_size
    joint_state_size = (source.nq - 7) + (source.nv - 6)
    append_range_mapping(
        old_indices,
        new_indices,
        old_joint_start,
        new_joint_start,
        joint_state_size,
    )
    append_range_mapping(
        old_indices,
        new_indices,
        old_joint_start + joint_state_size,
        new_joint_start + joint_state_size,
        source.action_size,
    )
    return np.asarray(old_indices), np.asarray(new_indices)


def privileged_index_mapping(
    source: ObservationLayout,
    target: ObservationLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source critic fields into the widened privileged observation."""
    old_indices, new_indices = state_index_mapping(source, target)
    old_list = old_indices.tolist()
    new_list = new_indices.tolist()

    common_physics_size = 3 + 6 + source.gym_privileged_size
    if source.gym_privileged_size != target.gym_privileged_size:
        raise ValueError("Source and target body layouts differ.")
    append_range_mapping(
        old_list,
        new_list,
        source.state_size,
        target.state_size,
        common_physics_size,
    )

    old_force_start = source.state_size + common_physics_size
    new_force_start = target.state_size + common_physics_size
    append_range_mapping(
        old_list,
        new_list,
        old_force_start,
        new_force_start,
        source.action_size,
    )

    old_foot_start = old_force_start + source.action_size
    new_foot_start = new_force_start + target.action_size
    foot_and_contact_size = 6 + 2
    append_range_mapping(
        old_list,
        new_list,
        old_foot_start,
        new_foot_start,
        foot_and_contact_size,
    )
    append_range_mapping(
        old_list,
        new_list,
        old_foot_start + foot_and_contact_size,
        new_foot_start + foot_and_contact_size,
        source.action_size,
    )
    return np.asarray(old_list), np.asarray(new_list)


def widen_vector(
    source_values: np.ndarray,
    target_size: int,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Create a target vector and copy semantically matching source entries."""
    target_values = np.full(target_size, fill_value, dtype=source_values.dtype)
    target_values[target_indices] = source_values[source_indices]
    return target_values


def running_count_as_float(count: Any) -> float:
    """Convert Brax's UInt64 running count without losing its high word."""
    if hasattr(count, "hi") and hasattr(count, "lo"):
        return float(int(np.asarray(count.hi)) * 2**32 + int(np.asarray(count.lo)))
    return float(np.asarray(count))


def migrate_normalizer(
    source_stats: running_statistics.RunningStatisticsState,
    source_layout: ObservationLayout,
    target_layout: ObservationLayout,
) -> running_statistics.RunningStatisticsState:
    """Preserve old running statistics and initialize new fields neutrally."""
    mappings = {
        "state": state_index_mapping(source_layout, target_layout),
        "privileged_state": privileged_index_mapping(source_layout, target_layout),
    }
    target_sizes = {
        "state": target_layout.state_size,
        "privileged_state": target_layout.privileged_size,
    }
    count = running_count_as_float(source_stats.count)
    default_summed_variance = max(count, 1.0)
    migrated: dict[str, dict[str, np.ndarray]] = {
        "mean": {},
        "std": {},
        "summed_variance": {},
    }
    for observation_key, (source_indices, target_indices) in mappings.items():
        migrated["mean"][observation_key] = widen_vector(
            np.asarray(source_stats.mean[observation_key]),
            target_sizes[observation_key],
            source_indices,
            target_indices,
            fill_value=0.0,
        )
        migrated["std"][observation_key] = widen_vector(
            np.asarray(source_stats.std[observation_key]),
            target_sizes[observation_key],
            source_indices,
            target_indices,
            fill_value=1.0,
        )
        migrated["summed_variance"][observation_key] = widen_vector(
            np.asarray(source_stats.summed_variance[observation_key]),
            target_sizes[observation_key],
            source_indices,
            target_indices,
            fill_value=default_summed_variance,
        )

    return running_statistics.RunningStatisticsState(
        count=source_stats.count,
        mean=migrated["mean"],
        std=migrated["std"],
        summed_variance=migrated["summed_variance"],
        std_eps=source_stats.std_eps,
        mode=source_stats.mode,
    )


def migrate_policy_params(
    source_params: dict[str, Any],
    source_layout: ObservationLayout,
    target_layout: ObservationLayout,
    arm_scale_raw: float,
) -> dict[str, Any]:
    """Widen policy input/output layers while preserving locomotion behavior."""
    migrated = copy.deepcopy(source_params)
    source_layers = source_params["params"]
    target_layers = migrated["params"]
    source_indices, target_indices = state_index_mapping(
        source_layout,
        target_layout,
    )

    source_input = np.asarray(source_layers["hidden_0"]["kernel"])
    target_input = np.zeros(
        (target_layout.state_size, source_input.shape[1]),
        dtype=source_input.dtype,
    )
    target_input[target_indices] = source_input[source_indices]
    target_layers["hidden_0"]["kernel"] = target_input

    source_output_kernel = np.asarray(source_layers["hidden_2"]["kernel"])
    source_output_bias = np.asarray(source_layers["hidden_2"]["bias"])
    if source_output_bias.shape != (2 * source_layout.action_size,):
        raise ValueError("Expected a tanh-normal policy with 2*action_size outputs.")

    target_output_size = 2 * target_layout.action_size
    target_output_kernel = np.zeros(
        (source_output_kernel.shape[0], target_output_size),
        dtype=source_output_kernel.dtype,
    )
    target_output_bias = np.zeros(target_output_size, dtype=source_output_bias.dtype)
    source_mean = slice(0, source_layout.action_size)
    target_mean = slice(0, source_layout.action_size)
    source_scale = slice(source_layout.action_size, 2 * source_layout.action_size)
    target_scale = slice(
        target_layout.action_size,
        target_layout.action_size + source_layout.action_size,
    )
    target_output_kernel[:, target_mean] = source_output_kernel[:, source_mean]
    target_output_bias[target_mean] = source_output_bias[source_mean]
    target_output_kernel[:, target_scale] = source_output_kernel[:, source_scale]
    target_output_bias[target_scale] = source_output_bias[source_scale]
    arm_scale_start = target_layout.action_size + source_layout.action_size
    target_output_bias[arm_scale_start:] = arm_scale_raw
    target_layers["hidden_2"]["kernel"] = target_output_kernel
    target_layers["hidden_2"]["bias"] = target_output_bias
    return migrated


def migrate_value_params(
    source_params: dict[str, Any],
    source_layout: ObservationLayout,
    target_layout: ObservationLayout,
) -> dict[str, Any]:
    """Widen critic input while keeping its learned scalar value function."""
    migrated = copy.deepcopy(source_params)
    source_indices, target_indices = privileged_index_mapping(
        source_layout,
        target_layout,
    )
    source_input = np.asarray(source_params["params"]["hidden_0"]["kernel"])
    target_input = np.zeros(
        (target_layout.privileged_size, source_input.shape[1]),
        dtype=source_input.dtype,
    )
    target_input[target_indices] = source_input[source_indices]
    migrated["params"]["hidden_0"]["kernel"] = target_input
    return migrated


def sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_checkpoint_path(output_root: Path) -> Path:
    """Return the deterministic migrated checkpoint directory."""
    return output_root / "checkpoints/000000000000"


def validate_existing_migration(
    output_root: Path,
    source_checkpoint: Path,
) -> Path | None:
    """Reuse only an existing migration with matching recorded provenance."""
    checkpoint_path = expected_checkpoint_path(output_root)
    metadata_path = output_root / "migration.json"
    network_path = checkpoint_path / "ppo_network_config.json"
    migration_is_complete = (
        checkpoint_path.is_dir()
        and metadata_path.is_file()
        and network_path.is_file()
    )
    if not migration_is_complete:
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    network = json.loads(network_path.read_text(encoding="utf-8"))
    if Path(metadata["source_checkpoint"]).resolve() != source_checkpoint.resolve():
        raise ValueError("Existing migration was made from a different checkpoint.")
    if int(network["action_size"]) != 26:
        raise ValueError("Existing migration does not have 26 policy actions.")
    return checkpoint_path


def write_run_guards(
    output_root: Path,
    source_run_config: Path,
    target_xml: Path,
) -> None:
    """Write the config and XML identity expected by train.py resume guards."""
    run_config = json.loads(source_run_config.read_text(encoding="utf-8"))
    env_config = run_config["env"]
    env_config.update(
        {
            "arm_actuators": True,
            "command_profile": "standard_easy",
            "deepmimic_reward_mode": "mixed",
            "xml_path": target_xml.relative_to(PROJECT_ROOT).as_posix(),
        }
    )
    run_config["train"]["resume_from"] = str(DEFAULT_SOURCE_CHECKPOINT)
    run_config["train"]["run_tag"] = "arms_joystick_v1_warmstart"
    (output_root / "config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    (output_root / "xml_manifest.json").write_text(
        json.dumps(
            {
                "xml_path": target_xml.relative_to(PROJECT_ROOT).as_posix(),
                "xml_sha256": sha256(target_xml),
                "action_size": 26,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def migrate_checkpoint(
    source_checkpoint: Path,
    source_run_config: Path,
    target_xml: Path,
    output_root: Path,
    arm_scale_raw: float,
) -> Path:
    """Create or validate the deterministic 18-to-26 DOF warm start."""
    source_checkpoint = source_checkpoint.resolve()
    source_run_config = source_run_config.resolve()
    target_xml = target_xml.resolve()
    output_root = output_root.resolve()
    existing = validate_existing_migration(output_root, source_checkpoint)
    if existing is not None:
        print(existing)
        return existing
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite incomplete migration directory: {output_root}"
        )

    source_layout = ObservationLayout(action_size=18)
    target_layout = ObservationLayout(action_size=26)
    source_params = ppo_checkpoint.load(source_checkpoint)
    if len(source_params) != 3:
        raise ValueError("Expected PPO checkpoint tuple: normalizer, policy, value.")
    if np.asarray(source_params[1]["params"]["hidden_0"]["kernel"]).shape[0] != (
        source_layout.state_size
    ):
        raise ValueError("Source checkpoint policy observation layout is not 294-D.")
    if np.asarray(source_params[2]["params"]["hidden_0"]["kernel"]).shape[0] != (
        source_layout.privileged_size
    ):
        raise ValueError("Source checkpoint critic observation layout is not 721-D.")

    migrated_params = [
        migrate_normalizer(source_params[0], source_layout, target_layout),
        migrate_policy_params(
            source_params[1],
            source_layout,
            target_layout,
            arm_scale_raw,
        ),
        migrate_value_params(source_params[2], source_layout, target_layout),
    ]
    # Brax 2026's loader incorrectly indexes its initializer registry when an
    # explicitly serialized optional initializer is null. Saving accepts the
    # original JSON representation, so load it directly and retain those nulls.
    network_config_data = json.loads(
        (source_checkpoint / "ppo_network_config.json").read_text(
            encoding="utf-8"
        )
    )
    factory_config = network_config_data["network_factory_kwargs"]
    activation_name = factory_config.get("activation")
    if activation_name is not None:
        factory_config["activation"] = networks.ACTIVATION[activation_name]
    for initializer_name in (
        "policy_network_kernel_init_fn",
        "value_network_kernel_init_fn",
        "mean_kernel_init_fn",
    ):
        registered_name = factory_config.get(initializer_name)
        if registered_name is not None:
            factory_config[initializer_name] = networks.KERNEL_INITIALIZER[
                registered_name
            ]
    network_config = config_dict.ConfigDict(network_config_data)
    network_config.action_size = target_layout.action_size
    network_config.observation_size["state"]["shape"] = [target_layout.state_size]
    network_config.observation_size["privileged_state"]["shape"] = [
        target_layout.privileged_size
    ]

    output_root.mkdir(parents=True)
    checkpoint_parent = output_root / "checkpoints"
    ppo_checkpoint.save(
        checkpoint_parent,
        step=0,
        params=migrated_params,
        config=network_config,
    )
    write_run_guards(output_root, source_run_config, target_xml)
    metadata = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_metadata_sha256": sha256(
            source_checkpoint / "_CHECKPOINT_METADATA"
        ),
        "source_action_size": source_layout.action_size,
        "target_action_size": target_layout.action_size,
        "source_state_size": source_layout.state_size,
        "target_state_size": target_layout.state_size,
        "source_privileged_size": source_layout.privileged_size,
        "target_privileged_size": target_layout.privileged_size,
        "new_arm_action_mean": 0.0,
        "new_arm_action_scale_raw": arm_scale_raw,
        "target_xml": str(target_xml),
        "target_xml_sha256": sha256(target_xml),
    }
    (output_root / "migration.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    checkpoint_path = expected_checkpoint_path(output_root)
    print(checkpoint_path)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    """Parse checkpoint migration arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT,
    )
    parser.add_argument(
        "--source-run-config",
        type=Path,
        default=DEFAULT_SOURCE_RUN_CONFIG,
    )
    parser.add_argument("--target-xml", type=Path, default=DEFAULT_TARGET_XML)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--arm-scale-raw",
        type=float,
        default=-3.0,
        help="Raw softplus scale for the eight new arm action distributions.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the first-walk checkpoint migration."""
    args = parse_args()
    migrate_checkpoint(
        source_checkpoint=args.source_checkpoint,
        source_run_config=args.source_run_config,
        target_xml=args.target_xml,
        output_root=args.output_root,
        arm_scale_raw=args.arm_scale_raw,
    )


if __name__ == "__main__":
    main()
