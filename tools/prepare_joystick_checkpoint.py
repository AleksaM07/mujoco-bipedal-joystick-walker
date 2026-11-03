"""Prepare a pure 26-DOF tracker for command conditioning without an input shock.

Pure imitation observed zero commands throughout training. Those input weights
were never trained, and their running standard deviation is close to zero. Start
command learning with zero input weights and unit command standard deviation,
while preserving every other weight and statistic in a separate checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from brax.training import networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from ml_collections import config_dict


COMMAND_CHANNELS = slice(9, 12)
OBSERVATION_SIZES = {"state": 366, "privileged_state": 809}


def prepare_command_inputs(source_params: list[Any]) -> list[Any]:
    """Preserve pure-tracker outputs while making unseen commands initially inert.

    Accept only this project's 26-action, dictionary-observation pure tracker.
    Normalizer count and all non-command values remain unchanged. Unit command
    variance includes the stored epsilon so subsequent Welford updates agree
    with the saved standard deviation.
    """
    if len(source_params) != 3:
        raise ValueError("Expected normalizer, policy and value checkpoint entries.")
    source_stats = source_params[0]
    if source_stats.mode != running_statistics.NormalizationMode.WELFORD:
        raise ValueError("Only Welford running statistics are supported.")
    count = source_stats.count
    if hasattr(count, "hi") and hasattr(count, "lo"):
        count_float = float(int(np.asarray(count.hi)) * 2**32 + int(np.asarray(count.lo)))
    else:
        count_float = float(np.asarray(count))
    epsilon = float(source_stats.std_eps)
    if not np.isfinite(count_float) or count_float <= 0:
        raise ValueError("Source normalizer must have a positive sample count.")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("Source standard-deviation epsilon must be in [0, 1).")
    if set(source_stats.mean) != set(OBSERVATION_SIZES):
        raise ValueError("Expected state and privileged_state observation statistics.")

    prepared = copy.deepcopy(list(source_params))
    means = {}
    deviations = {}
    variances = {}
    for parameter_index, (key, expected_size) in enumerate(
        OBSERVATION_SIZES.items(), start=1
    ):
        mean = np.array(source_stats.mean[key], copy=True)
        deviation = np.array(source_stats.std[key], copy=True)
        variance = np.array(source_stats.summed_variance[key], copy=True)
        if any(value.shape != (expected_size,) for value in (mean, deviation, variance)):
            raise ValueError(f"Unexpected observation layout for {key}.")
        if not all(np.isfinite(value).all() for value in (mean, deviation, variance)):
            raise ValueError(f"Non-finite normalizer values for {key}.")
        if np.any(deviation <= 0):
            raise ValueError(f"Non-positive normalizer standard deviation for {key}.")
        if np.any(mean[COMMAND_CHANNELS] != 0.0):
            raise ValueError("Source commands must have exactly zero recorded means.")
        if np.any(variance[COMMAND_CHANNELS] != 0.0):
            raise ValueError("Source commands must have zero recorded variation.")

        layer = prepared[parameter_index]["params"]["hidden_0"]
        kernel = np.array(layer["kernel"], copy=True)
        if kernel.ndim != 2 or kernel.shape[0] != expected_size:
            raise ValueError(f"Unexpected first-layer input shape for {key}.")
        if not np.isfinite(kernel).all():
            raise ValueError(f"Non-finite first-layer weights for {key}.")
        kernel[COMMAND_CHANNELS] = 0.0
        layer["kernel"] = kernel
        deviation[COMMAND_CHANNELS] = 1.0
        variance[COMMAND_CHANNELS] = count_float * (1.0 - epsilon)
        means[key] = mean
        deviations[key] = deviation
        variances[key] = variance

    policy_output = source_params[1]["params"]["hidden_2"]["bias"]
    if np.asarray(policy_output).shape != (52,):
        raise ValueError("Expected 26-action tanh-normal policy outputs.")
    prepared[0] = source_stats.replace(
        mean=means, std=deviations, summed_variance=variances
    )
    return prepared


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_checkpoint(source_checkpoint: Path, output_root: Path) -> Path:
    """Save a separate command-safe checkpoint and retain its source XML guards."""
    source_checkpoint = source_checkpoint.resolve(strict=True)
    output_root = output_root.resolve()
    source_run = next(
        (parent for parent in source_checkpoint.parents if (parent / "config.json").is_file()),
        None,
    )
    if source_run is None:
        raise FileNotFoundError("Cannot find the source run's config.json.")
    run_config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    if run_config["env"]["deepmimic_reward_mode"] != "pure":
        raise ValueError("Only a pure imitation checkpoint can use this migration.")
    manifest_path = source_run / "xml_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("action_size") != 26 or not manifest.get("xml_sha256"):
        raise ValueError("A 26-action source XML hash manifest is required.")
    metadata_hash = file_sha256(source_checkpoint / "_CHECKPOINT_METADATA")
    provenance = {
        "source_checkpoint": str(source_checkpoint),
        "source_metadata_sha256": metadata_hash,
        "command_channels": [9, 10, 11],
        "command_input_kernel": "zero for actor and critic",
        "command_standard_deviation": 1.0,
        "non_command_parameters": "unchanged",
    }
    output_checkpoint = output_root / "checkpoints" / "000000000000"
    migration_file = output_root / "joystick_migration.json"
    if output_root.exists():
        if (
            migration_file.is_file()
            and json.loads(migration_file.read_text(encoding="utf-8")) == provenance
            and (output_checkpoint / "_CHECKPOINT_METADATA").is_file()
            and (output_checkpoint / "ppo_network_config.json").is_file()
            and (output_root / "config.json").is_file()
            and (output_root / "xml_manifest.json").is_file()
        ):
            return output_checkpoint
        raise FileExistsError(f"Refusing to overwrite migration directory: {output_root}")

    network_config = json.loads(
        (source_checkpoint / "ppo_network_config.json").read_text(encoding="utf-8")
    )
    if network_config["action_size"] != 26 or not network_config["normalize_observations"]:
        raise ValueError("Expected a normalized 26-action source policy.")
    for key, size in OBSERVATION_SIZES.items():
        if network_config["observation_size"][key]["shape"] != [size]:
            raise ValueError(f"Unsupported network observation shape for {key}.")
    factory = network_config["network_factory_kwargs"]
    if factory.get("distribution_type") != "tanh_normal":
        raise ValueError("Only tanh-normal policy checkpoints are supported.")
    if factory.get("policy_obs_key") != "state" or factory.get("value_obs_key") != "privileged_state":
        raise ValueError("Unexpected actor/critic observation keys.")
    # The installed loader rejects optional initializer names serialized as null.
    # Restore only explicitly configured callables, leaving optional values alone.
    if factory.get("activation") is not None:
        factory["activation"] = networks.ACTIVATION[factory["activation"]]
    for name in (
        "policy_network_kernel_init_fn",
        "value_network_kernel_init_fn",
        "mean_kernel_init_fn",
    ):
        if factory.get(name) is not None:
            factory[name] = networks.KERNEL_INITIALIZER[factory[name]]
    prepared = prepare_command_inputs(ppo_checkpoint.load(source_checkpoint))
    output_root.mkdir(parents=True)
    ppo_checkpoint.save(
        output_root / "checkpoints", 0, prepared, config_dict.ConfigDict(network_config)
    )
    shutil.copy2(source_run / "config.json", output_root / "config.json")
    shutil.copy2(manifest_path, output_root / "xml_manifest.json")
    migration_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output_checkpoint


def main() -> None:
    """Prepare a new joystick warm start without modifying the source checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_checkpoint(args.source_checkpoint, args.output_root))


if __name__ == "__main__":
    main()
