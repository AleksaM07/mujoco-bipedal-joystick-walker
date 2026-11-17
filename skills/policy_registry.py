"""Canonical labels for frozen production locomotion policies.

The registry points at existing checkpoint directories.  It never copies or
rewrites those archives, so experiments can depend on stable, reviewable
artifacts while keeping each policy's own observation normalizer.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
EXPECTED_XML_SHA256: Final = (
    "494cb6707a59a957a27e248f07d7d7597227250588a9c6ca70b6314709895ef4"
)

PolicyLabel = Literal["FINAL_WALKING", "FINAL_STANDING"]
LocomotionSkill = Literal["walk", "stand"]


@dataclass(frozen=True)
class FrozenPolicy:
    """One immutable policy artifact used by the skill supervisor."""

    label: PolicyLabel
    checkpoint_relative_path: str
    locomotion_skill: LocomotionSkill
    evidence_relative_path: str
    expected_observation_size: int = 366
    expected_action_size: int = 26
    expected_xml_sha256: str = EXPECTED_XML_SHA256

    @property
    def checkpoint(self) -> Path:
        """Return the checkpoint path in this checkout."""
        return PROJECT_ROOT / self.checkpoint_relative_path

    @property
    def evidence(self) -> Path:
        """Return the human-readable evidence path in this checkout."""
        return PROJECT_ROOT / self.evidence_relative_path


FINAL_WALKING: Final = FrozenPolicy(
    label="FINAL_WALKING",
    checkpoint_relative_path=(
        "prezentacija/5th_success_final_natural_walk/model_000001024000"
    ),
    locomotion_skill="walk",
    evidence_relative_path="prezentacija/5th_success_final_natural_walk/FINAL_SUCCESS.txt",
)

FINAL_STANDING: Final = FrozenPolicy(
    label="FINAL_STANDING",
    checkpoint_relative_path=(
        "runs/stand_horizon60_20261106_1m_lr3e-4/resume/"
        "20261106_1414_stand_balance_v1_resume_1000000_lr0_0003_"
        "V1_steps-1m_reward-8554p6201_s/checkpoints/000001024000"
    ),
    locomotion_skill="stand",
    evidence_relative_path="docs/STAND_HORIZON60_RESULTS.md",
)

FINAL_POLICIES: Final = (FINAL_WALKING, FINAL_STANDING)


def _network_shape(network_config: dict, key: str) -> int:
    observation_size = network_config["observation_size"]
    return int(observation_size[key]["shape"][0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_policy(policy: FrozenPolicy) -> dict[str, str | int]:
    """Fail closed if a labeled checkpoint no longer matches its contract."""
    checkpoint = policy.checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"{policy.label} checkpoint is missing: {checkpoint}")
    if not policy.evidence.is_file():
        raise FileNotFoundError(f"{policy.label} evidence is missing: {policy.evidence}")

    network_path = checkpoint / "ppo_network_config.json"
    network_config = json.loads(network_path.read_text(encoding="utf-8"))
    observation_size = _network_shape(network_config, "state")
    action_size = int(network_config["action_size"])
    if observation_size != policy.expected_observation_size:
        raise ValueError(
            f"{policy.label} expects observation size "
            f"{policy.expected_observation_size}, found {observation_size}."
        )
    if action_size != policy.expected_action_size:
        raise ValueError(
            f"{policy.label} expects {policy.expected_action_size} actions, "
            f"found {action_size}."
        )

    run_dir = checkpoint
    while run_dir != PROJECT_ROOT and not (run_dir / "xml_manifest.json").is_file():
        run_dir = run_dir.parent
    manifest_path = run_dir / "xml_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{policy.label} has no XML manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    xml_hash = str(manifest.get("xml_sha256", ""))
    if xml_hash != policy.expected_xml_sha256:
        raise ValueError(
            f"{policy.label} XML hash {xml_hash[:12]} does not match "
            f"{policy.expected_xml_sha256[:12]}."
        )

    return {
        "label": policy.label,
        "checkpoint": str(checkpoint),
        "observation_size": observation_size,
        "action_size": action_size,
        "xml_sha256": xml_hash,
        "checkpoint_metadata_sha256": _sha256(checkpoint / "_CHECKPOINT_METADATA"),
    }


def validate_final_policy_pair() -> tuple[dict[str, str | int], ...]:
    """Validate both frozen policies and their shared physical interface."""
    reports = tuple(validate_frozen_policy(policy) for policy in FINAL_POLICIES)
    if reports[0]["xml_sha256"] != reports[1]["xml_sha256"]:
        raise ValueError("FINAL_WALKING and FINAL_STANDING use different robot XMLs.")
    return reports
