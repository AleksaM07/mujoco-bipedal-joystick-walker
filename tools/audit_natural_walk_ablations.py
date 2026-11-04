"""Audit isolated natural-walk reference ablations before PPO training.

This tool measures source-to-host signal retention through the production
retargeting pipeline.  It does not train a policy and does not treat kinematic
quality as proof of dynamically feasible walking.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bvh_reference import (
    MotionSegment,
    _bvh_elbow_flexion_target,
    _parse_bvh,
)
from config import EnvConfig
from tools.check_retarget_stages import capture_retarget_stages


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "prezentacija/4th_success_better_arms_stiff_elbows/config.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "analysis_outputs/natural_walk_ablations/reference_matrix.json"
)

ABLATIONS: dict[str, dict[str, object]] = {
    "current": {},
    "root_xy_060": {"reference_root_xy_scale": 0.60},
    "root_xy_100": {"reference_root_xy_scale": 1.00},
    "lateral_060": {"reference_stability_other_alpha": 0.60},
    "lateral_100": {"reference_stability_other_alpha": 1.00},
    "no_stance_lock": {"reference_lock_stance_feet": False},
    # Root scaling and stance locking interact: the lock reconstructs root XY
    # from foot anchors after scaling.  These conditional variants identify the
    # otherwise masked root-scale effect; they are not candidate presets.
    "root_xy_060_no_lock": {
        "reference_root_xy_scale": 0.60,
        "reference_lock_stance_feet": False,
    },
    "root_xy_100_no_lock": {
        "reference_root_xy_scale": 1.00,
        "reference_lock_stance_feet": False,
    },
}


def load_base_config(path: Path) -> EnvConfig:
    """Load one archived environment config for deterministic comparison."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("env"), dict):
        raise ValueError(f"Expected an archived config with an env section: {path}")
    values = dict(payload["env"])
    values["playground_impl"] = "jax"
    values["physics_backend"] = "mjx_jax"
    values["warp_num_worlds"] = 1
    values["reference_elbow_flexion_alpha"] = 1.0
    return EnvConfig(**values)


def apply_overrides(config: EnvConfig, overrides: dict[str, object]) -> EnvConfig:
    """Return a fresh config with one isolated ablation applied."""
    values = asdict(config)
    values.update(overrides)
    return EnvConfig(**values)


def source_elbow_ranges(
    config: EnvConfig,
    metadata: dict[str, Any],
) -> dict[str, float]:
    """Measure geometric flexion range for the exact retained source segment."""
    source_paths = config.reference_gait_file
    if len(source_paths) != 1:
        raise ValueError("Natural-walk ablation requires exactly one BVH source file.")
    source_path = Path(source_paths[0])
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    parsed = _parse_bvh(source_path)
    start = int(metadata["source_start_frames"][0])
    end = int(metadata["source_end_frames"][0])
    segment = parsed.slice(MotionSegment(start, end))
    return {
        side: float(np.degrees(np.ptp(_bvh_elbow_flexion_target(segment, side))))
        for side in ("left", "right")
    }


def stage_summary(
    env: Any,
    stage: dict[str, Any],
    count: int,
    source_ranges_deg: dict[str, float],
) -> dict[str, Any]:
    """Summarize joint amplitude and root travel for one production stage."""
    joints = np.asarray(stage["joints"])[0, :count]
    roots = np.asarray(stage["roots"])[0, :count]
    if not np.isfinite(joints).all() or not np.isfinite(roots).all():
        raise ValueError(f"Non-finite values in stage {stage['name']}.")

    joint_ranges_deg = {
        name: float(np.degrees(np.ptp(joints[:, index])))
        for index, name in enumerate(env._actuator_joint_names)
    }
    elbow: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        name = f"{side}_elbow_z"
        source_range = source_ranges_deg[side]
        host_range = joint_ranges_deg[name]
        elbow[side] = {
            "source_range_deg": source_range,
            "host_range_deg": host_range,
            "retention_ratio": host_range / max(source_range, 1e-9),
        }

    root_delta = roots[-1] - roots[0]
    return {
        "stage": stage["name"],
        "joint_ranges_deg": joint_ranges_deg,
        "elbow": elbow,
        "root_net_xyz_m": root_delta.tolist(),
        "root_net_xy_m": float(np.linalg.norm(root_delta[:2])),
    }


def audit_variant(name: str, config: EnvConfig) -> dict[str, Any]:
    """Run the real retargeting pipeline and evaluate elbow retention gates."""
    env, metadata, stages = capture_retarget_stages(config)
    if len(metadata["counts"]) != 1:
        raise ValueError(f"Expected one retained clip, got {len(metadata['counts'])}.")
    count = int(metadata["counts"][0])
    source_ranges_deg = source_elbow_ranges(config, metadata)
    summaries = [
        stage_summary(env, stage, count, source_ranges_deg) for stage in stages
    ]
    by_name = {row["stage"]: row for row in summaries}
    required = ("raw_mapped_pose", "ik_solution", "stability_prior")
    missing = [stage for stage in required if stage not in by_name]
    if missing:
        raise ValueError(f"Missing production stages for {name}: {missing}")

    gates: dict[str, dict[str, object]] = {}
    for side in ("left", "right"):
        seed_ratio = by_name["raw_mapped_pose"]["elbow"][side]["retention_ratio"]
        ik_ratio = by_name["ik_solution"]["elbow"][side]["retention_ratio"]
        final_ratio = by_name["stability_prior"]["elbow"][side]["retention_ratio"]
        gates[side] = {
            "seed_retention_at_least_95pct": seed_ratio >= 0.95,
            "ik_retention_at_least_70pct": ik_ratio >= 0.70,
            "final_retention_at_least_70pct": final_ratio >= 0.70,
            "pass": seed_ratio >= 0.95 and ik_ratio >= 0.70 and final_ratio >= 0.70,
        }

    return {
        "name": name,
        "overrides": ABLATIONS[name],
        "source_frames": {
            "start": int(metadata["source_start_frames"][0]),
            "end": int(metadata["source_end_frames"][0]),
            "count": count,
        },
        "source_elbow_ranges_deg": source_ranges_deg,
        "curve_preparation_error": metadata.get("curve_preparation_error"),
        "stages": summaries,
        "gates": gates,
        "pass": all(bool(gate["pass"]) for gate in gates.values()),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(ABLATIONS),
        default=list(ABLATIONS),
    )
    return parser.parse_args()


def main() -> None:
    """Run selected ablations and write one reproducible JSON report."""
    args = parse_args()
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    config_path = args.config.resolve(strict=True)
    base = load_base_config(config_path)
    results = []
    for name in args.variants:
        print(f"Auditing {name}...", flush=True)
        results.append(audit_variant(name, apply_overrides(base, ABLATIONS[name])))

    payload = {
        "schema_version": 1,
        "purpose": "Kinematic reference ablation; not PPO or dynamic-walk proof.",
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": {
            path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for path in (
                "bvh_reference.py",
                "biomechanics_env.py",
                "tools/audit_natural_walk_ablations.py",
            )
        },
        "action_smoothing_note": (
            "action_smoothing does not alter the kinematic reference and must be "
            "tested later with matched policy rollouts."
        ),
        "variants": results,
    }
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}", flush=True)
    for result in results:
        elbow = {
            side: result["gates"][side]["pass"] for side in ("left", "right")
        }
        print(f"{result['name']}: pass={result['pass']} elbow={elbow}", flush=True)


if __name__ == "__main__":
    main()
