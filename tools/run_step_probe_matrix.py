"""Run a small BVH step-probe diagnostic matrix and summarize the results.

This wrapper exists so we can compare the same short clip segment under
different root/gravity conditions without hand-editing long shell commands.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDER_TOOL = PROJECT_ROOT / "tools" / "render_bvh_reference_videos.py"


@dataclass(frozen=True)
class ProbeCase:
    """One step-probe render configuration."""

    name: str
    title: str
    hypothesis: str
    gravity_scale: float
    pin_root_to_reference: bool
    pin_root_position_to_reference: bool = False
    pin_root_xy_to_reference: bool = False
    pin_root_z_to_reference: bool = False
    pin_root_rotation_to_reference: bool = False
    actuator_force_scale: float = 1.0
    actuator_kp_scale: float = 1.0
    trunk_kp_scale: float = 1.0
    pelvis_kp_scale: float = 1.0
    ankle_kp_scale: float = 1.0
    hip_kp_scale: float = 1.0
    contact_friction_scale: float = 1.0
    root_z_assist_kp: float = 0.0
    root_z_assist_kd: float = 0.0
    root_z_assist_max_force: float = 0.0
    root_pitch_assist_kp: float = 0.0
    root_pitch_assist_kd: float = 0.0
    root_pitch_assist_max_torque: float = 0.0
    reference_speed_scale: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the step-probe matrix runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-gait-file",
        type=Path,
        action="append",
        required=True,
        help="BVH file(s) passed through to the render tool.",
    )
    parser.add_argument("--clip-id", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("videos/step_probe_matrix"))
    parser.add_argument("--start-phase", type=float, default=0.0)
    parser.add_argument("--segment-seconds", type=float, default=0.35)
    parser.add_argument("--reference-speed-scale", type=float, default=0.20)
    parser.add_argument("--trace-dt", type=float, default=0.01)
    parser.add_argument("--video-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--reference-loop-mode",
        choices=["auto", "wrap", "clamp"],
        default="wrap",
    )
    parser.add_argument(
        "--arms-on",
        dest="arm_actuators",
        action="store_true",
        help="Run the matrix with arm actuators enabled.",
    )
    parser.add_argument(
        "--arms-off",
        dest="arm_actuators",
        action="store_false",
        help="Run the matrix with arm actuators disabled.",
    )
    parser.set_defaults(arm_actuators=False)
    parser.add_argument(
        "--reference-lock-stance-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass through stance-foot locking to the renderer.",
    )
    return parser.parse_args()


def build_cases() -> list[ProbeCase]:
    """Return the current diagnostic matrix."""
    return [
        ProbeCase(
            name="case_a_rootpinned_control",
            title="CASE A | ROOT PINNED CONTROL | g=1.0",
            hypothesis="Positive control: joint-space tracking should succeed when balance/root dynamics are removed.",
            gravity_scale=1.0,
            pin_root_to_reference=True,
        ),
        ProbeCase(
            name="case_b_rootfree_baseline",
            title="CASE B | ROOT FREE BASELINE | g=1.0",
            hypothesis="Baseline failure mode with full free-root locomotion dynamics.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
        ),
        ProbeCase(
            name="case_c_rootpos_pinned",
            title="CASE C | ROOT POS PINNED ONLY | g=1.0",
            hypothesis="Tests whether world-space translation / COM support is the main failure driver.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            pin_root_position_to_reference=True,
        ),
        ProbeCase(
            name="case_d_rootrot_pinned",
            title="CASE D | ROOT ROT PINNED ONLY | g=1.0",
            hypothesis="Tests whether trunk orientation stabilization is the main failure driver.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            pin_root_rotation_to_reference=True,
        ),
        ProbeCase(
            name="case_e_rootfree_authority2x",
            title="CASE E | ROOT FREE | 2x KP + 2x FORCE",
            hypothesis="Tests whether insufficient actuator authority/stiffness is the dominant problem.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            actuator_force_scale=2.0,
            actuator_kp_scale=2.0,
        ),
        ProbeCase(
            name="case_f_rootfree_speed0p10",
            title="CASE F | ROOT FREE | SPEED 0.10x",
            hypothesis="Tests whether the failure is mainly due to dynamic timing / motion speed.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            reference_speed_scale=0.10,
        ),
        ProbeCase(
            name="case_g_rootfree_friction3x",
            title="CASE G | ROOT FREE | 3x CONTACT FRICTION",
            hypothesis="Tests whether contact support / foot-ground traction is the main bottleneck.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            contact_friction_scale=3.0,
        ),
        ProbeCase(
            name="case_h_rootfree_xypin",
            title="CASE H | ROOT FREE | ROOT XY PIN ONLY",
            hypothesis="Tests whether horizontal root progression is the main missing ingredient while vertical balance remains free.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            pin_root_xy_to_reference=True,
        ),
        ProbeCase(
            name="case_i_rootfree_zpin",
            title="CASE I | ROOT FREE | ROOT Z PIN ONLY",
            hypothesis="Tests whether vertical support / root height retention is the main missing ingredient while horizontal progression remains free.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            pin_root_z_to_reference=True,
        ),
        ProbeCase(
            name="case_j_rootfree_kp4x",
            title="CASE J | ROOT FREE | 4x KP ONLY",
            hypothesis="Tests the 'too loose' hypothesis by increasing stiffness without increasing force limits proportionally.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            actuator_kp_scale=4.0,
        ),
        ProbeCase(
            name="case_k_rootfree_force4x",
            title="CASE K | ROOT FREE | 4x FORCE ONLY",
            hypothesis="Tests whether pure actuator headroom, without extra stiffness, is enough to stabilize the motion.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            actuator_force_scale=4.0,
        ),
        ProbeCase(
            name="case_l_rootfree_selective_supportkp",
            title="CASE L | ROOT FREE | PELVIS4x TRUNK2x ANKLE2x",
            hypothesis="Tests backward-fall stabilization with selective support-chain stiffness instead of a global high-KP gait freeze.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            trunk_kp_scale=2.0,
            pelvis_kp_scale=4.0,
            ankle_kp_scale=2.0,
        ),
        ProbeCase(
            name="case_m_rootfree_soft_zassist",
            title="CASE M | ROOT FREE | SOFT ROOT-Z ASSIST",
            hypothesis="Tests whether a modest vertical support controller alone prevents the early backward-collapse pattern.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            root_z_assist_kp=1500.0,
            root_z_assist_kd=250.0,
            root_z_assist_max_force=1200.0,
        ),
        ProbeCase(
            name="case_n_rootfree_soft_pitchassist",
            title="CASE N | ROOT FREE | SOFT ROOT-PITCH ASSIST",
            hypothesis="Tests whether a modest sagittal orientation controller alone suppresses the pelvis/torso backward lean.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            root_pitch_assist_kp=300.0,
            root_pitch_assist_kd=40.0,
            root_pitch_assist_max_torque=180.0,
        ),
        ProbeCase(
            name="case_o_rootfree_soft_zpitchassist",
            title="CASE O | ROOT FREE | SOFT ROOT-Z + PITCH ASSIST",
            hypothesis="Tests whether the dominant failure is specifically the coupling of lost root height and backward pitch, not joint-space tracking itself.",
            gravity_scale=1.0,
            pin_root_to_reference=False,
            root_z_assist_kp=1500.0,
            root_z_assist_kd=250.0,
            root_z_assist_max_force=1200.0,
            root_pitch_assist_kp=300.0,
            root_pitch_assist_kd=40.0,
            root_pitch_assist_max_torque=180.0,
        ),
    ]


def run_case(args: argparse.Namespace, case: ProbeCase) -> Path:
    """Invoke the render tool for one case and return its output directory."""
    case_dir = args.out_dir / case.name
    command = [
        sys.executable,
        str(RENDER_TOOL),
        "--out-dir",
        str(case_dir),
        "--reference-loop-mode",
        args.reference_loop_mode,
        "--clip-id",
        str(args.clip_id),
        "--mode",
        "compare",
        "--start-phase",
        str(args.start_phase),
        "--segment-seconds",
        str(args.segment_seconds),
        "--reference-speed-scale",
        str(
            case.reference_speed_scale
            if case.reference_speed_scale is not None
            else args.reference_speed_scale
        ),
        "--trace-dt",
        str(args.trace_dt),
        "--video-seconds",
        str(args.video_seconds),
        "--fps",
        str(args.fps),
        "--gravity-scale",
        str(case.gravity_scale),
        "--actuator-force-scale",
        str(case.actuator_force_scale),
        "--actuator-kp-scale",
        str(case.actuator_kp_scale),
        "--trunk-kp-scale",
        str(case.trunk_kp_scale),
        "--pelvis-kp-scale",
        str(case.pelvis_kp_scale),
        "--ankle-kp-scale",
        str(case.ankle_kp_scale),
        "--hip-kp-scale",
        str(case.hip_kp_scale),
        "--contact-friction-scale",
        str(case.contact_friction_scale),
        "--root-z-assist-kp",
        str(case.root_z_assist_kp),
        "--root-z-assist-kd",
        str(case.root_z_assist_kd),
        "--root-z-assist-max-force",
        str(case.root_z_assist_max_force),
        "--root-pitch-assist-kp",
        str(case.root_pitch_assist_kp),
        "--root-pitch-assist-kd",
        str(case.root_pitch_assist_kd),
        "--root-pitch-assist-max-torque",
        str(case.root_pitch_assist_max_torque),
        "--overlay-title",
        case.title,
    ]
    if case.pin_root_to_reference:
        command.append("--pin-root-to-reference")
    else:
        command.append("--no-pin-root-to-reference")
    if case.pin_root_position_to_reference:
        command.append("--pin-root-position-to-reference")
    else:
        command.append("--no-pin-root-position-to-reference")
    if case.pin_root_xy_to_reference:
        command.append("--pin-root-xy-to-reference")
    else:
        command.append("--no-pin-root-xy-to-reference")
    if case.pin_root_z_to_reference:
        command.append("--pin-root-z-to-reference")
    else:
        command.append("--no-pin-root-z-to-reference")
    if case.pin_root_rotation_to_reference:
        command.append("--pin-root-rotation-to-reference")
    else:
        command.append("--no-pin-root-rotation-to-reference")
    if args.arm_actuators:
        command.append("--arms-on")
    else:
        command.append("--arms-off")
    if args.reference_lock_stance_feet:
        command.append("--reference-lock-stance-feet")
    else:
        command.append("--no-reference-lock-stance-feet")
    for path in args.reference_gait_file:
        command.extend(["--reference-gait-file", str(path)])

    print(f"\n=== Running {case.name} ===")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return case_dir


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one manifest row in {manifest_path}.")
    return rows[0]


def _load_trace(trace_path: Path) -> list[dict[str, str]]:
    with trace_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _float_series(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        values.append(float(raw))
    return values


def summarize_case(case_dir: Path) -> dict[str, object]:
    """Summarize one completed case from manifest + detail trace."""
    manifest = _load_manifest(case_dir / "manifest_compare.csv")
    rows = _load_trace(case_dir / "trace_detail_compare.csv")
    rmse = _float_series(rows, "pd_pose_rmse")
    force_ratio = _float_series(rows, "pd_max_actuator_force_ratio")
    torso_up = _float_series(rows, "pd_torso_up")
    pelvis_pitch = _float_series(rows, "pd_pelvis_pitch_deg")
    torso_pitch = _float_series(rows, "pd_torso_pitch_deg")
    com_support_sagittal = _float_series(rows, "pd_com_support_sagittal")
    com_support_lateral = _float_series(rows, "pd_com_support_lateral")
    pd_root_pitch = _float_series(rows, "pd_root_pitch_deg")
    ref_root_pitch = _float_series(rows, "ref_root_pitch_deg")
    root_pitch_assist_torque = _float_series(rows, "pd_root_pitch_assist_torque")
    root_x = _float_series(rows, "pd_root_x")
    root_y = _float_series(rows, "pd_root_y")
    root_z = _float_series(rows, "pd_root_z")
    kin_x = _float_series(rows, "kin_root_x")
    kin_y = _float_series(rows, "kin_root_y")
    kin_z = _float_series(rows, "kin_root_z")
    root_err = [
        math.sqrt((px - kx) ** 2 + (py - ky) ** 2 + (pz - kz) ** 2)
        for px, py, pz, kx, ky, kz in zip(root_x, root_y, root_z, kin_x, kin_y, kin_z)
    ]
    root_pitch_err = [
        pd_pitch - ref_pitch for pd_pitch, ref_pitch in zip(pd_root_pitch, ref_root_pitch)
    ]
    return {
        "case_dir": case_dir,
        "pd_fail_reason": manifest.get("pd_fail_reason", ""),
        "pd_fail_time_s": manifest.get("pd_fail_time_s", ""),
        "gravity_scale": manifest.get("gravity_scale", ""),
        "reference_speed_scale": manifest.get("reference_speed_scale", ""),
        "pin_root_to_reference": manifest.get("pin_root_to_reference", ""),
        "pin_root_position_to_reference": manifest.get(
            "pin_root_position_to_reference",
            "",
        ),
        "pin_root_xy_to_reference": manifest.get("pin_root_xy_to_reference", ""),
        "pin_root_z_to_reference": manifest.get("pin_root_z_to_reference", ""),
        "pin_root_rotation_to_reference": manifest.get(
            "pin_root_rotation_to_reference",
            "",
        ),
        "actuator_force_scale": manifest.get("actuator_force_scale", ""),
        "actuator_kp_scale": manifest.get("actuator_kp_scale", ""),
        "trunk_kp_scale": manifest.get("trunk_kp_scale", ""),
        "pelvis_kp_scale": manifest.get("pelvis_kp_scale", ""),
        "ankle_kp_scale": manifest.get("ankle_kp_scale", ""),
        "hip_kp_scale": manifest.get("hip_kp_scale", ""),
        "contact_friction_scale": manifest.get("contact_friction_scale", ""),
        "root_z_assist_kp": manifest.get("root_z_assist_kp", ""),
        "root_z_assist_kd": manifest.get("root_z_assist_kd", ""),
        "root_z_assist_max_force": manifest.get("root_z_assist_max_force", ""),
        "root_pitch_assist_kp": manifest.get("root_pitch_assist_kp", ""),
        "root_pitch_assist_kd": manifest.get("root_pitch_assist_kd", ""),
        "root_pitch_assist_max_torque": manifest.get("root_pitch_assist_max_torque", ""),
        "rmse_mean": statistics.mean(rmse) if rmse else float("nan"),
        "rmse_max": max(rmse) if rmse else float("nan"),
        "force_mean": statistics.mean(force_ratio) if force_ratio else float("nan"),
        "force_max": max(force_ratio) if force_ratio else float("nan"),
        "torso_min": min(torso_up) if torso_up else float("nan"),
        "pelvis_pitch_mean": statistics.mean(pelvis_pitch) if pelvis_pitch else float("nan"),
        "pelvis_pitch_min": min(pelvis_pitch) if pelvis_pitch else float("nan"),
        "pelvis_pitch_max": max(pelvis_pitch) if pelvis_pitch else float("nan"),
        "torso_pitch_mean": statistics.mean(torso_pitch) if torso_pitch else float("nan"),
        "com_support_sagittal_mean": (
            statistics.mean(com_support_sagittal) if com_support_sagittal else float("nan")
        ),
        "com_support_sagittal_maxabs": (
            max(abs(value) for value in com_support_sagittal)
            if com_support_sagittal
            else float("nan")
        ),
        "com_support_lateral_mean": (
            statistics.mean(com_support_lateral) if com_support_lateral else float("nan")
        ),
        "root_pitch_err_mean": (
            statistics.mean(root_pitch_err) if root_pitch_err else float("nan")
        ),
        "root_pitch_err_maxabs": (
            max(abs(value) for value in root_pitch_err) if root_pitch_err else float("nan")
        ),
        "root_pitch_assist_torque_mean": (
            statistics.mean(root_pitch_assist_torque)
            if root_pitch_assist_torque
            else float("nan")
        ),
        "root_pitch_assist_torque_maxabs": (
            max(abs(value) for value in root_pitch_assist_torque)
            if root_pitch_assist_torque
            else float("nan")
        ),
        "root_err_mean": statistics.mean(root_err) if root_err else float("nan"),
        "root_err_max": max(root_err) if root_err else float("nan"),
        "trace_rows": len(rows),
    }


def print_summary(case_name: str, summary: dict[str, object]) -> None:
    """Print a compact human-readable case summary."""
    print(f"\n--- {case_name} ---")
    print(f"dir={summary['case_dir']}")
    print(
        "fail_reason={fail} fail_time_s={time} rows={rows}".format(
            fail=summary["pd_fail_reason"] or "none",
            time=summary["pd_fail_time_s"] or "n/a",
            rows=summary["trace_rows"],
        )
    )
    print(
        "rmse_mean={:.6f} rmse_max={:.6f} force_mean={:.6f} force_max={:.6f}".format(
            float(summary["rmse_mean"]),
            float(summary["rmse_max"]),
            float(summary["force_mean"]),
            float(summary["force_max"]),
        )
    )
    print(
        "root_err_mean={:.6f} root_err_max={:.6f} torso_min={:.6f}".format(
            float(summary["root_err_mean"]),
            float(summary["root_err_max"]),
            float(summary["torso_min"]),
        )
    )
    print(
        "pelvis_pitch_mean={:.3f} torso_pitch_mean={:.3f} "
        "com_support_sag_mean={:.3f} com_support_sag_maxabs={:.3f}".format(
            float(summary["pelvis_pitch_mean"]),
            float(summary["torso_pitch_mean"]),
            float(summary["com_support_sagittal_mean"]),
            float(summary["com_support_sagittal_maxabs"]),
        )
    )
    print(
        "root_pitch_err_mean={:.3f} root_pitch_err_maxabs={:.3f} "
        "pitch_assist_torque_mean={:.3f} pitch_assist_torque_maxabs={:.3f}".format(
            float(summary["root_pitch_err_mean"]),
            float(summary["root_pitch_err_maxabs"]),
            float(summary["root_pitch_assist_torque_mean"]),
            float(summary["root_pitch_assist_torque_maxabs"]),
        )
    )


def main() -> None:
    """Run the configured matrix and print summaries."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    summary_rows: list[dict[str, object]] = []
    for case in cases:
        case_dir = run_case(args, case)
        summary = summarize_case(case_dir)
        print_summary(case.name, summary)
        summary_rows.append(
            {
                "case_name": case.name,
                "title": case.title,
                "hypothesis": case.hypothesis,
                **summary,
            }
        )
    summary_path = args.out_dir / "matrix_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
