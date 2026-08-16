"""Summarize a reference playback trace for manual debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty trace file: {path}")
    return rows


def _first_index(rows: list[dict], key: str, predicate) -> int | None:
    for index, row in enumerate(rows):
        value = row.get(key)
        if value is not None and predicate(value):
            return index
    return None


def _support_contact_mismatch(row: dict) -> str:
    support = str(row.get("support_foot") or "").upper()
    left = float(row.get("left_foot_contact", 0.0)) > 0.5
    right = float(row.get("right_foot_contact", 0.0)) > 0.5
    if support == "L" and not left:
        return "support_left_missing"
    if support == "R" and not right:
        return "support_right_missing"
    return ""


def _print_row(label: str, row: dict) -> None:
    mismatch = _support_contact_mismatch(row)
    print(
        f"{label}: step={row['step']} time={row['time_s']:.2f}s "
        f"motion_t={row.get('reference_motion_time', 0.0):.3f}s "
        f"height={row.get('height', 0.0):.3f} "
        f"pose={row.get('deepmimic_pose', 0.0):.3f} "
        f"root_xy_err={row.get('deepmimic_root_xy_error', 0.0):.3f} "
        f"root_h_err={row.get('deepmimic_root_height_error', 0.0):.3f} "
        f"root_vel_err={row.get('deepmimic_root_vel_error', 0.0):.3f} "
        f"key={row.get('deepmimic_key_position', 0.0):.3f} "
        f"key_err={row.get('deepmimic_key_pos_error', 0.0):.3f} "
        f"l_contact={row.get('left_foot_contact', 0.0):.0f} "
        f"r_contact={row.get('right_foot_contact', 0.0):.0f} "
        f"l_foot_err={row.get('left_foot_height_tracking_error', 0.0):.3f} "
        f"r_foot_err={row.get('right_foot_height_tracking_error', 0.0):.3f} "
        f"torso_up={row.get('torso_up', 0.0):.3f}"
        + (f" mismatch={mismatch}" if mismatch else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize playback_trace.jsonl for manual reference debugging."
    )
    parser.add_argument("trace", type=Path, help="Path to playback_trace.jsonl")
    args = parser.parse_args()

    rows = _load_rows(args.trace)
    first = rows[0]
    last = rows[-1]

    print(f"trace={args.trace}")
    print(
        f"clip_id={first.get('clip_id')} loop_mode={first.get('loop_mode')} "
        f"support_foot={first.get('support_foot')} "
        f"source_frames={first.get('clip_source_start_frame')}.."
        f"{first.get('clip_source_end_frame')} "
        f"frame_count={first.get('clip_frame_count')} "
        f"motion_length_s={first.get('clip_motion_length_s')}"
    )
    print(
        f"final_step={last.get('step')} done={last.get('done')} "
        f"low={last.get('done_low_height')} tipped={last.get('done_tipped')} "
        f"motion_over={last.get('done_motion_over')}"
    )

    thresholds = (
        ("first key<0.5", "deepmimic_key_position", lambda value: value < 0.5),
        ("first key<0.1", "deepmimic_key_position", lambda value: value < 0.1),
        ("first root_xy>0.5", "deepmimic_root_xy_error", lambda value: value > 0.5),
        ("first root_xy>1.0", "deepmimic_root_xy_error", lambda value: value > 1.0),
        ("first root_h>0.1", "deepmimic_root_height_error", lambda value: value > 0.1),
        ("first root_vel>1.0", "deepmimic_root_vel_error", lambda value: value > 1.0),
        (
            "first support mismatch",
            "step",
            lambda _value: False,
        ),
    )
    for label, key, predicate in thresholds:
        if label == "first support mismatch":
            index = next(
                (i for i, row in enumerate(rows) if _support_contact_mismatch(row)),
                None,
            )
        else:
            index = _first_index(rows, key, predicate)
        if index is None:
            print(f"{label}: none")
            continue
        _print_row(label, rows[index])

    print("samples:")
    sample_indices = sorted(
        set(
            [
                0,
                len(rows) // 4,
                len(rows) // 2,
                max(len(rows) - 4, 0),
                len(rows) - 1,
            ]
        )
    )
    for index in sample_indices:
        _print_row(f"sample[{index}]", rows[index])


if __name__ == "__main__":
    main()
