from pathlib import Path
import pandas as pd
import argparse


def split_bvh_into_steps(input_file, steps, output_dir="steps"):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r") as f:
        lines = f.readlines()

    # MOTION part
    motion_index = None

    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_index = i
            break

    if motion_index is None:
        raise ValueError("No MOTION part in this BVH.")

    # HIERARCHY part
    hierarchy = lines[:motion_index]

    # Frames
    frames_line_index = motion_index + 1
    frame_time_line_index = motion_index + 2

    original_frame_time = lines[frame_time_line_index]

    motion_data = lines[frame_time_line_index + 1:]

    motion_data = [
        line for line in motion_data
        if line.strip()
    ]

    print(f"N frames: {len(motion_data)}")

    # New BVH for each step
    for step_number, (start_frame, end_frame) in enumerate(steps, start=1):
        selected_frames = motion_data[start_frame:end_frame + 1]

        if not selected_frames:
            print(
                f"WARNING: step {step_number} "
                f"no frames ({start_frame}-{end_frame})"
            )
            continue

        output_file = output_dir / f"step_{step_number:02d}.bvh"

        num_frames = len(selected_frames)

        with open(output_file, "w") as f:

            # HIERARCHY
            f.writelines(hierarchy)

            # MOTION
            f.write("MOTION\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(original_frame_time)

            f.writelines(selected_frames)

        print(
            f"Step {step_number}: "
            f"frame {start_frame}-{end_frame} "
            f"-> {output_file}"
        )


def read_start_end(steps_path: str):
    steps = []
    try:
        data = pd.read_csv(steps_path)

    except pd.errors.EmptyDataError:
        print("CSV file is completely empty")
        return None

    # print(data)
    steps = list(
        zip(
            data["start_frame"],
            data["end_frame"]
        )
    )

    step_duration = list(
        data["end_frame"] - data["start_frame"]
    )
    # print(step_duration)
    mean_duration = sum(step_duration) / len(step_duration)
    # print(mean_duration)
    return steps, mean_duration


def main(input_bvh: str, steps_path: str, save_dir: str):
    steps, mean_step_len = read_start_end(steps_path)
    # print(steps)
    print(f"Mean step length in frames: {mean_step_len}")

    split_bvh_into_steps(
        input_file=input_bvh,
        steps=steps,
        output_dir=save_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVH Segmentation to Single Steps")

    parser.add_argument(
        "--input_bvh",
        required=False,
        type=str,
        default="0005_Walking001.bvh",
        help="MoCap BVH"
    )

    parser.add_argument(
        "--save_dir",
        required=False,
        type=str,
        default="steps",
        help="Saving dir"
    )

    parser.add_argument(
        "--input_csv",
        required=False,
        type=str,
        default="steps.csv",
        help="CSV Steps"
    )

    args = parser.parse_args()

    bvh_path = args.input_bvh
    save = args.save_dir
    steps_path = args.input_csv

    main(bvh_path, steps_path, save)

