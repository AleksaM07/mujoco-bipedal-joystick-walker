import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import csv
import argparse


# ============================================================
# BVH PARSER
# ============================================================

class Joint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)
        self.channels = []
        self.channel_indices = []

    def add_child(self, child):
        self.children.append(child)


class BVH:
    def __init__(self, filename):
        self.filename = filename
        self.joints = {}
        self.root = None

        self.frames = None
        self.frame_time = None

        self._parse()

    def _parse(self):

        with open(self.filename, "r") as f:
            lines = [line.strip() for line in f]

        stack = []
        current_joint = None

        channel_counter = 0

        motion_start = None

        i = 0

        while i < len(lines):

            line = lines[i]

            # =================================================
            # ROOT
            # =================================================

            if line.startswith("ROOT"):

                name = line.split()[1]

                joint = Joint(name)

                self.root = joint
                self.joints[name] = joint

                current_joint = joint

                # The next "{" belongs to this joint
                stack.append(("joint", joint))

            # =================================================
            # JOINT
            # =================================================

            elif line.startswith("JOINT"):

                name = line.split()[1]

                if not stack:
                    raise ValueError(
                        f"JOINT {name} has no parent."
                    )

                # Find nearest joint in stack
                parent = None

                for item_type, item in reversed(stack):

                    if item_type == "joint":
                        parent = item
                        break

                if parent is None:
                    raise ValueError(
                        f"Could not find parent of {name}"
                    )

                joint = Joint(name, parent)

                parent.add_child(joint)

                self.joints[name] = joint

                current_joint = joint

                stack.append(("joint", joint))

            # =================================================
            # END SITE
            # =================================================

            elif line == "End Site":

                # End Site is not a joint.
                #
                # We add a marker to the stack so that its
                # closing "}" does not accidentally pop the
                # parent joint.

                stack.append(("end_site", None))

            # =================================================
            # OFFSET
            # =================================================

            elif line.startswith("OFFSET"):

                values = np.fromstring(
                    line.replace("OFFSET", ""),
                    sep=" "
                )

                # Ignore End Site offsets.
                if current_joint is not None:
                    current_joint.offset = values

            # =================================================
            # CHANNELS
            # =================================================

            elif line.startswith("CHANNELS"):

                parts = line.split()

                n_channels = int(parts[1])

                channels = parts[2:2 + n_channels]

                current_joint.channels = channels

                current_joint.channel_indices = list(
                    range(
                        channel_counter,
                        channel_counter + n_channels
                    )
                )

                channel_counter += n_channels

            # =================================================
            # OPEN BRACE
            # =================================================

            elif line == "{":
                pass

            # =================================================
            # CLOSE BRACE
            # =================================================

            elif line == "}":

                if not stack:
                    raise ValueError(
                        f"Unexpected '}}' at line {i}"
                    )

                item_type, item = stack.pop()

                if item_type == "joint":

                    # Restore parent as current joint
                    current_joint = None

                    for t, obj in reversed(stack):

                        if t == "joint":
                            current_joint = obj
                            break

            # =================================================
            # MOTION
            # =================================================

            elif line == "MOTION":

                motion_start = i
                break

            i += 1

        # =====================================================
        # CHECK STRUCTURE
        # =====================================================

        if self.root is None:
            raise ValueError("No ROOT found.")

        if motion_start is None:
            raise ValueError("No MOTION section found.")

        # =====================================================
        # MOTION HEADER
        # =====================================================

        frames_line = lines[motion_start + 1]
        frame_time_line = lines[motion_start + 2]

        n_frames = int(
            frames_line.split(":")[1].strip()
        )

        self.frame_time = float(
            frame_time_line.split(":")[1].strip()
        )

        # =====================================================
        # MOTION DATA
        # =====================================================

        motion_data = []

        for line in lines[motion_start + 3:]:

            if line.strip():
                values = np.fromstring(
                    line,
                    sep=" "
                )

                motion_data.append(values)

        self.frames = np.asarray(motion_data)

        # =====================================================
        # VALIDATION
        # =====================================================

        if len(self.frames) != n_frames:
            raise ValueError(
                f"Expected {n_frames} frames, "
                f"but found {len(self.frames)}."
            )

        if self.frames.shape[1] != channel_counter:
            raise ValueError(
                f"Expected {channel_counter} channels, "
                f"but found {self.frames.shape[1]}."
            )

        print("BVH loaded successfully.")
        print(f"Frames: {n_frames}")
        print(f"Frame time: {self.frame_time}")
        print(f"Channels: {channel_counter}")
        print(f"Joints: {len(self.joints)}")


# ============================================================
# ROTATION UTILITIES
# ============================================================

def rotation_matrix(axis, angle_deg):

    angle = np.deg2rad(angle_deg)

    c = np.cos(angle)
    s = np.sin(angle)

    if axis == "X":

        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    elif axis == "Y":

        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    elif axis == "Z":

        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    raise ValueError(f"Unknown axis: {axis}")


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def compute_global_positions(bvh):

    n_frames = bvh.frames.shape[0]

    positions = {
        name: np.zeros((n_frames, 3))
        for name in bvh.joints
    }

    # --------------------------------------------------------
    # Recursive FK
    # --------------------------------------------------------

    def traverse(joint, frame, parent_position, parent_rotation):

        channel_values = bvh.frames[
            frame,
            joint.channel_indices
        ]

        # ----------------------------------------------------
        # Translation
        # ----------------------------------------------------

        translation = np.zeros(3)

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        rotation = np.eye(3)

        value_index = 0

        for channel in joint.channels:

            value = channel_values[value_index]

            value_index += 1

            # -----------------------------------------------
            # Root translation
            # -----------------------------------------------

            if channel == "Xposition":
                translation[0] = value

            elif channel == "Yposition":
                translation[1] = value

            elif channel == "Zposition":
                translation[2] = value

            # -----------------------------------------------
            # Rotation
            # -----------------------------------------------

            elif channel == "Xrotation":

                rotation = rotation @ \
                    rotation_matrix("X", value)

            elif channel == "Yrotation":

                rotation = rotation @ \
                    rotation_matrix("Y", value)

            elif channel == "Zrotation":

                rotation = rotation @ \
                    rotation_matrix("Z", value)

        # ----------------------------------------------------
        # Global transform
        # ----------------------------------------------------

        if joint.parent is None:

            global_position = translation

            global_rotation = rotation

        else:

            global_position = (
                parent_position
                + parent_rotation @ joint.offset
            )

            global_rotation = (
                parent_rotation @ rotation
            )

        positions[joint.name][frame] = global_position

        # ----------------------------------------------------
        # Children
        # ----------------------------------------------------

        for child in joint.children:

            traverse(
                child,
                frame,
                global_position,
                global_rotation
            )

    # --------------------------------------------------------
    # Process all frames
    # --------------------------------------------------------

    for frame in range(n_frames):

        traverse(
            bvh.root,
            frame,
            np.zeros(3),
            np.eye(3)
        )

    return positions



# ============================================================
# FOOT VELOCITY
# ============================================================

def compute_foot_speed(position, dt):

    velocity = np.gradient(
        position,
        dt,
        axis=0
    )

    speed = np.linalg.norm(
        velocity,
        axis=1
    )

    # malo filtriranja
    speed = medfilt(speed, kernel_size=5)

    return speed


# ============================================================
# CONTACT DETECTION
# ============================================================

def detect_contacts(speed, threshold):

    contact = speed < threshold

    contact = medfilt(
        contact.astype(float),
        7
    ).astype(bool)

    return contact


# ============================================================
# HEEL STRIKE DETECTION
# ============================================================

def detect_heel_strikes(contact):

    hs = np.where(
        np.diff(contact.astype(int)) == 1
    )[0] + 1

    return hs


# ============================================================
# STEP SEGMENTATION
# ============================================================

def segment_steps(left_hs, right_hs):

    events = []

    for f in left_hs:
        events.append((f, "L"))

    for f in right_hs:
        events.append((f, "R"))

    events.sort(key=lambda x: x[0])

    steps = []

    for i in range(len(events)-1):

        start = events[i][0]
        end = events[i+1][0]

        support = events[i][1]

        steps.append(
            (i, start, end, support)
        )

    return steps


# ============================================================
# SAVE CSV
# ============================================================

def save_steps_csv(steps,
                   filename="steps.csv"):

    with open(filename, "w",
              newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            ["step_id",
             "start_frame",
             "end_frame",
             "support_foot"]
        )

        writer.writerows(steps)

    print("Saved:", filename)


# ============================================================
# PLOT
# ============================================================

def plot_results(
    left_speed,
    right_speed,
    left_hs,
    right_hs
):

    plt.figure(figsize=(14, 6))

    plt.plot(left_speed,
             label="Left Foot")

    plt.plot(right_speed,
             label="Right Foot")

    for x in left_hs:
        plt.axvline(
            x,
            linestyle="--",
            alpha=0.5
        )

    for x in right_hs:
        plt.axvline(
            x,
            linestyle=":"
        )

    plt.xlabel("Frame")
    plt.ylabel("Speed [cm/s]")
    plt.title(
        "Foot Speed and Heel Strikes"
    )

    plt.legend()
    plt.grid(True)

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main(bvh_file_path: str, save_filename: str, velocity_threshold: int):
    bvh = BVH(bvh_file_path)

    positions = compute_global_positions(bvh)

    left_foot = positions["LeftFoot"]
    right_foot = positions["RightFoot"]

    # print("LeftFoot first frame:")
    # print(left_foot[0])
    #
    # print("RightFoot first frame:")
    # print(right_foot[0])
    #
    # print("LeftFoot shape:")
    # print(left_foot.shape)

    left_speed = compute_foot_speed(
        left_foot,
        bvh.frame_time
    )

    right_speed = compute_foot_speed(
        right_foot,
        bvh.frame_time
    )

    left_contact = detect_contacts(
        left_speed,
        threshold=velocity_threshold
    )

    right_contact = detect_contacts(
        right_speed,
        threshold=velocity_threshold
    )

    left_hs = detect_heel_strikes(
        left_contact
    )

    right_hs = detect_heel_strikes(
        right_contact
    )

    steps = segment_steps(
        left_hs,
        right_hs
    )

    save_steps_csv(
        steps,
        save_filename
    )

    print("\nDetected steps:")
    for s in steps:
        print(s)

    plot_results(
        left_speed,
        right_speed,
        left_hs,
        right_hs
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
        "--save_csv",
        required=False,
        type=str,
        default="steps.csv",
        help="CSV file to save"
    )

    parser.add_argument(
        "--vel_threshold",
        required=False,
        type=int,
        default=5,
        help="Velocity threshold"
    )

    args = parser.parse_args()

    bvh_path = args.input_bvh
    save = args.save_csv
    vel_tr = args.vel_threshold

    main(bvh_path, save, vel_tr)

