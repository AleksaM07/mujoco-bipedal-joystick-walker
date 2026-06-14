import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Otvori generated human model u training standing-home pozi."
    )
    parser.add_argument(
        "--env-version",
        choices=["standard", "hardcore"],
        default="standard",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Pusti pasivnu simulaciju sa default standing ctrl targetima.",
    )
    parser.add_argument(
        "--pose-scale",
        type=float,
        default=1.0,
        help="Pomnozi standing-home zglobove radi vizuelnog debug-a.",
    )
    parser.add_argument(
        "--init-qpos-file",
        type=Path,
        default=None,
        help="Opcioni MJDATA/QPOS fajl za pocetnu pozu.",
    )
    parser.add_argument(
        "--fast-physics",
        action="store_true",
        help="Koristi sim_dt=0.01 kao training --fast-physics.",
    )
    args = parser.parse_args()

    config_overrides = {
        "enable_erfi": False,
        "impl": "jax",
    }
    if args.init_qpos_file is not None:
        config_overrides["init_qpos_file"] = str(args.init_qpos_file)
    if not args.fast_physics:
        config_overrides["sim_dt"] = 0.005

    env = BiomechanicsJoystickEnv(
        env_version=args.env_version,
        config_overrides=config_overrides,
    )
    model = env.mj_model
    data = mujoco.MjData(model)
    init_q = np.asarray(env._init_q).copy()
    default_ctrl = np.asarray(env._default_ctrl).copy()
    if args.pose_scale != 1.0:
        for joint_name in env.NEUTRAL_JOINT_POSE:
            joint_id = model.joint(joint_name).id
            qpos_id = model.jnt_qposadr[joint_id]
            init_q[qpos_id] *= args.pose_scale
        default_ctrl = init_q[np.asarray(env._actuator_qpos_indices)]

    data.qpos[:] = init_q
    data.qvel[:] = 0.0
    data.ctrl[:] = default_ctrl
    mujoco.mj_forward(model, data)

    print(f"xml={env.xml_path}", flush=True)
    print(
        "standing-home | "
        f"z={data.qpos[2]:.3f} | "
        f"init_qpos_file={args.init_qpos_file} | "
        f"pose_scale={args.pose_scale:.2f} | "
        f"hip_x={data.qpos[model.jnt_qposadr[model.joint('left_hip_x').id]]:.3f} | "
        f"hip_z={data.qpos[model.jnt_qposadr[model.joint('left_hip_z').id]]:.3f} | "
        f"knee_z={data.qpos[model.jnt_qposadr[model.joint('left_knee_z').id]]:.3f} | "
        f"ankle_y={data.qpos[model.jnt_qposadr[model.joint('left_ankle_y').id]]:.3f} | "
        f"ankle_z={data.qpos[model.jnt_qposadr[model.joint('left_ankle_z').id]]:.3f}",
        flush=True,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20

        while viewer.is_running():
            if args.simulate:
                data.ctrl[:] = default_ctrl
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
