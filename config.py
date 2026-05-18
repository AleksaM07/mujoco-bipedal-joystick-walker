from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BIOMECH_DIR = WORKSPACE_ROOT / "mujoco-biomechanics"
RUNS_DIR = PROJECT_ROOT / "runs"


# Aktuiramo samo zglobove koji su direktno bitni za hod i balans.
# Ruke, zglobovi sake i vrat ostaju pasivni da prostor akcija ne eksplodira.
LOCOMOTION_JOINTS = [
    "abdomen_x",
    "abdomen_y",
    "abdomen_z",
    "pelvis_x",
    "pelvis_y",
    "pelvis_z",
    "left_hip_x",
    "left_hip_y",
    "left_hip_z",
    "left_knee_z",
    "left_ankle_y",
    "left_ankle_z",
    "right_hip_x",
    "right_hip_y",
    "right_hip_z",
    "right_knee_z",
    "right_ankle_y",
    "right_ankle_z",
]


@dataclass
class EnvConfig:
    # Epizodna randomizacija coveka.
    min_mass: float = 55.0
    max_mass: float = 95.0
    min_height: float = 1.55
    max_height: float = 1.95
    sex_probability_male: float = 0.5

    # Goal-conditioned ulaz: zeljena brzina tela.
    command_x_range: tuple[float, float] = (-0.4, 1.2)
    command_y_range: tuple[float, float] = (-0.35, 0.35)
    command_yaw_range: tuple[float, float] = (-1.2, 1.2)

    # Simulacija.
    frame_skip: int = 5
    episode_seconds: float = 12.0
    healthy_min_height_fraction: float = 0.45
    healthy_max_height_fraction: float = 1.35

    # Aktuatori i perturbacije. Torque noise je analogan sumu u momentima motora.
    torque_limit: float = 50.0
    torque_noise_std: float = 1.0
    episode_torque_bias_std: float = 2.0
    push_probability: float = 0.015
    push_force_std: float = 25.0

    # Reward skale.
    velocity_sigma: float = 0.45
    healthy_reward: float = 0.5
    control_cost: float = 0.001
    action_rate_cost: float = 0.01
    lateral_height_cost: float = 0.05
