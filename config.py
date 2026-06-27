import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ml_collections import config_dict


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BIOMECH_DIR = WORKSPACE_ROOT / "mujoco-biomechanics"
RUNS_DIR = PROJECT_ROOT / "runs"
GENERATED_MODEL_DIR = PROJECT_ROOT / "generated_models"
SCENE_XML_VERSION = "trainfast_v17"

DEFAULT_HUMAN_MASS_KG = 75.0
DEFAULT_HUMAN_HEIGHT_M = 1.80
DEFAULT_HUMAN_SEX = "male"
DEFAULT_HUMAN_ALPHA = 1.0

XmlAttributes = dict[str, str]

# MuJoCo viewer keyboard codes. These numeric values come from the MuJoCo/GLFW
# viewer key callback, not from Python's `keyboard` or terminal input APIs.
# Keep them centralized because both `evaluate.py` and the Berkeley legacy
# viewer use the same controls.
KEY_SPACE: Final = 32
KEY_LEFT: Final = 263
KEY_RIGHT: Final = 262
KEY_DOWN: Final = 264
KEY_UP: Final = 265
KEY_A: Final = 65
KEY_D: Final = 68
KEY_E: Final = 69
KEY_Q: Final = 81
KEY_S: Final = 83
KEY_W: Final = 87
KEY_NUMPAD_2: Final = 322
KEY_NUMPAD_4: Final = 324
KEY_NUMPAD_6: Final = 326
KEY_NUMPAD_7: Final = 327
KEY_NUMPAD_8: Final = 328
KEY_NUMPAD_9: Final = 329

# Joystick policies store the commanded x/y/yaw velocity in observation indices
# 9:12. This is not arbitrary UI state: it must match each joystick env's
# `_get_obs()` layout:
#   local linear velocity (3), gyro (3), gravity (3), command (3), ...
# If an env observation layout changes, update these indices and retrain or
# use a compatible checkpoint.
COMMAND_OBS_START: Final = 9
COMMAND_OBS_END: Final = 12

# Viewer/evaluation defaults shared by joystick viewers.
# `DEBUG_PRINT_INTERVAL` throttles console diagnostics while the viewer runs.
# `DEFAULT_WALK_COMMAND_X` is only for the old "walk" command profile fallback.
# `DEFAULT_COMMAND_STEP` is how much one keyboard press changes x/y/yaw command.
DEBUG_PRINT_INTERVAL: Final = 120
DEFAULT_WALK_COMMAND_X: Final = 0.25
DEFAULT_COMMAND_STEP: Final = 0.1

# Training progress diagnostics printed by `TrainingProgressLogger`.
# Each pair is `(metric_key_from_env, short_log_label)`: the key must match a
# metric emitted by the env/eval wrapper, while the label is only the compact
# name shown in `train.log`.
TRAIN_DIAGNOSTIC_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("eval/episode_tracking_lin_vel", "tracking"),
    ("eval/episode_command_progress", "progress"),
    ("eval/episode_command_norm", "cmd_norm"),
    ("eval/episode_torso_up", "torso_up"),
    ("eval/episode_head_up", "head_up"),
    ("eval/episode_height", "height"),
    ("eval/episode_foot_slip", "foot_slip"),
    ("eval/episode_swing_drag", "swing_drag"),
    ("eval/episode_swing_clearance", "swing_clear"),
    ("eval/episode_swing_clearance_deficit", "clear_deficit"),
    ("eval/episode_variable_posture", "var_pose"),
    ("eval/episode_gait_reward", "gait"),
    ("eval/episode_reference_gait", "ref_gait"),
    ("eval/episode_reference_velocity", "ref_vel"),
    ("eval/episode_contact_force", "contact_force"),
    ("eval/episode_done_low_height", "done_low"),
    ("eval/episode_done_tipped", "done_tip"),
    ("eval/episode_done_invalid", "done_nan"),
    ("eval/episode_done", "done"),
)

# BVH curriculum list generation. These values are deliberately simple
# text-description heuristics over CMU index files; they are not runtime
# training logic. `bvh_reference.py` uses them only when run as a helper script
# to regenerate tier list files.
BVH_ROOT: Final = PROJECT_ROOT / "BVH_walking_animation"
BVH_INDEX_PATTERN: Final = re.compile(r"^\s*(\d{2,3}_\d{2})\s+(.+?)\s*$")
BVH_TIER1_EXCLUDE: Final[set[str]] = {
    "back",
    "backward",
    "backwards",
    "bent",
    "bouncy",
    "carry",
    "carries",
    "crouch",
    "crouched",
    "digital",
    "duck",
    "figure",
    "hobble",
    "jump",
    "ladder",
    "lean",
    "left",
    "limp",
    "march",
    "navigate",
    "obstacle",
    "right",
    "run",
    "side",
    "sideway",
    "sideways",
    "stairs",
    "stop",
    "style",
    "turn",
    "uneven",
    "veer",
    "weird",
    "with",
    "zigzag",
}
BVH_TIER2_HINTS: Final[set[str]] = {
    "brisk",
    "fast",
    "forward",
    "jog",
    "left",
    "right",
    "run",
    "slow",
    "start",
    "stop",
    "stride",
    "turn",
    "veer",
}
BVH_UNEVEN_HINTS: Final[set[str]] = {"stair", "stairs", "terrain", "uneven"}

# Stable `state.info` key used by the domain-randomization training wrapper.
# It records which randomized MJX model from the reset-time model bank belongs
# to each vectorized episode. Keep the string stable because wrapper reset/step
# both read it from `state.info`.
DOMAIN_RANDOMIZATION_ID: Final = "domain_randomization_id"

GENERATOR_MODULE_NAME: Final = "mujoco_biomechanics_generator"

FOOT_BODY_NAMES: Final[tuple[str, ...]] = ("left_foot", "right_foot")

TRUNK_ACTUATED_JOINTS: Final[tuple[str, ...]] = (
    "abdomen_x",
    "abdomen_y",
    "abdomen_z",
    "pelvis_x",
    "pelvis_y",
    "pelvis_z",
)

LEG_ACTUATED_JOINTS: Final[tuple[str, ...]] = (
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
)

LOCOMOTION_ACTUATED_JOINTS: Final[tuple[str, ...]] = (
    TRUNK_ACTUATED_JOINTS + LEG_ACTUATED_JOINTS
)

SOLE_CONTACT_GEOM_ATTRIBUTES: Final[XmlAttributes] = {
    "friction": "1.0 0.01 0.001",
    "solref": "0.02 1",
    "solimp": "0.85 0.95 0.005",
}

GROUND_TEXTURE_ATTRIBUTES: Final[XmlAttributes] = {
    "type": "2d",
    "name": "groundplane",
    "builtin": "checker",
    "width": "300",
    "height": "300",
    "rgb1": "1 1 1",
    "rgb2": ".85 .85 .85",
}

GROUND_MATERIAL_ATTRIBUTES: Final[XmlAttributes] = {
    "name": "groundplane",
    "texture": "groundplane",
    "texrepeat": "6 6",
    "texuniform": "true",
}

FLOOR_GEOM_ATTRIBUTES: Final[XmlAttributes] = {
    "name": "floor",
    "type": "plane",
    "size": "0 0 0.01",
    "material": "groundplane",
    "friction": "0.8",
    "condim": "3",
}

ROUGH_BLOCK_COUNT: Final = 12
ROUGH_BLOCK_START_X: Final = 0.7
ROUGH_BLOCK_X_STEP: Final = 0.45
ROUGH_BLOCK_Y_EVEN: Final = 0.35
ROUGH_BLOCK_Y_ODD: Final = -0.45
ROUGH_BLOCK_BASE_HEIGHT: Final = 0.025
ROUGH_BLOCK_HEIGHT_STEP: Final = 0.01
ROUGH_BLOCK_HEIGHT_PERIOD: Final = 3
ROUGH_BLOCK_SIZE_X: Final = 0.18
ROUGH_BLOCK_SIZE_Y: Final = 0.22
ROUGH_BLOCK_GEOM_ATTRIBUTES: Final[XmlAttributes] = {
    "type": "box",
    "rgba": ".55 .55 .55 1",
    "friction": "0.9",
    "condim": "3",
}

TRUNK_JOINT_SPECS: Final[dict[str, XmlAttributes]] = {
    "abdomen_x": {
        "range": "-12.5 12.5",
        "stiffness": "45",
        "damping": "8",
        "frictionloss": "1.0",
        "armature": "0.02",
    },
    "abdomen_y": {
        "range": "-15 15",
        "stiffness": "40",
        "damping": "8",
        "frictionloss": "1.0",
        "armature": "0.02",
    },
    "abdomen_z": {
        "range": "-18 18",
        "stiffness": "45",
        "damping": "8",
        "frictionloss": "1.0",
        "armature": "0.02",
    },
    "pelvis_x": {
        "range": "-10 10",
        "stiffness": "70",
        "damping": "12",
        "frictionloss": "1.5",
        "armature": "0.025",
    },
    "pelvis_y": {
        "range": "-10 10",
        "stiffness": "65",
        "damping": "12",
        "frictionloss": "1.5",
        "armature": "0.025",
    },
    "pelvis_z": {
        "range": "-12 12",
        "stiffness": "70",
        "damping": "12",
        "frictionloss": "1.5",
        "armature": "0.025",
    },
}

LEG_JOINT_SPECS: Final[dict[str, XmlAttributes]] = {
    "left_hip_x": {
        "range": "-30 45",
        "damping": "2.0",
        "frictionloss": "0.3",
        "armature": "0.015",
    },
    "right_hip_x": {
        "range": "-45 30",
        "damping": "2.0",
        "frictionloss": "0.3",
        "armature": "0.015",
    },
    "left_hip_y": {
        "range": "-22 22",
        "damping": "2.0",
        "frictionloss": "0.4",
        "armature": "0.015",
    },
    "right_hip_y": {
        "range": "-22 22",
        "damping": "2.0",
        "frictionloss": "0.4",
        "armature": "0.015",
    },
    "left_hip_z": {
        "range": "-30 60",
        "damping": "2.0",
        "frictionloss": "0.4",
        "armature": "0.015",
    },
    "right_hip_z": {
        "range": "-30 60",
        "damping": "2.0",
        "frictionloss": "0.4",
        "armature": "0.015",
    },
    "left_knee_z": {
        "range": "-135 0",
        "damping": "2.5",
        "frictionloss": "0.35",
        "armature": "0.02",
    },
    "right_knee_z": {
        "range": "-135 0",
        "damping": "2.5",
        "frictionloss": "0.35",
        "armature": "0.02",
    },
    "left_ankle_y": {
        "range": "-25 25",
        "damping": "1.5",
        "frictionloss": "0.25",
        "armature": "0.01",
    },
    "right_ankle_y": {
        "range": "-25 25",
        "damping": "1.5",
        "frictionloss": "0.25",
        "armature": "0.01",
    },
    "left_ankle_z": {
        "range": "-12 12",
        "damping": "1.5",
        "frictionloss": "0.3",
        "armature": "0.01",
    },
    "right_ankle_z": {
        "range": "-12 12",
        "damping": "1.5",
        "frictionloss": "0.3",
        "armature": "0.01",
    },
}

ACTUATOR_SPECS: Final[dict[str, XmlAttributes]] = {
    "abdomen_x": {"kp": "180", "ctrlrange": "-0.18 0.18", "forcerange": "-120 120"},
    "abdomen_y": {"kp": "180", "ctrlrange": "-0.14 0.14", "forcerange": "-120 120"},
    "abdomen_z": {"kp": "180", "ctrlrange": "-0.18 0.18", "forcerange": "-120 120"},
    "pelvis_x": {"kp": "220", "ctrlrange": "-0.12 0.12", "forcerange": "-150 150"},
    "pelvis_y": {"kp": "220", "ctrlrange": "-0.10 0.10", "forcerange": "-150 150"},
    "pelvis_z": {"kp": "220", "ctrlrange": "-0.12 0.12", "forcerange": "-150 150"},
    "left_hip_x": {
        "kp": "100",
        "ctrlrange": "-0.349066 0.698132",
        "forcerange": "-180 180",
    },
    "left_hip_y": {
        "kp": "100",
        "ctrlrange": "-0.383972 0.383972",
        "forcerange": "-180 180",
    },
    "left_hip_z": {
        "kp": "100",
        "ctrlrange": "-0.523599 1.047198",
        "forcerange": "-180 180",
    },
    "left_knee_z": {
        "kp": "120",
        "ctrlrange": "-2.356194 0.000000",
        "forcerange": "-180 180",
    },
    "left_ankle_y": {
        "kp": "120",
        "ctrlrange": "-0.436332 0.436332",
        "forcerange": "-220 220",
    },
    "left_ankle_z": {
        "kp": "120",
        "ctrlrange": "-0.209440 0.209440",
        "forcerange": "-220 220",
    },
    "right_hip_x": {
        "kp": "100",
        "ctrlrange": "-0.698132 0.349066",
        "forcerange": "-180 180",
    },
    "right_hip_y": {
        "kp": "100",
        "ctrlrange": "-0.383972 0.383972",
        "forcerange": "-180 180",
    },
    "right_hip_z": {
        "kp": "100",
        "ctrlrange": "-0.523599 1.047198",
        "forcerange": "-180 180",
    },
    "right_knee_z": {
        "kp": "120",
        "ctrlrange": "-2.356194 0.000000",
        "forcerange": "-180 180",
    },
    "right_ankle_y": {
        "kp": "120",
        "ctrlrange": "-0.436332 0.436332",
        "forcerange": "-220 220",
    },
    "right_ankle_z": {
        "kp": "120",
        "ctrlrange": "-0.209440 0.209440",
        "forcerange": "-220 220",
    },
}

PASSIVE_UPPER_BODY_JOINT_SPECS: Final[dict[str, XmlAttributes]] = {
    "head_x": {
        "stiffness": "35",
        "damping": "5",
        "frictionloss": "0.5",
        "armature": "0.003",
    },
    "head_y": {
        "stiffness": "35",
        "damping": "5",
        "frictionloss": "0.5",
        "armature": "0.003",
    },
    "head_z": {
        "stiffness": "30",
        "damping": "4",
        "frictionloss": "0.5",
        "armature": "0.003",
    },
    "left_shoulder_x": {
        "stiffness": "25",
        "damping": "3",
        "frictionloss": "0.8",
        "armature": "0.006",
    },
    "left_shoulder_y": {
        "stiffness": "25",
        "damping": "3",
        "frictionloss": "0.8",
        "armature": "0.006",
    },
    "left_shoulder_z": {
        "stiffness": "25",
        "damping": "3",
        "frictionloss": "0.8",
        "armature": "0.006",
    },
    "right_shoulder_x": {
        "stiffness": "25",
        "damping": "3",
        "frictionloss": "0.8",
        "armature": "0.006",
    },
    "right_shoulder_y": {
        "stiffness": "25",
        "damping": "3",
        "frictionloss": "0.8",
        "armature": "0.006",
    },
    "right_shoulder_z": {
        "stiffness": "25",
        "damping": "3",
        "frictionloss": "0.8",
        "armature": "0.006",
    },
    "left_elbow_z": {
        "stiffness": "12",
        "damping": "2",
        "frictionloss": "0.5",
        "armature": "0.006",
    },
    "right_elbow_z": {
        "stiffness": "12",
        "damping": "2",
        "frictionloss": "0.5",
        "armature": "0.006",
    },
}


def expand_reference_gait_files(
    reference_gait_files: list[Path] | list[str] | None = None,
    reference_gait_lists: list[Path] | list[str] | None = None,
) -> list[str] | None:
    """Expand direct BVH files plus one-path-per-line list files."""
    expanded: list[str] = []
    for file_path in reference_gait_files or []:
        expanded.append(normalize_reference_path(str(file_path)))
    for list_path in reference_gait_lists or []:
        expanded.extend(read_reference_gait_list(Path(list_path)))
    return expanded or None


def read_reference_gait_list(list_path: Path) -> list[str]:
    """Read BVH paths from a text list, ignoring blank and comment lines."""
    resolved_list_path = resolve_project_path(list_path)
    paths: list[str] = []
    raw_text = resolved_list_path.read_text(encoding="utf-8")
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(normalize_reference_path(line, resolved_list_path))
    return paths


def normalize_reference_path(raw_path: str, list_path: Path | None = None) -> str:
    """Keep repo-relative BVH paths stable, with list-relative fallback."""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if (PROJECT_ROOT / candidate).exists():
        return candidate.as_posix()
    if list_path is not None:
        list_relative = list_path.parent / candidate
        if list_relative.exists():
            return relative_to_project_or_absolute(list_relative)
    return candidate.as_posix()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a user path, trying project-root-relative paths if needed."""
    expanded = Path(path).expanduser()
    if expanded.is_absolute() or expanded.exists():
        return expanded
    project_relative = PROJECT_ROOT / expanded
    if project_relative.exists():
        return project_relative
    return expanded


def default_biomechanics_env_config() -> config_dict.ConfigDict:
    """Default config passed to the MuJoCo Playground MjxEnv base class."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        action_scale=0.5,
        action_smoothing=0.5,
        command_profile="standard",
        reference_gait="none",
        reference_gait_file=None,
        reference_target_observation=False,
        xml_path=None,
        legacy_action_prior=False,
        command_resample_steps=500,
        tracking_sigma=0.25,
        action_noise_std=0.03,
        episode_bias_std=0.02,
        rfi_torque_limit=2.0,
        rao_torque_limit=2.0,
        enable_erfi=True,
        init_qpos_file=None,
        impl="jax",
    )


def default_biomechanics_ppo_config() -> config_dict.ConfigDict:
    """Default Brax PPO hyperparameters for the biomechanics joystick env.

    These are training defaults, not runtime state. `train.py` may override a
    subset from CLI/TrainConfig, but this function is the single source of truth
    for the base PPO schedule, optimizer settings, rollout shape, and network
    architecture. JAX is imported locally so lightweight config users do not pay
    the JAX import cost unless they actually build the PPO config.
    """
    import jax

    return config_dict.create(
        num_timesteps=50_000_000,
        num_evals=10,
        num_envs=1024,
        num_eval_envs=32,
        episode_length=500,
        action_repeat=1,
        learning_rate=3e-4,
        entropy_cost=3e-3,
        discounting=0.97,
        unroll_length=20,
        batch_size=512,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        normalize_observations_std_eps=1e-3,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            activation=jax.nn.silu,
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
    )


def relative_to_project_or_absolute(path: Path) -> str:
    """Prefer repo-relative paths in config.json when possible."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# Dataclass ovde nije safety/filler sloj. Treba nam zato sto:
# 1. daje kratak konstruktor: EnvConfig(env_version="hardcore")
# 2. dozvoljava load iz checkpoint-a: EnvConfig(**saved_config)
# 3. daje __dict__ za snimanje konfiguracije u runs/config.json
@dataclass
class EnvConfig:
    # standard -> flat terrain, hardcore -> rough terrain.
    env_version: str = "standard"

    # "jax" je default jer lokalni "warp" backend trenutno puca na verzijskom
    # konfliktu warp/mujoco-mjx. Kad se verzije srede, warp moze biti brzi.
    playground_impl: str = "jax"

    # "standard" je pun joystick zadatak: napred/nazad, lateralno i yaw.
    # "forward" ostaje dostupan samo kao bootstrap curriculum.
    command_profile: str = "standard"

    # "sine" dodaje rucno dizajniranu ciklicnu referentnu putanju za noge.
    # "bvh" koristi jednu BVH animaciju kao motion-imitation prior.
    # "none" koristi samo task/style reward bez explicit pose imitation.
    reference_gait: str = "none"
    reference_gait_file: str | list[str] | None = None
    reference_target_observation: bool = False

    # Opcioni konkretan XML model. Korisno za nastavak starog checkpoint-a kada
    # je globalni generated XML version u kodu vec promenjen.
    xml_path: str | None = None

    # Za nastavak V10/slow checkpoint-a pre Unitree-style action prior-a.
    legacy_action_prior: bool = False

    # Referentni humanoid walking setup filtrira targete pre PD kontrole.
    # 0.5 znaci: pola nova akcija politike, pola prethodni target.
    action_smoothing: float = 0.5

    # Opcioni MJDATA/QPOS fajl za pocetnu pozu, npr. neutralni polucucanj.
    # None koristi built-in standing-home pozu.
    init_qpos_file: str | None = None

    # Stabilniji physics step za biomehanicki model.
    # Default: sim_dt=0.005 -> 4 substep-a po policy koraku.
    accurate_physics: bool = True


# Dataclass iz istog razloga kao EnvConfig: trening skripti treba kratak
# konstruktor, checkpoint cuva __dict__, a load moze da uradi TrainConfig(**...).
@dataclass
class TrainConfig:
    # Fiksan seed daje ponovljivost dok uporedjujemo izmene.
    seed: int = 7

    # None znaci: koristi tuned vrednost iz default_biomechanics_ppo_config().
    num_timesteps: int | None = None
    num_evals: int | None = None
    num_envs: int | None = None
    num_eval_envs: int | None = None
    episode_length: int | None = None
    unroll_length: int | None = None
    batch_size: int | None = None
    num_minibatches: int | None = None
    num_updates_per_batch: int | None = None
    learning_rate: float | None = None
    no_erfi: bool = False
    no_domain_randomization: bool = False
    save_checkpoints: bool = True
    checkpoint_out: str | None = None
    resume_from: str | None = None
    run_tag: str | None = None
    debug_run: bool = False
    bare: bool = False
