from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BIOMECH_DIR = WORKSPACE_ROOT / "mujoco-biomechanics"
RUNS_DIR = PROJECT_ROOT / "runs"

KEY_SPACE = 32
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_DOWN = 264
KEY_UP = 265
KEY_A = 65
KEY_D = 68
KEY_E = 69
KEY_Q = 81
KEY_S = 83
KEY_W = 87
KEY_NUMPAD_2 = 322
KEY_NUMPAD_4 = 324
KEY_NUMPAD_6 = 326
KEY_NUMPAD_7 = 327
KEY_NUMPAD_8 = 328
KEY_NUMPAD_9 = 329


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


def resolve_project_path(path: Path) -> Path:
    """Resolve a user path, trying project-root-relative paths if needed."""
    expanded = path.expanduser()
    if expanded.is_absolute() or expanded.exists():
        return expanded
    project_relative = PROJECT_ROOT / expanded
    if project_relative.exists():
        return project_relative
    return expanded


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
    # "biomechanics" je pravi projekat: human iz mujoco-biomechanics.
    # "prototip" je ono sto smo prvo napravili sa Berkeley robotom iz Playground-a.
    env_source: str = "biomechanics"

    # standard -> flat terrain, hardcore -> rough terrain.
    env_version: str = "standard"

    # "jax" je default jer lokalni "warp" backend trenutno puca na verzijskom
    # konfliktu warp/mujoco-mjx. Kad se verzije srede, warp moze biti brzi.
    playground_impl: str = "jax" #aleksa moras da promenis ovo obavezno !!! i da istestiras
    prototype_flat_env: str = "BerkeleyHumanoidJoystickFlatTerrain"
    prototype_hardcore_env: str = "BerkeleyHumanoidJoystickRoughTerrain"

    # Korak promene joystick komande u viewer-u po pritisku strelice/WASD.
    command_change_rate: float = 0.1

    # "standard" je pun joystick zadatak: napred/nazad, lateralno i yaw.
    # "forward" ostaje dostupan samo kao bootstrap curriculum.
    command_profile: str = "standard"

    # "sine" dodaje rucno dizajniranu ciklicnu referentnu putanju za noge.
    # "bvh" koristi jednu BVH animaciju kao motion-imitation prior.
    # "none" koristi samo task/style reward bez explicit pose imitation.
    reference_gait: str = "none"
    reference_gait_file: str | list[str] | None = None

    # Referentni humanoid walking setup filtrira targete pre PD kontrole.
    # 0.5 znaci: pola nova akcija politike, pola prethodni target.
    action_smoothing: float = 0.5

    # Opcioni MJDATA/QPOS fajl za pocetnu pozu, npr. neutralni polucucanj.
    # None koristi built-in standing-home pozu.
    init_qpos_file: str | None = None

    # Stabilniji physics step za biomehanicki model.
    # Default: sim_dt=0.005 -> 4 substep-a po policy koraku.
    accurate_physics: bool = True

    def prototype_env_name(self) -> str:
        """Mapira standard/hardcore na Berkeley prototip env."""
        if self.env_version == "standard":
            return self.prototype_flat_env
        if self.env_version == "hardcore":
            return self.prototype_hardcore_env
        raise ValueError("env_version mora biti 'standard' ili 'hardcore'.")


# Dataclass iz istog razloga kao EnvConfig: trening skripti treba kratak
# konstruktor, checkpoint cuva __dict__, a load moze da uradi TrainConfig(**...).
@dataclass
class TrainConfig:
    # Fiksan seed daje ponovljivost dok uporedjujemo izmene.
    seed: int = 7

    # None znaci: koristi tuned vrednost iz MuJoCo Playground locomotion_params.
    # Za Berkeley humanoid Playground default je PPO sa 150M stepova, 8192
    # paralelna env-a, policy MLP (512, 256, 128) i privileged critic obs.
    num_timesteps: int | None = None
    num_evals: int | None = None
    num_envs: int | None = None
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

    # PPO je izabran jer Playground vec ima podesen Brax/MJX PPO config za
    # Berkeley humanoid joystick env. SAC/TD3 nisu odbaceni teorijski, nego nisu
    # tuned/provided put za ovaj locomotion env u Playground paketu koji koristimo.
    algorithm: str = "ppo"
