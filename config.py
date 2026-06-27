from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BIOMECH_DIR = WORKSPACE_ROOT / "mujoco-biomechanics"
RUNS_DIR = PROJECT_ROOT / "runs"


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

    # None znaci: koristi tuned vrednost iz biomechanics_ppo_config().
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
