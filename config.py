from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BIOMECH_DIR = WORKSPACE_ROOT / "mujoco-biomechanics"
RUNS_DIR = PROJECT_ROOT / "runs"


# Ovo vazi samo za nas biomech prototip, ne za MuJoCo Playground env.
# Nije tvrdnja da ruke nisu bitne za hod; ovo je prva redukcija action space-a.
# Kontrolisemo trup, kukove, kolena i clanke jer su oni minimum za lokomociju.
# Ruke/vrat/sake ostaju pasivni dok ne imamo razlog da prosirimo kontrolu.
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


# Dataclass ovde nije safety/filler sloj. Treba nam zato sto:
# 1. daje kratak konstruktor: EnvConfig(env_version="hardcore")
# 2. dozvoljava load iz checkpoint-a: EnvConfig(**saved_config)
# 3. daje __dict__ za snimanje konfiguracije u runs/config.json
@dataclass
class EnvConfig:
    # Primarni cilj projekta: postojece MuJoCo Playground humanoid joystick okruzenje.
    env_backend: str = "playground"

    # standard -> flat terrain, hardcore -> rough terrain.
    # Dostupni humanoid/robot joystick env-ovi u instaliranom Playground paketu:
    # ApolloJoystickFlatTerrain, BerkeleyHumanoidJoystickFlatTerrain,
    # BerkeleyHumanoidJoystickRoughTerrain, G1JoystickFlatTerrain,
    # G1JoystickRoughTerrain, H1JoystickGaitTracking,
    # T1JoystickFlatTerrain, T1JoystickRoughTerrain, Op3Joystick.
    # Berkeley biramo jer jedini u imenu eksplicitno nosi "HumanoidJoystick",
    # pa je najdirektniji match za temu projekta.
    env_version: str = "standard"

    # "jax" je default jer lokalni "warp" backend trenutno puca na verzijskom
    # konfliktu warp/mujoco-mjx. Kad se verzije srede, warp moze biti brzi.
    playground_impl: str = "jax" #aleksa moras da promenis ovo obavezno !!! i da istestiras
    playground_flat_env: str = "BerkeleyHumanoidJoystickFlatTerrain"
    playground_hardcore_env: str = "BerkeleyHumanoidJoystickRoughTerrain"

    # Biomech-only: raspon odraslih osoba za pocetnu domain randomization probu.
    # Ovo nije finalna antropometrijska statistika; treba ga zameniti percentilima
    # iz biomehanickih tabela kad API za direktnu randomizaciju modela bude spreman.
    # 55-95 kg i 1.55-1.95 m su namerno siroki, ali jos uvek realni trening opseg.
    min_mass: float = 55.0
    max_mass: float = 95.0
    min_height: float = 1.55
    max_height: float = 1.95

    sex_probability_male: float = 0.5

    # Biomech-only command sampling. Playground ima svoje lin_vel_x/lin_vel_y/
    # ang_vel_yaw opsege u internom config-u. Ovde ostavljamo slican opseg:
    # x pokriva hod unazad i malo brzi hod unapred, y levo/desno, yaw okretanje.
    command_x_range: tuple[float, float] = (-1.0, 1.2)
    command_y_range: tuple[float, float] = (-0.8, 0.8)
    command_yaw_range: tuple[float, float] = (-1.2, 1.2)

    # Biomech-only simulacija. frame_skip=5 uz MuJoCo timestep 0.002 daje 100 Hz
    # control rate, sto je tipicno dovoljno gusto za humanoidnu kontrolu.
    frame_skip: int = 5 # aleksa !!!

    # 12 s daje epizodu dovoljno dugu da se vidi stabilan hod, a ne samo start.
    episode_seconds: float = 12.0

    # Biomech-only health check: dozvoljavamo sirok odnos visine trupa prema
    # ukupnoj visini, jer randomizovani modeli nece imati identicnu nominalnu pozu.
    healthy_min_height_fraction: float = 0.45
    healthy_max_height_fraction: float = 1.35

    # Korak promene joystick komande u viewer-u po pritisku strelice/WASD.
    command_change_rate: float = 0.1

    # Biomech-only aktuatori i perturbacije. 50 Nm je pocetni limit za zglobove
    # nogu/trupa; treba ga kalibrisati nakon prvih stabilnih simulacija.
    torque_limit: float = 50.0

    # Sluzi za robustnost: mali trenutni sum u momentima u svakom step-u.
    torque_noise_std: float = 1.0

    # Konstantni bias po epizodi modeluje sistematsku gresku aktuatora.
    episode_torque_bias_std: float = 2.0

    # Spoljasnji push se desava retko, ali dovoljno cesto da politika nauci oporavak.
    push_probability: float = 0.015
    push_force_std: float = 25.0

    # Biomech-only reward skale. Playground koristi svoj reward_config.
    # velocity_sigma odredjuje koliko strogo nagradjujemo pracenje komande.
    velocity_sigma: float = 0.45

    # Mala nagrada za ostajanje u zdravom/stabilnom stanju.
    healthy_reward: float = 0.5

    # Kazne za jake i nagle akcije, da politika ne nauci trzanje.
    control_cost: float = 0.001
    action_rate_cost: float = 0.01
