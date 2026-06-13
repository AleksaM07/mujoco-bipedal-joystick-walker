import contextlib
import importlib.util
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

from config import BIOMECH_DIR, PROJECT_ROOT


GENERATED_MODEL_DIR = PROJECT_ROOT / "generated_models"
SCENE_XML_VERSION = "trainfast_v8"
FOOT_BODY_NAMES = {"left_foot", "right_foot"}
LOCKED_TORSO_JOINTS = (
    "abdomen_x",
    "abdomen_y",
    "abdomen_z",
    "pelvis_x",
    "pelvis_y",
    "pelvis_z",
)
_GENERATOR_CACHE = None

LOCOMOTION_ACTUATED_JOINTS = (
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
LEG_FRICTION_LOSS = {
    "hip": 0.3,
    "knee": 0.8,
    "ankle": 0.6,
}
PASSIVE_JOINT_STIFFNESS = {
    "head": 12.0,
    "abdomen": 350.0,
    "pelvis": 350.0,
    "shoulder": 25.0,
    "elbow": 12.0,
    "wrist": 6.0,
}


@dataclass(frozen=True)
class HumanSpec:
    mass: float = 75.0
    height: float = 1.80
    sex: str = "male"
    alpha: float = 1.0

    @property
    def file_stem(self) -> str:
        mass = int(round(self.mass))
        height = int(round(self.height * 100))
        return f"human_{self.sex}_{height}cm_{mass}kg"


@contextlib.contextmanager
def working_directory(path: Path):
    """Privremeno promeni cwd jer generator cita CSV fajlove relativno."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_generator():
    """Ucita generator iz susednog `mujoco-biomechanics` repozitorijuma."""
    global _GENERATOR_CACHE
    if _GENERATOR_CACHE is not None:
        return _GENERATOR_CACHE

    generator_path = BIOMECH_DIR / "generate_human_model.py"
    spec = importlib.util.spec_from_file_location(
        "mujoco_biomechanics_generator",
        generator_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GENERATOR_CACHE = module.generate_human_model
    return _GENERATOR_CACHE


def generate_base_human_xml(spec: HumanSpec) -> Path:
    """Generise osnovni human XML sa antropometrijskim parametrima."""
    GENERATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_MODEL_DIR / f"{spec.file_stem}_base.xml"
    if output_path.exists():
        return output_path

    generate_human_model = load_generator()
    with working_directory(BIOMECH_DIR):
        generate_human_model(
            filename=str(output_path),
            mass=spec.mass,
            height=spec.height,
            sex=spec.sex,
            alpha=spec.alpha,
        )
    return output_path


def build_trainable_scene_xml(env_version: str, spec: HumanSpec) -> Path:
    """Napravi human XML sa aktuatorima i training-friendly kolizijama."""
    output_path = GENERATED_MODEL_DIR / (
        f"{spec.file_stem}_{env_version}_{SCENE_XML_VERSION}.xml"
    )
    if output_path.exists():
        return output_path

    base_path = generate_base_human_xml(spec)
    tree = ET.parse(base_path)
    root = tree.getroot()

    remove_generated_floor(root)
    ensure_compiler(root)
    ensure_option(root)
    ensure_visual(root)
    configure_joint_passive_properties(root)
    add_terrain(root, env_version)
    set_training_collision_filters(root)
    add_stable_foot_contacts(root)
    add_torso_joint_locks(root)
    add_actuators(root)
    add_keyframe_ctrl(root)

    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def remove_generated_floor(root: ET.Element) -> None:
    """Ukloni floor iz generatora da terrain kontrolise nas env."""
    worldbody = root.find("worldbody")
    for geom in list(worldbody.findall("geom")):
        if geom.get("name") == "floor":
            worldbody.remove(geom)


def ensure_compiler(root: ET.Element) -> None:
    """Doda compiler parametre koji olaksavaju MJX ucitavanje."""
    if root.find("compiler") is None:
        root.insert(0, ET.Element("compiler", angle="degree", inertiafromgeom="false"))
        return

    compiler = root.find("compiler")
    compiler.set("angle", "degree")
    compiler.set("inertiafromgeom", "false")


def ensure_option(root: ET.Element) -> None:
    """Postavi timestep i solver opcije za locomotion trening."""
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.01")
    option.set("integrator", "implicitfast")
    option.set("iterations", "4")
    option.set("ls_iterations", "5")


def ensure_visual(root: ET.Element) -> None:
    """Doda viewer podesavanja bez menjanja dinamike."""
    if root.find("statistic") is None:
        ET.SubElement(root, "statistic", center="0 0 0.9", extent="2.2")
    if root.find("visual") is not None:
        return

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", azimuth="120", elevation="-20")
    ET.SubElement(visual, "headlight", diffuse=".8 .8 .8", ambient=".2 .2 .2")


def configure_joint_passive_properties(root: ET.Element) -> None:
    """Doda joint damping, armature, friction i pasivne opruge."""
    worldbody = root.find("worldbody")
    for joint in worldbody.iter("joint"):
        if joint.get("type") == "free":
            continue

        joint_name = joint.get("name", "")
        if joint_name in LOCOMOTION_ACTUATED_JOINTS:
            joint.set("damping", "1.5")
            joint.set("armature", "0.006")
            joint.set("frictionloss", str(actuated_joint_friction(joint_name)))
            continue

        joint.set("damping", str(passive_joint_damping(joint_name)))
        joint.set("armature", str(passive_joint_armature(joint_name)))
        joint.set("frictionloss", str(passive_joint_friction(joint_name)))
        joint.set("stiffness", str(passive_joint_stiffness(joint_name)))
        joint.set("springref", "0")


def actuated_joint_friction(joint_name: str) -> float:
    """Vraca Berkeley-like frictionloss za kontrolisane zglobove nogu."""
    if "knee" in joint_name:
        return LEG_FRICTION_LOSS["knee"]
    if "ankle" in joint_name:
        return LEG_FRICTION_LOSS["ankle"]
    return LEG_FRICTION_LOSS["hip"]


def passive_joint_stiffness(joint_name: str) -> float:
    """Vraca pasivnu oprugu za zglobove koje policy ne kontrolise."""
    for name_part, stiffness in PASSIVE_JOINT_STIFFNESS.items():
        if name_part in joint_name:
            return stiffness
    return 5.0


def passive_joint_damping(joint_name: str) -> float:
    """Vraca pasivno prigusenje za nekontrolisane zglobove."""
    if "abdomen" in joint_name or "pelvis" in joint_name:
        return 8.0
    if "shoulder" in joint_name:
        return 3.0
    return 2.0


def passive_joint_armature(joint_name: str) -> float:
    """Dodaje numericku inerciju da pasivni zglobovi ne budu previse mlitavi."""
    if "abdomen" in joint_name or "pelvis" in joint_name:
        return 0.02
    if "shoulder" in joint_name or "elbow" in joint_name:
        return 0.006
    return 0.003


def passive_joint_friction(joint_name: str) -> float:
    """Dodaje Coulomb friction za pasivne zglobove."""
    if "abdomen" in joint_name or "pelvis" in joint_name:
        return 2.0
    if "shoulder" in joint_name:
        return 0.8
    if "elbow" in joint_name:
        return 0.5
    return 0.3


def add_terrain(root: ET.Element, env_version: str) -> None:
    """Doda standard flat ili hardcore rough terrain u worldbody."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        width="300",
        height="300",
        rgb1="1 1 1",
        rgb2=".85 .85 .85",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="groundplane",
        texrepeat="6 6",
        texuniform="true",
    )

    worldbody = root.find("worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.01",
        material="groundplane",
        friction="0.8",
        condim="3",
    )
    if env_version == "hardcore":
        add_rough_blocks(worldbody)


def add_rough_blocks(worldbody: ET.Element) -> None:
    """Doda deterministicke niske prepreke za hardcore varijantu."""
    for index in range(12):
        x_pos = 0.7 + 0.45 * index
        y_pos = -0.45 if index % 2 else 0.35
        height = 0.025 + 0.01 * (index % 3)
        ET.SubElement(
            worldbody,
            "geom",
            name=f"rough_block_{index}",
            type="box",
            pos=f"{x_pos} {y_pos} {height}",
            size=f"0.18 0.22 {height}",
            rgba=".55 .55 .55 1",
            friction="0.9",
            condim="3",
        )


def set_training_collision_filters(root: ET.Element) -> None:
    """Ostavi fizicku koliziju samo za teren i stopala."""
    worldbody = root.find("worldbody")
    for geom in worldbody.findall("geom"):
        mark_terrain_geom(geom)
    for body in worldbody.findall("body"):
        mark_body_collision(body, in_foot=False)


def mark_body_collision(body: ET.Element, in_foot: bool) -> None:
    """Rekurzivno oznaci body geometrije kao visual-only ili foot collision."""
    current_is_foot = in_foot or body.get("name") in FOOT_BODY_NAMES
    for geom in body.findall("geom"):
        if current_is_foot:
            mark_foot_geom(geom)
        else:
            mark_visual_only_geom(geom)
    for child in body.findall("body"):
        mark_body_collision(child, current_is_foot)


def mark_terrain_geom(geom: ET.Element) -> None:
    """Teren prima kontakt od stopala, ali ne pravi nepotrebne parove."""
    geom.set("contype", "1")
    geom.set("conaffinity", "0")
    geom.set("condim", "3")


def mark_foot_geom(geom: ET.Element) -> None:
    """Stopala su jedini delovi humanoida koji kolidiraju sa terenom."""
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    geom.set("condim", "3")


def mark_visual_only_geom(geom: ET.Element) -> None:
    """Geometrija ostaje vidljiva, ali ne ulazi u contact solver."""
    geom.set("contype", "0")
    geom.set("conaffinity", "0")


def add_stable_foot_contacts(root: ET.Element) -> None:
    """Zameni zaobljenu foot kapsulu stabilnijim box djonom za kontakt."""
    worldbody = root.find("worldbody")
    for body in worldbody.iter("body"):
        if body.get("name") not in FOOT_BODY_NAMES:
            continue

        for geom in body.findall("geom"):
            mark_visual_only_geom(geom)

        ET.SubElement(
            body,
            "geom",
            name=f"{body.get('name')}_sole",
            type="box",
            pos="0.09 -0.045 0",
            size="0.145 0.012 0.075",
            rgba="0.1 0.1 0.1 0.35",
            friction="1.0 0.01 0.001",
            solref="0.02 1",
            solimp="0.85 0.95 0.005",
            contype="1",
            conaffinity="1",
            condim="3",
        )


def add_torso_joint_locks(root: ET.Element) -> None:
    """Zakljuca spine/pelvis lanac da gornji deo bude Berkeley-like rigid."""
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    existing = {item.get("joint1") for item in equality.findall("joint")}
    for joint_name in LOCKED_TORSO_JOINTS:
        if joint_name in existing:
            continue
        ET.SubElement(
            equality,
            "joint",
            name=f"lock_{joint_name}",
            joint1=joint_name,
            polycoef="0 1 0 0 0",
            solref="0.005 1",
            solimp="0.95 0.99 0.001",
        )


def add_actuators(root: ET.Element) -> None:
    """Doda position servo aktuatore za zglobove koje policy kontrolise."""
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    joints = {
        joint.get("name"): joint
        for joint in root.find("worldbody").iter("joint")
    }
    existing = {item.get("joint") for item in actuator}
    for joint_name in LOCOMOTION_ACTUATED_JOINTS:
        if joint_name in existing:
            continue
        ctrlrange = actuator_ctrlrange(joints[joint_name])
        ET.SubElement(
            actuator,
            "position",
            name=f"{joint_name}_position",
            joint=joint_name,
            kp="35",
            ctrllimited="true",
            ctrlrange=ctrlrange,
            forcelimited="true",
            forcerange="-80 80",
        )


def actuator_ctrlrange(joint: ET.Element) -> str:
    """Pretvori degree joint range iz generatora u radian actuator ctrlrange."""
    lower_deg, upper_deg = (float(value) for value in joint.get("range").split())
    return f"{math.radians(lower_deg):.6f} {math.radians(upper_deg):.6f}"


def add_keyframe_ctrl(root: ET.Element) -> None:
    """Doda zero ctrl u keyframe-ove jer model ima aktuatore."""
    ctrl = " ".join("0" for _ in LOCOMOTION_ACTUATED_JOINTS)
    keyframe = root.find("keyframe")
    if keyframe is None:
        return
    for key in keyframe.findall("key"):
        key.set("ctrl", ctrl)


def validate_xml(path: Path) -> None:
    """Ucita XML jednom da odmah uhvatimo MJCF greske."""
    mujoco.MjModel.from_xml_path(str(path))
