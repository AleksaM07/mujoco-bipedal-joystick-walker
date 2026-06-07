import contextlib
import importlib.util
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

from config import BIOMECH_DIR, PROJECT_ROOT


GENERATED_MODEL_DIR = PROJECT_ROOT / "generated_models"
SCENE_XML_VERSION = "trainfast_v3"
FOOT_BODY_NAMES = {"left_foot", "right_foot"}
_GENERATOR_CACHE = None

LOCOMOTION_ACTUATED_JOINTS = (
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
)


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
    add_passive_joint_damping(root)
    add_terrain(root, env_version)
    set_training_collision_filters(root)
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
        root.insert(0, ET.Element("compiler", angle="radian", inertiafromgeom="false"))


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


def add_passive_joint_damping(root: ET.Element) -> None:
    """Doda osnovno prigusenje da pasivni delovi tela ne budu potpuno mlitavi."""
    worldbody = root.find("worldbody")
    for joint in worldbody.iter("joint"):
        if joint.get("type") != "free":
            joint.set("damping", "1.0")


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


def add_actuators(root: ET.Element) -> None:
    """Doda position servo aktuatore za zglobove koje policy kontrolise."""
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {item.get("joint") for item in actuator}
    for joint_name in LOCOMOTION_ACTUATED_JOINTS:
        if joint_name in existing:
            continue
        ET.SubElement(
            actuator,
            "position",
            name=f"{joint_name}_position",
            joint=joint_name,
            kp="35",
            ctrllimited="true",
            ctrlrange="-3.14 3.14",
            forcelimited="true",
            forcerange="-80 80",
        )


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
