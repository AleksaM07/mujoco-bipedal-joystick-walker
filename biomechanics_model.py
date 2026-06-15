import contextlib
import hashlib
import importlib.util
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

from config import BIOMECH_DIR, PROJECT_ROOT


GENERATED_MODEL_DIR = PROJECT_ROOT / "generated_models"
SCENE_XML_VERSION = "trainfast_v15"

# Cache for the generator module to avoid reimporting
_GENERATOR_CACHE = None

FOOT_BODY_NAMES = ("left_foot", "right_foot")

TRUNK_ACTUATED_JOINTS = (
    "abdomen_x",
    "abdomen_y",
    "abdomen_z",
    "pelvis_x",
    "pelvis_y",
    "pelvis_z",
)

LEG_ACTUATED_JOINTS = (
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

LOCOMOTION_ACTUATED_JOINTS = TRUNK_ACTUATED_JOINTS + LEG_ACTUATED_JOINTS

TRUNK_JOINT_SPECS = {
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

ACTUATOR_SPECS = {
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
        "ctrlrange": "-0.698132 0.872665",
        "forcerange": "-180 180",
    },
    "left_hip_z": {
        "kp": "100",
        "ctrlrange": "-0.523599 1.745329",
        "forcerange": "-180 180",
    },
    "left_knee_z": {
        "kp": "120",
        "ctrlrange": "-2.617994 0.000000",
        "forcerange": "-180 180",
    },
    "left_ankle_y": {
        "kp": "120",
        "ctrlrange": "-0.523599 0.523599",
        "forcerange": "-220 220",
    },
    "left_ankle_z": {
        "kp": "120",
        "ctrlrange": "-0.349066 0.523599",
        "forcerange": "-220 220",
    },
    "right_hip_x": {
        "kp": "100",
        "ctrlrange": "-0.698132 0.349066",
        "forcerange": "-180 180",
    },
    "right_hip_y": {
        "kp": "100",
        "ctrlrange": "-0.698132 0.872665",
        "forcerange": "-180 180",
    },
    "right_hip_z": {
        "kp": "100",
        "ctrlrange": "-0.523599 1.745329",
        "forcerange": "-180 180",
    },
    "right_knee_z": {
        "kp": "120",
        "ctrlrange": "-2.617994 0.000000",
        "forcerange": "-180 180",
    },
    "right_ankle_y": {
        "kp": "120",
        "ctrlrange": "-0.523599 0.523599",
        "forcerange": "-220 220",
    },
    "right_ankle_z": {
        "kp": "120",
        "ctrlrange": "-0.349066 0.523599",
        "forcerange": "-220 220",
    },
}

PASSIVE_UPPER_BODY_JOINT_SPECS = {
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
    """Ucita generator iz susednog `mujoco-biomechanics` repozitorijuma (cached)."""
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


def spec_to_hash(spec: HumanSpec) -> str:
    """Generiše hash od HumanSpec za cache keying."""
    key = f"{spec.mass}_{spec.height}_{spec.sex}_{spec.alpha}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def generate_base_human_xml(spec: HumanSpec) -> Path:
    """Generise osnovni human XML sa antropometrijskim parametrima (cached)."""
    GENERATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_MODEL_DIR / f"{spec.file_stem}_base.xml"
    
    # Ako XML vec postoji, ne regenerisuj
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
    """Napravi human XML sa aktuatorima, senzorima i izabranim terenom (cached)."""
    output_path = GENERATED_MODEL_DIR / (
        f"{spec.file_stem}_{env_version}_{SCENE_XML_VERSION}.xml"
    )
    
    # Ako XML vec postoji, vrati ga bez regenerisanja
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
    tune_passive_upper_body_joints(root)
    unlock_trunk_joints(root)
    remove_trunk_equality_locks(root)
    set_training_collision_filters(root)
    add_stable_foot_contacts(root)
    add_terrain(root, env_version)
    add_actuators(root)
    add_keyframe_ctrl(root)

    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    # Validacija se preskace ako je XML vec generisan (mala verovatnoca greske)
    # Ako trebas validaciju, uglasi --validate-xml flag
    return output_path


def remove_generated_floor(root: ET.Element) -> None:
    """Ukloni floor iz generatora da terrain kontrolise nas env."""
    worldbody = root.find("worldbody")
    for geom in list(worldbody.findall("geom")):
        if geom.get("name") == "floor":
            worldbody.remove(geom)


def ensure_compiler(root: ET.Element) -> None:
    """Doda compiler parametre koji olaksavaju MJX ucitavanje."""
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "degree")
    compiler.set("inertiafromgeom", "false")


def ensure_option(root: ET.Element) -> None:
    """Postavi timestep i solver opcije za locomotion trening."""
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.005")
    option.set("integrator", "implicitfast")


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
        if joint.get("type") == "free":
            continue
        joint.set("damping", "1.0")


def tune_passive_upper_body_joints(root: ET.Element) -> None:
    """Ukruti vrat i ruke da pasivni upper body ne visi kao slobodna masa."""
    worldbody = root.find("worldbody")
    for joint in worldbody.iter("joint"):
        spec = PASSIVE_UPPER_BODY_JOINT_SPECS.get(joint.get("name"))
        if spec is None:
            continue
        for key, value in spec.items():
            joint.set(key, value)
        joint.set("springref", "0")


def unlock_trunk_joints(root: ET.Element) -> None:
    """Vrati abdomen/pelvis iz skoro fiksiranog stanja u mali kontrolisani opseg."""
    worldbody = root.find("worldbody")
    for joint in worldbody.iter("joint"):
        joint_name = joint.get("name")
        spec = TRUNK_JOINT_SPECS.get(joint_name)
        if spec is None:
            continue
        joint.set("limited", "true")
        for key, value in spec.items():
            joint.set(key, value)
        joint.set("springref", "0")


def remove_trunk_equality_locks(root: ET.Element) -> None:
    """Ukloni generator lockove koji bi pregazili trunk aktuatore."""
    equality = root.find("equality")
    if equality is None:
        return

    for constraint in list(equality):
        name = constraint.get("name", "")
        joint_name = constraint.get("joint1", "")
        if name.startswith(("lock_abdomen_", "lock_pelvis_")):
            equality.remove(constraint)
        elif joint_name in TRUNK_ACTUATED_JOINTS:
            equality.remove(constraint)

    if len(equality) == 0:
        root.remove(equality)


def set_training_collision_filters(root: ET.Element) -> None:
    """Ostavi kontakt samo za teren i dodate sole geometrije."""
    worldbody = root.find("worldbody")
    for geom in worldbody.findall("geom"):
        mark_terrain_geom(geom)
    for body in worldbody.findall("body"):
        mark_body_collision(body)


def mark_body_collision(body: ET.Element, in_foot: bool = False) -> None:
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
    """Postojece foot geometrije ostaju fizicke, ali sole nosi glavni kontakt."""
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    geom.set("condim", "3")


def mark_visual_only_geom(geom: ET.Element) -> None:
    """Geometrija ostaje vidljiva, ali ne ulazi u contact solver."""
    geom.set("contype", "0")
    geom.set("conaffinity", "0")


def add_stable_foot_contacts(root: ET.Element) -> None:
    """Dodaj stabilan box djon za svaki foot body."""
    worldbody = root.find("worldbody")
    for body in worldbody.iter("body"):
        if body.get("name") not in FOOT_BODY_NAMES:
            continue

        for geom in body.findall("geom"):
            mark_visual_only_geom(geom)

        sole_name = f"{body.get('name')}_sole"
        if body.find(f"geom[@name='{sole_name}']") is not None:
            continue

        ET.SubElement(
            body,
            "geom",
            name=sole_name,
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


def add_actuators(root: ET.Element) -> None:
    """Doda position servo aktuatore za zglobove koje policy kontrolise."""
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = {item.get("joint") for item in actuator}
    for joint_name in LOCOMOTION_ACTUATED_JOINTS:
        if joint_name in existing:
            continue
        spec = ACTUATOR_SPECS[joint_name]
        ET.SubElement(
            actuator,
            "position",
            name=f"{joint_name}_position",
            joint=joint_name,
            kp=spec["kp"],
            ctrllimited="true",
            ctrlrange=spec["ctrlrange"],
            forcelimited="true",
            forcerange=spec["forcerange"],
        )


def add_keyframe_ctrl(root: ET.Element) -> None:
    """Doda zero ctrl u keyframe-ove jer sada model ima aktuatore."""
    ctrl = " ".join("0" for _ in LOCOMOTION_ACTUATED_JOINTS)
    keyframe = root.find("keyframe")
    if keyframe is None:
        return
    for key in keyframe.findall("key"):
        key.set("ctrl", ctrl)


def validate_xml(path: Path) -> None:
    """Ucita XML jednom da odmah uhvatimo MJCF greske."""
    mujoco.MjModel.from_xml_path(str(path))
