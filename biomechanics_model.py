import contextlib
import importlib.util
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

from config import (
    ACTUATOR_SPECS,
    BIOMECH_DIR,
    DEFAULT_HUMAN_ALPHA,
    DEFAULT_HUMAN_HEIGHT_M,
    DEFAULT_HUMAN_MASS_KG,
    DEFAULT_HUMAN_SEX,
    FLOOR_GEOM_ATTRIBUTES,
    FOOT_BODY_NAMES,
    GENERATED_MODEL_DIR,
    GENERATOR_MODULE_NAME,
    GROUND_MATERIAL_ATTRIBUTES,
    GROUND_TEXTURE_ATTRIBUTES,
    LEG_ACTUATED_JOINTS,
    LEG_JOINT_SPECS,
    LOCOMOTION_ACTUATED_JOINTS,
    PASSIVE_UPPER_BODY_JOINT_SPECS,
    ROUGH_BLOCK_BASE_HEIGHT,
    ROUGH_BLOCK_COUNT,
    ROUGH_BLOCK_GEOM_ATTRIBUTES,
    ROUGH_BLOCK_HEIGHT_PERIOD,
    ROUGH_BLOCK_HEIGHT_STEP,
    ROUGH_BLOCK_SIZE_X,
    ROUGH_BLOCK_SIZE_Y,
    ROUGH_BLOCK_START_X,
    ROUGH_BLOCK_X_STEP,
    ROUGH_BLOCK_Y_EVEN,
    ROUGH_BLOCK_Y_ODD,
    SCENE_XML_VERSION,
    SOLE_CONTACT_GEOM_ATTRIBUTES,
    TRUNK_ACTUATED_JOINTS,
    TRUNK_JOINT_SPECS,
)


_GENERATOR_CACHE = None


@dataclass(frozen=True)
class HumanSpec:
    mass: float = DEFAULT_HUMAN_MASS_KG
    height: float = DEFAULT_HUMAN_HEIGHT_M
    sex: str = DEFAULT_HUMAN_SEX
    alpha: float = DEFAULT_HUMAN_ALPHA

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
        GENERATOR_MODULE_NAME,
        generator_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GENERATOR_CACHE = module.generate_human_model
    return _GENERATOR_CACHE


def generate_base_human_xml(spec: HumanSpec) -> Path:
    """Generise osnovni human XML sa antropometrijskim parametrima (cached)."""
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
    """Napravi human XML sa aktuatorima, senzorima i izabranim terenom (cached)."""
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
    tune_passive_upper_body_joints(root)
    unlock_trunk_joints(root)
    tune_leg_joints(root)
    remove_trunk_equality_locks(root)
    set_training_collision_filters(root)
    add_stable_foot_contacts(root)
    add_terrain(root, env_version)
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


def tune_leg_joints(root: ET.Element) -> None:
    """Ogranici noge na humanoidniji prostor bez zakljucavanja stride osa."""
    worldbody = root.find("worldbody")
    for joint in worldbody.iter("joint"):
        spec = LEG_JOINT_SPECS.get(joint.get("name"))
        if spec is None:
            continue
        joint.set("limited", "true")
        for key, value in spec.items():
            joint.set(key, value)


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
    """Koristi originalnu generated foot capsule geometriju za kontakt."""
    worldbody = root.find("worldbody")
    for body in worldbody.iter("body"):
        if body.get("name") not in FOOT_BODY_NAMES:
            continue

        foot_geoms = body.findall("geom")
        if not foot_geoms:
            raise ValueError(f"{body.get('name')} nema foot geom za kontakt.")

        sole_name = f"{body.get('name')}_sole"
        contact_geom = foot_geoms[0]
        for geom in foot_geoms:
            mark_visual_only_geom(geom)
            if geom.get("name") == sole_name:
                contact_geom = geom

        contact_geom.set("name", sole_name)
        mark_foot_geom(contact_geom)
        for key, value in SOLE_CONTACT_GEOM_ATTRIBUTES.items():
            contact_geom.set(key, value)


def add_terrain(root: ET.Element, env_version: str) -> None:
    """Doda standard flat ili hardcore rough terrain u worldbody."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", **GROUND_TEXTURE_ATTRIBUTES)
    ET.SubElement(asset, "material", **GROUND_MATERIAL_ATTRIBUTES)

    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "geom", **FLOOR_GEOM_ATTRIBUTES)
    if env_version == "hardcore":
        add_rough_blocks(worldbody)


def add_rough_blocks(worldbody: ET.Element) -> None:
    """Doda deterministicke niske prepreke za hardcore varijantu."""
    for index in range(ROUGH_BLOCK_COUNT):
        x_pos = ROUGH_BLOCK_START_X + ROUGH_BLOCK_X_STEP * index
        y_pos = ROUGH_BLOCK_Y_ODD if index % 2 else ROUGH_BLOCK_Y_EVEN
        height = (
            ROUGH_BLOCK_BASE_HEIGHT
            + ROUGH_BLOCK_HEIGHT_STEP * (index % ROUGH_BLOCK_HEIGHT_PERIOD)
        )
        ET.SubElement(
            worldbody,
            "geom",
            name=f"rough_block_{index}",
            pos=f"{x_pos} {y_pos} {height}",
            size=f"{ROUGH_BLOCK_SIZE_X} {ROUGH_BLOCK_SIZE_Y} {height}",
            **ROUGH_BLOCK_GEOM_ATTRIBUTES,
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
