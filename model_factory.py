import contextlib
import sys
import xml.etree.ElementTree as ET
import uuid
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import BIOMECH_DIR, LOCOMOTION_JOINTS, RUNS_DIR, EnvConfig


@dataclass
class HumanSpec:
    mass: float
    height: float
    sex: str


@contextlib.contextmanager
def _inside_biomech_dir():
    """Generator koristi relativne CSV putanje, pa ga kratko pokrecemo iz njegovog foldera."""
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    sys.path.insert(0, str(BIOMECH_DIR))
    try:
        import os

        os.chdir(BIOMECH_DIR)
        yield
    finally:
        import os

        os.chdir(old_cwd)
        sys.path[:] = old_path


class HumanModelFactory:
    """Pravi randomizovan MJCF model i dodaje aktuatore potrebne za RL kontrolu."""

    def __init__(self, config: EnvConfig):
        self.config = config
        # Drzimo generisane XML fajlove u workspace-u da radi i pod sandboxom.
        self.work_dir = RUNS_DIR / "tmp_models" / uuid.uuid4().hex[:8]
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        pass

    def sample_spec(self, rng: np.random.Generator) -> HumanSpec:
        sex = "male" if rng.random() < self.config.sex_probability_male else "female"
        return HumanSpec(
            mass=float(rng.uniform(self.config.min_mass, self.config.max_mass)),
            height=float(rng.uniform(self.config.min_height, self.config.max_height)),
            sex=sex,
        )

    def build_model(self, spec: HumanSpec) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
        raw_xml = self.work_dir / "human_raw.xml"
        rl_xml = self.work_dir / "human_rl.xml"

        with _inside_biomech_dir():
            from generate_human_model import generate_human_model

            generate_human_model(
                filename=str(raw_xml),
                mass=spec.mass,
                height=spec.height,
                sex=spec.sex,
            )

        self._add_actuators(raw_xml, rl_xml)
        model = mujoco.MjModel.from_xml_path(str(rl_xml))
        data = mujoco.MjData(model)
        return model, data, rl_xml

    def _add_actuators(self, source_xml: Path, output_xml: Path) -> None:
        """Dodaje motore na unapred izabrane zglobove i cuva novi XML."""
        tree = ET.parse(source_xml)
        root = tree.getroot()

        # Stabilniji numericki korak za RL. MuJoCo ce limite uglova sam prevesti u radijane.
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", "0.002")
        option.set("integrator", "implicit")
        option.set("iterations", "20")

        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")

        existing = {motor.get("joint") for motor in actuator.findall("motor")}
        ctrl = f"{-self.config.torque_limit} {self.config.torque_limit}"
        for joint_name in LOCOMOTION_JOINTS:
            if joint_name in existing:
                continue
            ET.SubElement(
                actuator,
                "motor",
                name=f"{joint_name}_motor",
                joint=joint_name,
                gear="1",
                ctrllimited="true",
                ctrlrange=ctrl,
            )

        ET.indent(tree, space="  ", level=0)
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
