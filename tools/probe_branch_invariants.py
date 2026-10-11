"""Check historical measurement formulas against native MuJoCo kinematics.

This uses deterministic synthetic velocities, not a policy rollout. Only selected
pure methods are loaded from Git; no training environment is imported or changed.
"""

import __future__

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs/branch_diagnosis_20261011"


def load_methods(branch: str, names: set[str]) -> dict[str, Callable]:
    """Extract the named measurement methods without importing environment code."""
    source = subprocess.check_output(
        ["git", "show", f"{branch}:biomechanics_env.py"],
        cwd=PROJECT_ROOT,
        encoding="utf-8",
    )
    tree = ast.parse(source)
    methods = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            namespace: dict[str, Any] = {"jp": np, "np": np}
            module = ast.Module(body=[node], type_ignores=[])
            code = compile(
                module,
                f"{branch}:biomechanics_env.py",
                "exec",
                flags=__future__.annotations.compiler_flag,
            )
            exec(code, namespace)
            methods[node.name] = namespace[node.name]
    missing = names.difference(methods)
    if missing:
        raise ValueError(f"Missing measurement methods on {branch}: {sorted(missing)}")
    return methods


def measure_model() -> dict[str, Any]:
    """Compare historical formulas to Jacobians and position finite differences."""
    xml_path = PROJECT_ROOT / (
        "generated_models/human_male_180cm_75kg_standard_trainfast_v21_noarms.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("a-pose").id)
    data.qpos[2] += 1.0
    data.qvel[:] = np.random.default_rng(7).normal(0.0, 0.4, model.nv)
    data.qvel[:3] = [0.3, -0.1, 0.05]
    data.qvel[3:6] = [0.4, 0.7, -0.2]
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    torso_id = model.body("thorax").id
    pelvis_id = model.body("pelvis").id
    body_ids = np.arange(1, model.nbody)
    mass = model.body_mass[body_ids]
    weights = mass / mass.sum()
    torso_rotation = data.xmat[torso_id].reshape(3, 3)
    state = SimpleNamespace(
        _reference_anchor_body_id=pelvis_id,
        _deepmimic_body_ids=body_ids,
        _deepmimic_body_weights=weights,
        _body_xmat=lambda frame: frame.xmat[torso_id].reshape(3, 3),
    )
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacobian_position, jacobian_rotation, pelvis_id)
    correct_pelvis_velocity = jacobian_position @ data.qvel
    mujoco.mj_jacBody(model, data, jacobian_position, jacobian_rotation, torso_id)
    correct_gyro = torso_rotation.T @ (jacobian_rotation @ data.qvel)
    correct_com = np.sum(data.xipos[body_ids] * weights[:, None], axis=0)
    correct_com_velocity = data.subtree_linvel[torso_id].copy()

    derivative_samples = []
    timestep = 1e-6
    for direction in (-1.0, 1.0):
        offset_data = mujoco.MjData(model)
        offset_data.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, offset_data.qpos, data.qvel, direction * timestep)
        mujoco.mj_forward(model, offset_data)
        derivative_samples.append(
            (
                offset_data.xpos[pelvis_id].copy(),
                np.sum(offset_data.xipos[body_ids] * weights[:, None], axis=0),
            )
        )
    finite_difference_pelvis = (derivative_samples[1][0] - derivative_samples[0][0]) / (
        2.0 * timestep
    )
    finite_difference_com = (derivative_samples[1][1] - derivative_samples[0][1]) / (
        2.0 * timestep
    )
    np.testing.assert_allclose(
        correct_pelvis_velocity, finite_difference_pelvis, atol=1e-7
    )
    np.testing.assert_allclose(correct_com_velocity, finite_difference_com, atol=1e-7)
    expected = {
        "_local_root_angvel": correct_gyro,
        "_reference_anchor_linvel": correct_pelvis_velocity,
        "_center_of_mass": correct_com,
        "_center_of_mass_velocity": correct_com_velocity,
    }
    measured = {}
    for branch in ("project_v2_refrences_only", "experiment/v21-stable-baseline"):
        methods = load_methods(branch, set(expected))
        measured[branch] = {}
        for name, reference in expected.items():
            result = np.asarray(methods[name](state, data))
            measured[branch][name] = {
                "branch_result": result.tolist(),
                "native_reference": reference.tolist(),
                "l2_error": float(np.linalg.norm(result - reference)),
            }
    main_gyro = load_methods("main", {"_local_root_angvel"})["_local_root_angvel"]
    measured["main"] = {
        "_local_root_angvel": {
            "branch_result": np.asarray(main_gyro(state, data)).tolist(),
            "native_reference": correct_gyro.tolist(),
            "l2_error": float(np.linalg.norm(main_gyro(state, data) - correct_gyro)),
        }
    }
    return {
        "mujoco_version": mujoco.__version__,
        "xml": xml_path.relative_to(PROJECT_ROOT).as_posix(),
        "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "state_description": (
            "a-pose raised 1m; seed7 joint velocities std0.4 rad/s; "
            "explicit root velocity"
        ),
        "root_qvel": data.qvel[:6].tolist(),
        "native_pelvis_jacobian_fd_error": float(
            np.linalg.norm(correct_pelvis_velocity - finite_difference_pelvis)
        ),
        "native_com_subtree_fd_error": float(
            np.linalg.norm(correct_com_velocity - finite_difference_com)
        ),
        "branch_measurements": measured,
    }


def main() -> None:
    """Write reproducible measurement counterexamples and reward-scale examples."""
    result = measure_model()
    result["reward_examples"] = [
        {
            "description": (
                f"{count} of 18 independent hinges each off by {angle} degrees"
            ),
            "project_mean_pose_reward": float(
                np.exp(-0.25 * count * np.deg2rad(angle) ** 2 / 18)
            ),
            "same_hinges_unweighted_sum_reward": float(
                np.exp(-0.25 * count * np.deg2rad(angle) ** 2)
            ),
        }
        for count, angle in ((1, 20), (18, 20), (18, 45))
    ]
    result["limitations"] = [
        "Synthetic instantaneous state is a formula counterexample, not a gait test.",
        "Same v21 model isolates measurement formulas; no checkpoint was evaluated.",
        "MimicKit uses weighted quaternion-angle sums, not unweighted hinge sums.",
        "The gyro issue also exists on main; it alone cannot explain regression.",
        "These checks do not measure causal contributions to training failure.",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "measurement_probes.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
