"""A reference-free environment must step with the same metrics as reset."""

import unittest
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from biomechanics_env import BiomechanicsJoystickEnv


class NonReferenceStepTests(unittest.TestCase):
    def test_actual_physics_step_traces_with_complete_imitation_metrics(self):
        archive = Path(__file__).resolve().parents[1] / "prezentacija"
        xml_path = (
            archive
            / "5th_success_final_natural_walk/generated_models"
            / "human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
        )
        env = BiomechanicsJoystickEnv(
            config_overrides={
                "xml_path": str(xml_path),
                "impl": "jax",
                "physics_backend": "mjx_jax",
                "reference_gait": "none",
                "recovery_physics_version": "consistent_v6",
                "enable_erfi": False,
            }
        )
        # Trace the real reset/physics/reward pipeline, catching missing dict keys
        # without compiling a second full humanoid solver just for this branch.
        state = jax.eval_shape(env.reset, jax.random.PRNGKey(7))
        next_state = jax.eval_shape(env.step, state, jp.zeros(env.action_size))
        self.assertEqual(set(state.metrics), set(next_state.metrics))
        for name in (
            "deepmimic_arm",
            "deepmimic_arm_pose",
            "deepmimic_arm_velocity",
            "deepmimic_arm_pose_error",
            "deepmimic_arm_velocity_error",
        ):
            self.assertEqual(next_state.metrics[name].shape, ())
            self.assertTrue(np.issubdtype(next_state.metrics[name].dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
