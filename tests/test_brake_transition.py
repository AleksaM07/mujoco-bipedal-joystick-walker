import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate
import train
from config import EnvConfig
from skills.brake_transition import newline_compatible_sha256


class BrakeTransitionTests(unittest.TestCase):
    def test_xml_hash_accepts_lf_and_crlf_checkout_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "model.xml"
            xml_path.write_bytes(b"<mujoco>\n  <worldbody/>\n</mujoco>\n")
            crlf = xml_path.read_bytes().replace(b"\n", b"\r\n")
            crlf_hash = hashlib.sha256(crlf).hexdigest()

            hashes = newline_compatible_sha256(xml_path)

        self.assertIn(crlf_hash, hashes)

    def test_training_factory_selects_specialized_bridge(self) -> None:
        config = EnvConfig(
            locomotion_skill="brake_transition",
            transition_min_walk_seconds=0.0,
        )
        with patch("skills.brake_transition.BrakeTransitionEnv") as constructor:
            train.make_environment(config)

        constructor.assert_called_once()
        forwarded = constructor.call_args.kwargs["config_overrides"]
        self.assertEqual(forwarded["brake_lead_seconds_min"], 0.2)
        self.assertEqual(forwarded["brake_lead_seconds_max"], 0.8)
        self.assertIsNone(forwarded["brake_state_bank_file"])

    def test_evaluation_factory_selects_same_bridge(self) -> None:
        config = EnvConfig(locomotion_skill="brake_transition")
        with patch("skills.brake_transition.BrakeTransitionEnv") as constructor:
            evaluate.make_environment(config)

        constructor.assert_called_once()
        self.assertNotIn(
            "locomotion_skill", constructor.call_args.kwargs["config_overrides"]
        )

    def test_invalid_lead_interval_is_rejected_by_config_contract(self) -> None:
        config = EnvConfig(
            locomotion_skill="brake_transition",
            brake_lead_seconds_min=0.8,
            brake_lead_seconds_max=0.2,
        )
        self.assertGreater(
            config.brake_lead_seconds_min, config.brake_lead_seconds_max
        )


if __name__ == "__main__":
    unittest.main()
