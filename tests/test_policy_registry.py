import unittest

from skills.policy_registry import (
    EXPECTED_XML_SHA256,
    FINAL_STANDING,
    FINAL_WALKING,
    validate_final_policy_pair,
)


class FinalPolicyRegistryTests(unittest.TestCase):
    def test_labels_are_explicit_and_archives_exist(self) -> None:
        self.assertEqual(FINAL_WALKING.label, "FINAL_WALKING")
        self.assertEqual(FINAL_STANDING.label, "FINAL_STANDING")
        self.assertTrue(FINAL_WALKING.checkpoint.is_dir())
        self.assertTrue(FINAL_STANDING.checkpoint.is_dir())

    def test_frozen_policies_share_the_physical_interface(self) -> None:
        reports = validate_final_policy_pair()

        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report["action_size"] == 26 for report in reports))
        self.assertTrue(
            all(report["observation_size"] == 366 for report in reports)
        )
        self.assertTrue(
            all(report["xml_sha256"] == EXPECTED_XML_SHA256 for report in reports)
        )


if __name__ == "__main__":
    unittest.main()
