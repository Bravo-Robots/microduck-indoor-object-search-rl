"""Full sensor-to-actuator search contract in a controlled, unobstructed aisle."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")

from duck_search import physical  # noqa: E402
from duck_search.scene import ROOT, scenario  # noqa: E402


class PhysicalSearchContract(unittest.TestCase):
    def test_visible_goal_is_approached_and_confirmed_using_real_motors(self):
        # Fixed geometry isolates the integration contract from the performance
        # benchmark. The actor still receives rendered RGB/depth, never truth.
        scene = scenario(0)
        scene["spawn"] = [-1.95, -1.65, 0.0]
        scene["targets"][0] = [-0.95, -1.65]
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                seed=0,
                goal=0,
                seconds=30,
                no_video=True,
                viewer=False,
                checkpoint=ROOT / "outputs/release/best.pt",
                controller="rl",
                output=Path(temp) / "physical-search.json",
                quiet=True,
            )
            with patch.object(physical, "scenario", return_value=scene):
                report = physical.rehearse(args)
        self.assertTrue(report["success"], report)
        self.assertFalse(report["fall"])
        self.assertEqual(report["false_confirmations"], 0)
        self.assertEqual(report["contact_steps"], 0)
        self.assertGreater(report["detector_visible_observations"], 1)
        start = report["trace"][0]["xy"]
        end = report["trace"][-1]["xy"]
        self.assertGreater(end[0] - start[0], 0.15)
        self.assertGreater(report["trace"][-1]["evaluator_target_pixels"], 5)


if __name__ == "__main__":
    unittest.main()
