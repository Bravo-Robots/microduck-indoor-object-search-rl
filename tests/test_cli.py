"""User-facing contracts, input errors and portable output handling."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from duck_search.__main__ import main, parser
from duck_search.env import SearchEnv
from duck_search.guard import guarded_action


class CliContracts(unittest.TestCase):
    def test_invalid_arguments_fail_before_work(self):
        for arguments in (
            ["train", "--envs", "0"],
            ["evaluate", "missing.pt", "--episodes", "-1"],
            ["rehearse", "--seconds", "nan"],
            ["rehearse", "--seconds", "0"],
            ["rehearse", "--seed", "-1"],
            ["rehearse", "--goal", "3"],
        ):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    parser().parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_missing_checkpoint_reports_a_readable_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                main(["evaluate", "/nonexistent/duck-search-checkpoint.pt"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("error:", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_evaluation_creates_nested_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nested" / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "evaluate",
                        "outputs/release/best.pt",
                        "--episodes",
                        "2",
                        "--output",
                        str(output),
                    ]
                )
            self.assertTrue(output.is_file())

    def test_reset_uses_distinct_object_and_spawn_positions(self):
        env = SearchEnv(64, seed=25)
        points = torch.cat([env.pos[:, None], env.targets], dim=1)
        distances = torch.cdist(points, points)
        distances += torch.eye(4)[None] * 100
        self.assertGreater(float(distances.min()), 0.249)

    def test_physical_visit_updates_memory_without_a_proxy_step(self):
        env = SearchEnv(1, seed=25)
        env.visits[:] = 0
        self.assertTrue(bool(env.record_visit()[0]))
        self.assertFalse(bool(env.record_visit()[0]))
        self.assertEqual(float(env.visits.sum()), 2)

    def test_guard_preserves_turns_confirmation_and_clear_corridor(self):
        for action in range(6):
            self.assertEqual(guarded_action(action, [2.5] * 12), action)
        for action in (1, 2, 5):
            self.assertEqual(guarded_action(action, [0.05] * 12), action)

    def test_guard_replaces_blocked_advances_with_a_turn(self):
        for action in (0, 3, 4):
            self.assertIn(guarded_action(action, [0.15] * 12), (1, 2))


if __name__ == "__main__":
    unittest.main()
