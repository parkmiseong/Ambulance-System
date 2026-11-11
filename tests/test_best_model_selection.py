"""Best-only model selection and evaluation provenance, without running SUMO."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from experiment_config import SimulationConfig
from simulation_setup import latest_model, resolve_evaluation_model
import run_scenarios
import compare_queue_policies


class BestModelSelectionTests(unittest.TestCase):
    def model(self, root, name, timestamp=100):
        path = Path(root) / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.utime(path, (timestamp, timestamp))
        return path

    def test_latest_selects_best_even_when_last_is_newer(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model(directory, "dqn_v3_old_best.pth", 100)
            best = self.model(directory, "dqn_v5_current_best.pth", 200)
            self.model(directory, "dqn_v5_current_last.pth", 300)
            self.model(directory, "dqn_v4_other_last.pth", 400)
            fake = Path(directory) / "models" / "dqn_v5_directory_best.pth"
            fake.mkdir()
            self.assertEqual(latest_model(directory), best)

    def test_missing_best_never_falls_back_to_last(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model(directory, "dqn_v5_current_last.pth")
            with self.assertRaisesRegex(FileNotFoundError, "No best model"):
                latest_model(directory)

    def test_last_path_selects_its_best_even_if_last_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            best = self.model(directory, "dqn_v5_current_best.pth")
            requested = best.with_name("dqn_v5_current_last.pth")
            self.assertFalse(requested.exists())
            self.assertEqual(resolve_evaluation_model(requested), best)

    def test_missing_pair_does_not_choose_another_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model(directory, "dqn_v5_other_best.pth")
            requested = self.model(directory, "dqn_v5_target_last.pth")
            with self.assertRaisesRegex(FileNotFoundError, "Best model not found"):
                resolve_evaluation_model(requested, directory)

    def test_explicit_best_keeps_requested_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            older = self.model(directory, "dqn_v5_snapshot_best.pth", 100)
            self.model(directory, "dqn_v5_newer_best.pth", 200)
            self.assertEqual(resolve_evaluation_model(older, directory), older)

    def test_output_root_selects_only_that_experiments_best(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left"
            right = Path(directory) / "right"
            self.model(left, "dqn_v5_left_best.pth", 500)
            best = self.model(right, "dqn_v5_right_best.pth", 100)
            self.assertEqual(resolve_evaluation_model(output_root=right), best)

    def test_non_best_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.model(directory, "checkpoint_v5_current.pth")
            with self.assertRaisesRegex(ValueError, "requires a .*best"):
                resolve_evaluation_model(checkpoint)

    @contextmanager
    def cli(self, module, root, argv):
        agent = SimpleNamespace(loaded_episode=27)
        manifest = dict(signature="signature", ambulance_count=10,
                        queue_policy="severity", aging_seconds=300.)
        response = dict(generated_patients=2, arrived=2, failed=0, censored=0,
                        golden_success=1, golden_rate=50., avg_reward=2., reward=4.,
                        avg_wait=0., reward_version=4)
        output = io.StringIO()
        with patch.object(sys, "argv", argv), \
             patch.object(module, "load_experiment",
                          return_value=(None, [], agent, SimulationConfig(), manifest)) as load, \
             patch.object(module, "manifest_path_root", return_value=Path(root)), \
             patch.object(module, "run_single", return_value=response), \
             patch.object(module, "append_record") as record, \
             redirect_stdout(output):
            yield load, record, output

    def test_policy_cli_converts_last_and_records_best_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            best = self.model(directory, "dqn_v5_current_best.pth")
            last = best.with_name("dqn_v5_current_last.pth")
            argv = ["run_scenarios.py", "--model", str(last), "--strategies", "dqn",
                    "--seeds", "32001", "--demand-levels", "0.1"]
            with self.cli(run_scenarios, directory, argv) as (load, record, output):
                run_scenarios.main()
            load.assert_called_once_with(best)
            self.assertIn("episode=27 selection=best", output.getvalue())
            self.assertIn("using paired best", output.getvalue())
            start = next(c for c in record.call_args_list if c.args[2] == "run_start")
            self.assertEqual(start.kwargs["model_path"], str(best))
            self.assertEqual(start.kwargs["metrics"]["model_episode"], 27)
            self.assertEqual(start.kwargs["metrics"]["model_kind"], "best")
            self.assertEqual(start.kwargs["metrics"]["requested_model"], str(last))
            self.assertTrue(all(c.kwargs["model_path"] == str(best)
                                for c in record.call_args_list))

    def test_policy_cli_uses_best_from_custom_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            best = self.model(directory, "dqn_v5_current_best.pth")
            argv = ["run_scenarios.py", "--output-root", directory, "--strategies", "dqn",
                    "--seeds", "32001"]
            with self.cli(run_scenarios, directory, argv) as (load, _, _):
                run_scenarios.main()
            load.assert_called_once_with(best)

    def test_queue_cli_converts_last_and_records_best_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            best = self.model(directory, "dqn_v5_current_best.pth")
            last = best.with_name("dqn_v5_current_last.pth")
            argv = ["compare_queue_policies.py", "--model", str(last), "--seeds", "32001"]
            with self.cli(compare_queue_policies, directory, argv) as (load, record, output):
                compare_queue_policies.main()
            load.assert_called_once_with(best)
            self.assertIn("episode=27 selection=best", output.getvalue())
            start = next(c for c in record.call_args_list if c.args[2] == "run_start")
            self.assertEqual(start.kwargs["metrics"]["model_episode"], 27)
            self.assertEqual(start.kwargs["metrics"]["model_kind"], "best")
            self.assertTrue(all(c.kwargs["model_path"] == str(best)
                                for c in record.call_args_list))

    def test_cli_without_best_fails_before_loading_or_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model(directory, "dqn_v5_current_last.pth")
            for module in (run_scenarios, compare_queue_policies):
                argv = [module.__name__ + ".py", "--output-root", directory]
                with self.subTest(module=module.__name__), \
                     patch.object(sys, "argv", argv), \
                     patch.object(module, "load_experiment") as load, \
                     patch.object(module, "run_single") as run, \
                     redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.main()
                    self.assertEqual(error.exception.code, 2)
                    load.assert_not_called()
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
