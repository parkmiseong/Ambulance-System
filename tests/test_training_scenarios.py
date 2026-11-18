"""Restricted training scenarios, seed diversity, manifests and safe resume."""
from contextlib import contextmanager, redirect_stdout
import io
import json
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from experiment_config import SimulationConfig, LearningConfig
from simulation_setup import manifest_signature
from test_simulation_system import hospital
import train_agent


class TrainingScenarioTests(unittest.TestCase):
    @contextmanager
    def runners(self):
        result = dict(generated_patients=2, missions=2, arrived=2, failed=0,
                      censored=0, golden_rate=50., avg_wait=0., reward=4.,
                      avg_reward=2., avg_loss=0.)
        with patch.object(train_agent, "prepare_environment",
                          return_value=(None, [hospital()], ["e"], [["h"]])), \
             patch.object(train_agent, "make_manifest", return_value={"version": 5}), \
             patch.object(train_agent, "run_episode", return_value=result) as episode, \
             patch.object(train_agent, "append_record") as record, \
             patch.object(train_agent, "plot_training_metrics"), \
             redirect_stdout(io.StringIO()):
            yield episode, record

    def manifest(self, directory):
        path = next((Path(directory) / "training").glob("experiment_*.json"))
        return json.loads(path.read_text(encoding="utf-8"))

    def test_fixed_scenario_varies_seeds_and_keeps_nine_episode_curriculum(self):
        with tempfile.TemporaryDirectory() as directory, self.runners() as (episode, record):
            train_agent.train(num_episodes=20, eval_every=0, quiet=True,
                              training_scenarios=(6,), curriculum=(.1, .3),
                              cycle_curriculum=True, output_root=directory)
            calls = episode.call_args_list
            self.assertEqual([c.args[2]["id"] for c in calls], [6] * 20)
            self.assertEqual([c.args[3] for c in calls], list(range(10043, 10063)))
            self.assertEqual([c.args[5].demand_scale for c in calls],
                             [.1] * 9 + [.3] * 9 + [.1] * 2)
            manifest = self.manifest(directory)
            self.assertEqual(manifest["training_scenarios"], [6])
            self.assertEqual(manifest_signature(manifest), manifest["signature"])
            start = next(c for c in record.call_args_list if c.args[2] == "run_start")
            self.assertEqual(start.kwargs["metrics"]["training_scenarios"], [6])

    def test_default_preserves_scenario_cycle_and_existing_manifest_signature(self):
        with tempfile.TemporaryDirectory() as directory, self.runners() as (episode, _):
            train_agent.train(num_episodes=10, eval_every=0, quiet=True, output_root=directory)
            self.assertEqual([c.args[2]["id"] for c in episode.call_args_list],
                             list(range(1, 10)) + [1])
            expected = dict(version=5, validation_seeds=[30001, 30002, 30003],
                            validation_scenarios=[1], validation_demands=[1.],
                            cycle_curriculum=False, learning_config=LearningConfig().to_dict())
            expected["signature"] = manifest_signature(expected)
            self.assertEqual(self.manifest(directory), expected)

    def test_selected_scenarios_follow_requested_order(self):
        with tempfile.TemporaryDirectory() as directory, self.runners() as (episode, _):
            train_agent.train(num_episodes=3, eval_every=0, quiet=True,
                              training_scenarios=(6, 2), output_root=directory)
            self.assertEqual([c.args[2]["id"] for c in episode.call_args_list], [6, 2, 6])
            self.assertEqual(self.manifest(directory)["training_scenarios"], [6, 2])

    def test_invalid_scenarios_fail_before_environment_setup(self):
        for selected in ((), (0,), (10,), (6.5,), ("6",), (True,)):
            with self.subTest(selected=selected), \
                 patch.object(train_agent, "prepare_environment") as prepare:
                with self.assertRaisesRegex(ValueError, "Training scenario IDs"):
                    train_agent.train(training_scenarios=selected)
                prepare.assert_not_called()

    def test_resume_keeps_scenario_seed_and_curriculum_position(self):
        with tempfile.TemporaryDirectory() as directory, self.runners() as (episode, _):
            kwargs = dict(eval_every=0, quiet=True, training_scenarios=(6,),
                          curriculum=(.1, .3), cycle_curriculum=True, output_root=directory)
            train_agent.train(num_episodes=8, **kwargs)
            original_manifest = self.manifest(directory)
            episode.reset_mock()
            train_agent.train(num_episodes=3, resume=True, **kwargs)
            self.assertEqual([c.args[2]["id"] for c in episode.call_args_list], [6] * 3)
            self.assertEqual([c.args[3] for c in episode.call_args_list], [10051, 10052, 10053])
            self.assertEqual([c.args[5].demand_scale for c in episode.call_args_list],
                             [.1, .3, .3])
            self.assertEqual(self.manifest(directory), original_manifest)

    def test_other_scenario_selection_cannot_resume_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, self.runners() as (episode, _):
            train_agent.train(num_episodes=1, eval_every=0, quiet=True,
                              training_scenarios=(6,), output_root=directory)
            checkpoint = next((Path(directory) / "training").glob("checkpoint_*.pth"))
            original = checkpoint.read_bytes()
            episode.reset_mock()
            with self.assertRaisesRegex(FileNotFoundError, "No compatible checkpoint"):
                train_agent.train(num_episodes=1, eval_every=0, quiet=True, resume=True,
                                  training_scenarios=(1,), output_root=directory)
            episode.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_copied_experiment_resume_records_its_new_manifest_without_changing_source(self):
        import torch
        with tempfile.TemporaryDirectory() as directory, self.runners():
            source = Path(directory) / "source"
            copied = Path(directory) / "copied"
            options = dict(num_episodes=1, eval_every=0, quiet=True, training_scenarios=(6,))
            train_agent.train(output_root=source, **options)
            source_checkpoint = next((source / "training").glob("checkpoint_*.pth"))
            original = source_checkpoint.read_bytes()
            shutil.copytree(source, copied)
            train_agent.train(output_root=copied, resume=True, **options)
            checkpoint = next((copied / "training").glob("checkpoint_*.pth"))
            manifest = next((copied / "training").glob("experiment_*.json"))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(Path(payload["metadata"]["manifest_path"]), manifest.resolve())
            self.assertEqual(payload["episode"], 2)
            self.assertEqual(source_checkpoint.read_bytes(), original)

    def test_cli_passes_training_selection_separately_from_validation(self):
        argv = ["train_agent.py", "--episodes", "54", "--train-scenarios", "6",
                "--validation-scenarios", "6"]
        with patch.object(sys, "argv", argv), patch.object(train_agent, "train") as train:
            train_agent.main()
        self.assertEqual(train.call_args.args[0], 54)
        self.assertEqual(train.call_args.kwargs["training_scenarios"], [6])
        self.assertEqual(train.call_args.args[13], [6])


if __name__ == "__main__":
    unittest.main()
