"""The verified engine launcher resumes to a cumulative target, preserving env."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from experiment_config import LearningConfig, SimulationConfig
from simulation_setup import manifest_signature
import resume_sumo128
import prepare_resume_copy
import train_agent


class ResumeEngineLauncherTests(unittest.TestCase):
    def fixture(self, directory, episode=24):
        root = Path(directory)
        binary = root / "runtime/sumo.exe"
        binary.parent.mkdir(); binary.write_bytes(b"portable test engine")
        config = SimulationConfig(arrival_steps=3600, demand_scale=1. / 72.,
            admission_mode="arrival", congestion_reward_mode="balanced",
            emergency_yield=True, repeat_background_traffic=True)
        manifest = dict(version=6, config=config.to_dict(), learning_config=LearningConfig().to_dict(),
            hospitals=[dict(id="h")], ambulance_count=10, queue_policy="severity", aging_seconds=300.,
            curriculum=[], validation_seeds=[30001, 30002, 30003],
            validation_scenarios=list(range(1, 10)), validation_demands=[config.demand_scale],
            cycle_curriculum=False)
        manifest["signature"] = manifest_signature(manifest)
        tag = "v6_" + manifest["signature"]
        training = root / "training"; training.mkdir()
        (training / f"experiment_{tag}.json").write_text(json.dumps(manifest), encoding="utf-8")
        checkpoint = training / f"checkpoint_{tag}.pth"
        torch.save(dict(episode=episode, metadata=dict(experiment_signature=manifest["signature"])), checkpoint)
        (root / "engine_transition.json").write_text(json.dumps(dict(
            experiment_signature=manifest["signature"], resumed_engine=dict(path=str(binary), sha256="test", version="1.28.0"))),
            encoding="utf-8")
        return binary, checkpoint, manifest

    def test_default_target_45_calculates_21_and_uses_saved_options(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            binary, _, manifest = self.fixture(directory)
            with patch.dict(os.environ, {"SUMO_BINARY": "original.exe"}), \
                 patch.object(prepare_resume_copy, "validate_recorded_engine", return_value=dict(version="1.28.0")), \
                 patch.object(train_agent, "train", side_effect=lambda **kw: os.environ["SUMO_BINARY"]) as train:
                selected = resume_sumo128.resume_to_target(directory)
                self.assertEqual(selected, str(binary))
                self.assertEqual(os.environ["SUMO_BINARY"], "original.exe")
            options = train.call_args.kwargs
            self.assertEqual(options["num_episodes"], 21)
            self.assertTrue(options["resume"])
            self.assertEqual(options["eval_every"], 45)
            self.assertEqual(options["config"].to_dict(), manifest["config"])
            self.assertEqual(options["validation_scenarios"], tuple(range(1, 10)))
            self.assertEqual(options["virtual_count"], 0)

    def test_finished_target_starts_no_more_training(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            _, checkpoint, _ = self.fixture(directory, episode=45)
            with patch.object(train_agent, "train") as train:
                self.assertEqual(resume_sumo128.resume_to_target(directory), checkpoint)
            train.assert_not_called()

    def test_failed_training_restores_process_engine_environment(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            self.fixture(directory)
            with patch.dict(os.environ, {"SUMO_BINARY": "original.exe"}), \
                 patch.object(prepare_resume_copy, "validate_recorded_engine", return_value=dict(version="1.28.0")), \
                 patch.object(train_agent, "train", side_effect=RuntimeError("stop")):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    resume_sumo128.resume_to_target(directory)
                self.assertEqual(os.environ["SUMO_BINARY"], "original.exe")


if __name__ == "__main__":
    unittest.main()
