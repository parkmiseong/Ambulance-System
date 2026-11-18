"""Cross-engine resume preflight must match manifests and protect old files."""
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dqn_agent import DQNAmbulanceAgent
from evaluation_audit import file_sha256
from experiment_config import LearningConfig, SimulationConfig
import prepare_resume_copy as preparation
import simulation_setup
import test_simulation_system as fixtures


class ResumeCopyTests(unittest.TestCase):
    def test_engine_version_supports_both_installed_and_new_release_banner_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "sumo.exe"
            binary.write_bytes(b"fixture engine")
            for line, version in (("Eclipse SUMO sumo Version 1.27.1\n", "1.27.1"),
                                  ("Eclipse SUMO sumo 1.28.0\n", "1.28.0")):
                with self.subTest(line=line), \
                     patch.object(preparation.sumolib, "checkBinary", return_value=str(binary)), \
                     patch.object(preparation.subprocess, "run") as process:
                    process.return_value.stdout = line
                    self.assertEqual(preparation.current_engine()["version"], version)

    def test_engine_guard_rejects_the_old_engine_and_accepts_verified_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = dict(path="portable/sumo.exe", sha256="verified128", version="1.28.0")
            (root / "engine_transition.json").write_text(json.dumps(dict(resumed_engine=expected)), encoding="utf-8")
            with patch.object(preparation, "current_engine", return_value=dict(
                    path="installed/sumo.exe", sha256="old127", version="1.27.1")):
                with self.assertRaisesRegex(ValueError, "Set SUMO_BINARY"):
                    preparation.validate_recorded_engine(root)
            with patch.object(preparation, "current_engine", return_value=expected):
                self.assertEqual(preparation.validate_recorded_engine(root), expected)

    def test_legacy_experiment_has_no_implicit_engine_change_or_version_query(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(preparation, "current_engine") as engine:
            self.assertIsNone(preparation.validate_recorded_engine(directory))
            engine.assert_not_called()

    def fixture(self, directory):
        root = Path(directory)
        source = root / "source"
        config = SimulationConfig(arrival_steps=3600, demand_scale=1. / 72.,
            admission_mode="arrival", congestion_reward_mode="balanced",
            emergency_yield=True, repeat_background_traffic=True)
        hospitals = [fixtures.hospital()]
        with patch.object(simulation_setup, "network_digest", return_value="abc"):
            manifest = simulation_setup.make_manifest(None, hospitals, ["e"], [["h"]], config,
                10, "severity", 300., (), learning_config=LearningConfig())
        manifest.update(validation_seeds=[30001, 30002, 30003],
            validation_scenarios=list(range(1, 10)), validation_demands=[config.demand_scale],
            cycle_curriculum=False)
        manifest["signature"] = simulation_setup.manifest_signature(manifest)
        tag = "v6_" + manifest["signature"]
        mp = source / "training" / f"experiment_{tag}.json"
        mp.parent.mkdir(parents=True)
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        agent = DQNAmbulanceAgent(27, 1, architecture="hospital_shared_v6", learning_config=LearningConfig())
        agent.metadata = dict(manifest_path=str(mp), experiment_signature=manifest["signature"],
                              best_validation_reward=None)
        agent.training_updates = 292
        checkpoint = source / "training" / f"checkpoint_{tag}.pth"
        agent.save_model(str(checkpoint), episode=24, full_state=True)
        last = source / "models" / f"dqn_{tag}_last.pth"
        agent.save_model(str(last), episode=24)
        (source / "experiment_history.csv").write_text("source history", encoding="utf-8")
        binary = root / "new_sumo.exe"
        binary.write_bytes(b"new engine fixture")
        engine = dict(path=str(binary), sha256=file_sha256(binary), version="1.28.0",
                      version_line="Eclipse SUMO sumo Version 1.28.0")
        report = root / "old_engine.json"
        report.write_text(json.dumps(dict(installed_binary=dict(version="1.27.1", sha256="old"))), encoding="utf-8")
        return root, source, checkpoint, mp, last, hospitals, engine, report

    @contextmanager
    def preflight(self, hospitals, engine):
        with patch.object(preparation, "current_engine", return_value=engine), \
             patch.object(preparation, "prepare_environment", return_value=(None, hospitals, ["e"], [["h"]])) as setup, \
             patch.object(simulation_setup, "network_digest", return_value="abc"):
            yield setup

    def test_identical_preflight_copies_full_state_without_creating_best_or_copying_history(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root, source, cp, mp, last, hospitals, engine, old = self.fixture(temp)
            originals = {p: p.read_bytes() for p in (cp, mp, last, source / "experiment_history.csv")}
            target = root / "new_engine"
            with self.preflight(hospitals, engine) as setup:
                result = preparation.prepare_resume_copy(cp, target, root / "preflight", old)
            self.assertTrue(result["manifests_identical"])
            self.assertEqual(result["resume_start_episode"], 25)
            self.assertEqual(Path(result["checkpoint"]).read_bytes(), originals[cp])
            self.assertEqual(Path(result["manifest"]).read_bytes(), originals[mp])
            self.assertEqual(next((target / "models").glob("*_last.pth")).read_bytes(), originals[last])
            self.assertEqual(list(target.rglob("*_best.pth")), [])
            self.assertFalse((target / "experiment_history.csv").exists())
            transition = json.loads((target / "engine_transition.json").read_text(encoding="utf-8"))
            self.assertEqual(transition["source_completed_episode"], 24)
            self.assertEqual(transition["source_engine"]["version"], "1.27.1")
            self.assertEqual(transition["resumed_engine"]["version"], "1.28.0")
            self.assertEqual(transition["training_provenance"], "mixed_engine_continuation")
            self.assertTrue(setup.call_args.args[0].emergency_yield)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)

    def test_changed_hospital_data_rejects_copy_and_reports_exact_fields(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root, source, cp, mp, _, hospitals, engine, old = self.fixture(temp)
            changed = copy.deepcopy(hospitals)
            changed[0]["sumo_x"] += 1.
            target = root / "new_engine"
            before = cp.read_bytes()
            with self.preflight(changed, engine), self.assertRaisesRegex(ValueError, "differs"):
                preparation.prepare_resume_copy(cp, target, root / "preflight", old)
            self.assertFalse(target.exists())
            report = json.loads(next((root / "preflight").glob("*/preflight.json")).read_text(encoding="utf-8"))
            self.assertIn("hospitals", report["differing_fields"])
            self.assertFalse(report["manifests_identical"])
            self.assertEqual(cp.read_bytes(), before)

    def test_existing_destination_is_never_overwritten_or_prepared(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root, source, cp, _, _, hospitals, engine, old = self.fixture(temp)
            target = root / "existing"
            target.mkdir()
            marker = target / "user.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.preflight(hospitals, engine) as setup, self.assertRaises(FileExistsError):
                preparation.prepare_resume_copy(cp, target, root / "preflight", old)
            setup.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_source_changed_during_preflight_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root, source, cp, _, _, hospitals, engine, old = self.fixture(temp)
            def changed(*args, **kwargs):
                cp.write_bytes(b"externally replaced source checkpoint")
                return None, hospitals, ["e"], [["h"]]
            with self.preflight(hospitals, engine) as setup:
                setup.side_effect = changed
                with self.assertRaisesRegex(ValueError, "source experiment changed"):
                    preparation.prepare_resume_copy(cp, root / "new_engine", root / "preflight", old)
            self.assertFalse((root / "new_engine").exists())

    def test_weights_only_or_inherited_best_score_require_explicit_handling(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            weights = root / "weights.pth"
            torch.save(dict(model_state_dict={}), weights)
            with self.assertRaisesRegex(ValueError, "full training checkpoint"):
                preparation.prepare_resume_copy(weights, root / "target")
            root, source, cp, _, _, hospitals, engine, old = self.fixture(root / "second")
            data = torch.load(cp, weights_only=False)
            data["metadata"]["best_validation_reward"] = 1.
            torch.save(data, cp)
            with self.preflight(hospitals, engine) as setup, self.assertRaisesRegex(ValueError, "best score"):
                preparation.prepare_resume_copy(cp, root / "new_engine", root / "preflight", old)
            setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
