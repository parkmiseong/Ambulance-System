"""Held-out panels, matched patients, and seed-cluster confidence intervals."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"CODE"))
from evaluation_metrics import paired_patient_comparison, episode_uncertainty, validate_test_seeds
from compare_evaluations import compare_runs
from evaluation_audit import build_evaluation_environment
from experiment_config import SimulationConfig, SCENARIOS
from reward import REWARD_COMPONENT_NAMES
import run_scenarios
import test_best_model_selection as best_fixtures

def patient(seed,number,reward=1.):
    parts=dict.fromkeys(REWARD_COMPONENT_NAMES,0.); parts["golden"]=reward
    return dict(patient_id=f"p{number}",seed=seed,demand_scale=.1,scenario_id=1,ktas=3,
                occurrence_time=float(number),edge_id="e",reward=reward,reward_components=parts,
                arrived=True,golden_success=True,wait_seconds=2.)

class EvaluationMetricTests(unittest.TestCase):
    def test_aggregation_does_not_mix_congestion_reward_profiles(self):
        with self.assertRaisesRegex(ValueError, "congestion"):
            run_scenarios.aggregate_results([dict(reward_version=4, congestion_reward_mode="penalty"),
                                             dict(reward_version=4, congestion_reward_mode="balanced")])
    def test_aggregation_does_not_mix_admission_handling_profiles(self):
        with self.assertRaisesRegex(ValueError,"admission"):
            run_scenarios.aggregate_results([dict(reward_version=4,hospital_admission_mode="precheck"),
                                             dict(reward_version=4,hospital_admission_mode="request")])
    def test_patient_weighted_delta_and_component_sum(self):
        a=[patient(1,1)]+[patient(2,n) for n in range(1,4)]
        b=[patient(r["seed"],int(r["patient_id"][1:]),3.) for r in a]
        result=paired_patient_comparison(a,b)
        self.assertEqual((result["patient_pairs"],result["seed_pairs"]),(4,2))
        self.assertEqual(result["delta"]["reward_per_patient"],2.)
        self.assertEqual(result["ci95"]["reward_per_patient"],[2.,2.])
        self.assertAlmostEqual(sum(result["delta"]["component_"+k] for k in REWARD_COMPONENT_NAMES),2.)
    def test_bootstrap_uses_whole_seed_episodes(self):
        a=[patient(1,1,0.)]+[patient(2,n,0.) for n in range(100)]
        b=[patient(r["seed"],int(r["patient_id"][1:]),0. if r["seed"]==1 else 10.) for r in a]
        result=paired_patient_comparison(a,b)
        self.assertAlmostEqual(result["delta"]["reward_per_patient"],1000/101)
        self.assertEqual(result["ci95"]["reward_per_patient"],[0.,10.])
    def test_one_seed_has_no_claimed_interval(self):
        result=paired_patient_comparison([patient(1,1)],[patient(1,1,2.)])
        self.assertIsNone(result["ci95"])
        self.assertFalse(result["includes_training_seed_uncertainty"])
    def test_mismatched_patient_panels_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"identical complete"):
            paired_patient_comparison([patient(1,1)],[patient(1,2)])
        for field,value in (("ktas",5),("edge_id","other"),("occurrence_time",9.)):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):
                paired_patient_comparison([patient(1,1)],[dict(patient(1,1),**{field:value})])
    def test_missing_components_are_unavailable_not_fabricated(self):
        row=patient(1,1); row.pop("reward_components")
        result=paired_patient_comparison([row],[dict(row,reward=2.)])
        self.assertFalse(result["reward_components_available"])
        self.assertNotIn("component_golden",result["delta"])
    def test_test_seeds_exclude_training_validation_and_duplicates(self):
        manifest=dict(config=dict(seed=42),validation_seeds=[30001])
        validate_test_seeds([32001,32002],manifest,54)
        for seeds in ([10050],[30001],[32001,32001]):
            with self.subTest(seeds=seeds),self.assertRaises(ValueError):
                validate_test_seeds(seeds,manifest,54)
        validate_test_seeds([30001],manifest,54,diagnostic=True)
    def test_uncertainty_preserves_global_rng(self):
        rows=[dict(seed=s,reward=2.,arrived=1,golden_success=1,avg_wait=0.,generated_patients=1) for s in (1,2)]
        before=np.random.get_state()
        self.assertEqual(episode_uncertainty(rows),episode_uncertainty(rows))
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1]); self.assertEqual(before[2:],after[2:])
    def test_duplicate_seeds_keep_descriptive_aggregation_without_claiming_ci(self):
        row=dict(seed=1,reward=2.,arrived=1,golden_success=1,avg_wait=0.,generated_patients=1)
        result=episode_uncertainty([row,row])
        self.assertIsNone(result["intervals"])
        self.assertEqual(result["unavailable_reason"],"duplicate_episode_seeds")
    def test_zero_patient_panel_has_no_invalid_interval(self):
        rows=[dict(seed=s,reward=0.,arrived=0,golden_success=0,avg_wait=0.,generated_patients=0) for s in (1,2)]
        self.assertIsNone(episode_uncertainty(rows)["intervals"])

class EvaluationLogTests(unittest.TestCase):
    def log(self,path,last=False,incomplete=False):
        fields=["run_id","record_type","model_path","metrics_json","status","strategy","scenario_id"]
        with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for run,value in (("a",1.),("b",3.)):
                model=run+("_last.pth" if last else "_best.pth")
                w.writerow(dict(run_id=run,record_type="run_start",model_path=model,
                    metrics_json=json.dumps(dict(model_kind="last" if last else "best",
                        model_episode=18 if run=="a" else 27,evaluation_split="test",
                        evaluation_environment=build_evaluation_environment(
                            None,[],SimulationConfig(),[SCENARIOS[0]],[1.],10,"severity",300,(.2,.6))))))
                for seed in (1,2):
                    w.writerow(dict(run_id=run,record_type="mission",model_path=model,strategy="dqn",
                        scenario_id=1,metrics_json=json.dumps(patient(seed,1,value))))
                if not(incomplete and run=="b"):
                    w.writerow(dict(run_id=run,record_type="run_end",status="completed",model_path=model,metrics_json="{}"))
    def test_completed_best_runs_preserve_recorded_model_epochs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"history.csv"; self.log(p)
            result=compare_runs(p,["a","b"])
        self.assertEqual(result["runs"]["b"]["model_episode"],27)
        self.assertEqual(result["comparisons"][0]["delta"]["reward_per_patient"],2.)
    def test_comparison_rejects_different_driving_rules(self):
        from emergency_yield import yield_settings
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"history.csv"; self.log(path)
            with path.open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
            for row in rows:
                if row["record_type"]=="run_start" and row["run_id"]=="b":
                    data=json.loads(row["metrics_json"])
                    data["emergency_yield"]=yield_settings(True)
                    row["metrics_json"]=json.dumps(data)
            with path.open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,"driving settings"):
                compare_runs(path,["a","b"])
    def test_comparison_reads_different_experiment_logs(self):
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/"history.csv"; self.log(original)
            with original.open(encoding="utf-8-sig",newline="") as f:
                reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
            paths=[]
            for run in ("a","b"):
                path=Path(d)/(run+".csv");paths.append(path)
                with path.open("w",encoding="utf-8-sig",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                    w.writerows(r for r in rows if r["run_id"]==run)
            result=compare_runs(paths,["a","b"])
            self.assertEqual(result["comparisons"][0]["delta"]["reward_per_patient"],2.)
    def test_last_and_incomplete_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"history.csv"; self.log(p,last=True)
            with self.assertRaisesRegex(ValueError,"best-model"):compare_runs(p,["a","b"])
            self.log(p,incomplete=True)
            with self.assertRaisesRegex(ValueError,"not completed"):compare_runs(p,["a","b"])
    def test_defaults_follow_single_training_scenario_and_demands(self):
        helper=best_fixtures.BestModelSelectionTests()
        with tempfile.TemporaryDirectory() as d:
            best=helper.model(d,"dqn_v6_current_best.pth")
            argv=["run_scenarios.py","--output-root",d,"--strategies","dqn","--seeds","32001"]
            with helper.cli(run_scenarios,d,argv) as (load,record,_):
                load.return_value[4].update(training_scenarios=[6],validation_demands=[.1,.3],validation_seeds=[30001])
                run_scenarios.main()
                calls=run_scenarios.run_single.call_args_list
                self.assertEqual([c.args[2]["id"] for c in calls],[6,6])
                self.assertEqual([c.kwargs["config"].demand_scale for c in calls],[.1,.3])
                start=next(c for c in record.call_args_list if c.args[2]=="run_start")
                self.assertEqual(start.kwargs["metrics"]["evaluation_split"],"test")
                self.assertEqual(start.kwargs["model_path"],str(best))
