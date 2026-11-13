"""V6 schedules, resume, feature controls, and manifest compatibility."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from baseline import build_state_vector, state_dimension, HospitalRouters
from dqn_agent import DQNAmbulanceAgent, HospitalFeatureQNetwork
from experiment_config import LearningConfig, SimulationConfig
from reward import HOSPITAL_TYPES
import simulation_setup
import train_agent
import test_simulation_system as fixtures

torch.set_num_threads(1)

class V6LearningTests(unittest.TestCase):
    def agent(self, **options):
        settings = dict(batch_size=1, replay_capacity=64, target_sync_interval=2)
        settings.update(options)
        return DQNAmbulanceAgent(27, 1, architecture="hospital_shared_v6",
                                learning_config=LearningConfig(**settings))

    def step(self, agent):
        state = np.ones(27, dtype=np.float32)
        agent.store_transition(state, 0, 5., state, True)
        return agent.train_step()

    def test_hard_sync_counts_only_actual_updates(self):
        a = self.agent(batch_size=2)
        state = np.ones(27, dtype=np.float32)
        a.store_transition(state, 0, 5., state, True)
        self.assertIsNone(a.train_step())
        self.assertEqual(a.training_updates, 0)
        initial = [p.detach().clone() for p in a.target_model.parameters()]
        self.step(a)
        self.assertEqual((a.training_updates, a.target_updates), (1, 0))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(initial,a.target_model.parameters())))
        self.step(a)
        self.assertEqual((a.training_updates,a.target_updates),(2,1))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.model.parameters(),a.target_model.parameters())))

    def test_soft_sync_interpolates_each_update(self):
        a = self.agent(target_sync_mode="soft", target_tau=.5)
        old = [p.detach().clone() for p in a.target_model.parameters()]
        self.step(a)
        for x,online,target in zip(old,a.model.parameters(),a.target_model.parameters()):
            self.assertTrue(torch.allclose(target,.5*x+.5*online,atol=1e-7))
        self.assertEqual(a.target_updates,1)

    def test_episode_control_does_not_also_sync_after_gradients(self):
        a = self.agent(target_sync_mode="episodes")
        self.step(a); self.step(a)
        self.assertEqual(a.target_updates,0)
        a.update_target_network()
        self.assertEqual(a.target_updates,1)

    def test_decision_epsilon_is_independent_of_updates(self):
        a = self.agent(epsilon_schedule="decisions_linear",epsilon_decay_decisions=4)
        a.last_action_exploratory=True
        a.record_decision(); a.record_decision()
        self.assertAlmostEqual(a.epsilon,.525)
        self.step(a)
        self.assertAlmostEqual(a.epsilon,.525)
        self.assertEqual((a.total_decisions,a.exploratory_decisions),(2,2))

    def test_update_limit_is_exact_and_experience_is_still_accounted(self):
        a = self.agent(); a.update_limit=1
        self.assertIsNotNone(self.step(a))
        self.assertIsNone(self.step(a))
        self.assertEqual((a.training_updates,a.total_transitions,len(a.memory)),(1,2,2))

    def test_resume_restores_counters_epsilon_and_sync_boundary(self):
        a = self.agent(epsilon_schedule="decisions_linear",epsilon_decay_decisions=4)
        a.last_action_exploratory=True
        a.record_decision(); a.record_decision(); self.step(a)
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            path=Path(d)/"checkpoint.pth"
            a.save_model(path,episode=3,full_state=True)
            b=self.agent(epsilon_schedule="decisions_linear",epsilon_decay_decisions=4)
            b.load_model(path,resume=True)
            self.assertEqual(a.training_statistics(),b.training_statistics())
            self.assertEqual(b.loaded_episode,3)
            self.assertAlmostEqual(a.epsilon,b.epsilon)
            self.step(b)
            self.assertEqual((b.training_updates,b.target_updates),(2,1))
            with self.assertRaisesRegex(ValueError,"learning settings"):
                self.agent(gamma=.5).load_model(path,resume=True)

    def test_missing_historical_counters_are_explicit_and_new_resume_requires_them(self):
        a=self.agent()
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            p=Path(d)/"checkpoint.pth";a.save_model(p,episode=1,full_state=True)
            payload=torch.load(p,weights_only=False);payload.pop("learning_stats");torch.save(payload,p)
            b=self.agent()
            with self.assertRaisesRegex(ValueError,"learning counters"):b.load_model(p,resume=True)
            b.load_model(p)
            self.assertFalse(b.training_statistics()["counters_complete"])
    def test_invalid_learning_settings_are_rejected(self):
        for opts in (dict(gamma=2),dict(target_sync_interval=0),dict(reward_scale=0),
                     dict(epsilon_decay=float("nan")),dict(batch_size=32,replay_capacity=2)):
            with self.subTest(opts=opts),self.assertRaises(ValueError):
                LearningConfig(**opts)

    def test_shared_engine_records_choices_updates_and_reward_parts(self):
        config=SimulationConfig(arrival_steps=2,max_wait=20,mission_timeout=10,
                                treatment_seconds=3,bed_fraction=.1)
        a=self.agent()
        with patch.object(fixtures,"FakeAgent",return_value=a):
            result,rows,_=fixtures.SharedEngineTests().run_case(config)
        self.assertEqual((result["episode_decisions"],result["training_updates"]),(2,2))
        self.assertEqual((a.total_decisions,result["episode_exploratory_decisions"]),(2,2))
        for r in rows:
            self.assertAlmostEqual(sum(r["reward_components"].values()),r["reward"])
            self.assertEqual(r["edge_id"],"e")
        self.assertAlmostEqual(sum(result["avg_reward_components"].values()),result["avg_reward"])

class V6FeatureTests(unittest.TestCase):
    def hospital(self):
        return dict(fixtures.hospital(),type=HOSPITAL_TYPES[0],estimated_travel_time=60.,
                    occupancy=.4,reachable=True)
    def state(self,**flags):
        return build_state_vector((0.,0.),1,[self.hospital()],patient=dict(occurrence_time=0.),
                                  now=120.,config=SimulationConfig(**flags))
    def test_equal_shape_controls_include_wait_and_occupancy(self):
        zero=self.state(); margin=self.state(golden_margin_feature=True)
        reward=self.state(candidate_reward_feature=True)
        self.assertEqual((state_dimension(9),len(zero)),(115,27))
        self.assertEqual((zero[25],zero[26]),(0.,0.))
        self.assertAlmostEqual(margin[25],-.5)
        self.assertAlmostEqual(reward[26],-.132,places=6)
        np.testing.assert_array_equal(np.delete(zero,25),np.delete(margin,25))
        np.testing.assert_array_equal(np.delete(zero,26),np.delete(reward,26))
    def test_future_outcome_fields_do_not_enter_observation(self):
        config=SimulationConfig(golden_margin_feature=True,candidate_reward_feature=True)
        state=build_state_vector((0.,0.),1,[self.hospital()],
              patient=dict(occurrence_time=0.,actual_reward=999.,finish_time=999.),now=120.,config=config)
        np.testing.assert_array_equal(state,self.state(golden_margin_feature=True,candidate_reward_feature=True))
    def test_hospital_permutation_permutes_v6_scores(self):
        net=HospitalFeatureQNetwork(38,2); state=torch.rand(3,38)
        swapped=torch.cat((state[:,:16],state[:,16:].reshape(3,2,11)[:,[1,0]].reshape(3,22)),1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(swapped),net(state)[:,[1,0]],atol=1e-6))
    def test_greedy_reward_can_trade_time_for_suitability(self):
        a=dict(self.hospital(),estimated_travel_time=100.,occupancy=.7)
        b=dict(self.hospital(),type=HOSPITAL_TYPES[3],estimated_travel_time=110.,occupancy=.2)
        router=HospitalRouters([a,b])
        self.assertEqual(router.shortest_time_strategy([0,1]),0)
        self.assertEqual(router.expected_reward_strategy(5,[0,1]),1)

class V6ManifestTests(unittest.TestCase):
    def manifest(self,config=None,learning=None):
        with patch.object(simulation_setup,"network_digest",return_value="abc"):
            return simulation_setup.make_manifest(None,[fixtures.hospital()],["e"],[["h"]],
                 config or SimulationConfig(),10,"severity",300,(),learning_config=learning)
    def test_learning_and_features_change_signature(self):
        base=self.manifest()
        others=[self.manifest(learning=replace(LearningConfig(),gamma=.9)),
                self.manifest(learning=replace(LearningConfig(),target_sync_interval=500)),
                self.manifest(learning=replace(LearningConfig(),epsilon_decay=.999)),
                self.manifest(config=SimulationConfig(golden_margin_feature=True))]
        self.assertTrue(all(m["signature"]!=base["signature"] for m in others))
    def test_loader_round_trip_and_manifest_learning_guard(self):
        manifest=self.manifest()
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            mp=Path(d)/"manifest.json"; mp.write_text(json.dumps(manifest),encoding="utf-8")
            a=DQNAmbulanceAgent(27,1,architecture="hospital_shared_v6",learning_config=LearningConfig())
            a.metadata=dict(manifest_path=str(mp),experiment_signature=manifest["signature"])
            path=Path(d)/"model_best.pth"; a.save_model(path,episode=3)
            with patch.object(simulation_setup,"SumoMedicalEnvironment",return_value=SimpleNamespace()), \
                 patch.object(simulation_setup,"network_digest",return_value="abc"):
                _,_,b,config,_=simulation_setup.load_experiment(path)
                self.assertEqual((b.state_dim,config.observation_version),(27,5))
                np.testing.assert_allclose(a.q_values(np.zeros(27)),b.q_values(np.zeros(27)))
                saved=torch.load(path,weights_only=False); saved["learning_config"]["gamma"]=.5
                torch.save(saved,path)
                with self.assertRaisesRegex(ValueError,"learning settings"):
                    simulation_setup.load_experiment(path)
    def test_cli_infers_validation_scenario_and_passes_controls(self):
        argv=["train_agent.py","--train-scenarios","6","--golden-margin-feature",
              "--target-sync-interval","500","--max-updates","1000"]
        with patch.object(sys,"argv",argv),patch.object(train_agent,"train") as run:
            train_agent.main()
        self.assertEqual(run.call_args.args[13],[6])
        self.assertTrue(run.call_args.args[10].golden_margin_feature)
        self.assertFalse(run.call_args.args[10].candidate_reward_feature)
        self.assertEqual(run.call_args.kwargs["learning_config"].target_sync_interval,500)
        self.assertEqual(run.call_args.kwargs["max_updates"],1000)
    def test_update_budget_finishes_episode_and_saves_actual_count(self):
        result=dict(generated_patients=2,missions=2,arrived=2,failed=0,censored=0,
                    golden_rate=50.,avg_wait=0.,reward=4.,avg_reward=2.,avg_loss=0.)
        def episode(*args,**kwargs):
            kwargs["agent"].training_updates=1
            return dict(result)
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()), \
             patch.object(train_agent,"prepare_environment",return_value=(None,[fixtures.hospital()],["e"],[["h"]])), \
             patch.object(train_agent,"make_manifest",return_value={"version":6}), \
             patch.object(train_agent,"run_episode",side_effect=episode) as run, \
             patch.object(train_agent,"validate_panel",return_value=dict(avg_reward=2.,arrival_rate=100.)) as panel, \
             patch.object(train_agent,"append_record") as log,patch.object(train_agent,"plot_training_metrics"):
            train_agent.train(num_episodes=20,eval_every=9,max_updates=1,output_root=d)
            self.assertEqual(run.call_count,1); panel.assert_called_once()
            end=next(c for c in log.call_args_list if c.args[2]=="run_end")
            self.assertEqual(end.kwargs["metrics"]["total_completed_episodes"],1)
