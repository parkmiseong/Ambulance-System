"""Capacity, observation, accounting, and shared-engine regression tests."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from ambulance_sim import DEFERRED, MissionManager, Mission
from baseline import build_state_vector, state_dimension, HospitalRouters
from experiment_config import SimulationConfig, generate_schedule, seed_everything, isolated_random_state
from hospital_capacity import HospitalCapacity
from hospital_policy import valid_actions
from reward import HOSPITAL_TYPES, reward_components
import simulation
import train_agent
from run_scenarios import aggregate_results
from simulation_setup import make_manifest, manifest_signature


def hospital():
    return dict(id="H", type=HOSPITAL_TYPES[0], occupancy=.7, sumo_x=1., sumo_y=0.,
                reachable=True, route_length=10., estimated_travel_time=2.)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.hospitals = [hospital()]
        self.config = SimulationConfig(bed_fraction=.1, treatment_seconds=3)
        self.beds = HospitalCapacity(self.hospitals, self.config)

    def test_reservation_blocks_overbooking_and_failure_releases_it(self):
        self.beds.reserve(0, "A")
        self.assertEqual(valid_actions(self.hospitals, 1), [])
        with self.assertRaises(ValueError):
            self.beds.reserve(0, "B")
        self.beds.complete("A", False, 1)
        self.assertEqual(valid_actions(self.hospitals, 1), [0])
        self.assertAlmostEqual(self.hospitals[0]["occupancy"], .7)

    def test_arrival_converts_reservation_then_treatment_releases_once(self):
        self.beds.reserve(0, "A")
        self.beds.complete("A", True, 2)
        self.assertEqual(self.hospitals[0]["reserved_count"], 0)
        self.assertEqual(self.hospitals[0]["admitted_count"], 1)
        self.assertAlmostEqual(self.hospitals[0]["occupancy"], .8)
        self.beds.refresh(4)
        self.assertEqual(valid_actions(self.hospitals, 1), [])
        self.beds.refresh(5)
        self.beds.refresh(6)
        self.assertAlmostEqual(self.hospitals[0]["occupancy"], .7)
        self.assertEqual(self.hospitals[0]["admitted_count"], 0)

    def test_duplicate_reservation_and_backwards_time_are_rejected(self):
        self.beds.reserve(0, "A")
        with self.assertRaises(ValueError):
            self.beds.reserve(0, "A")
        self.beds.refresh(5)
        with self.assertRaises(ValueError):
            self.beds.refresh(4)


class ConfigAndStateTests(unittest.TestCase):
    def test_quiet_stdout_suppresses_optional_messages(self):
        from io import StringIO
        from contextlib import redirect_stdout
        visible = StringIO()
        with redirect_stdout(visible):
            with train_agent.suppress_stdout(True):
                print("hidden")
            with train_agent.suppress_stdout(False):
                print("shown")
        self.assertEqual(visible.getvalue(), "shown\n")

    def test_curriculum_keeps_final_level_and_resumes_by_episode(self):
        from experiment_config import curriculum_multiplier
        levels = (.1, .3, 1.)
        self.assertEqual([curriculum_multiplier(ep, levels) for ep in (1, 9, 10, 18, 19, 27, 28, 100)],
                         [.1, .1, .3, .3, 1., 1., 1., 1.])
        self.assertEqual(curriculum_multiplier(100, ()), 1.)

    def test_demand_levels_share_patient_attributes_and_nested_arrivals(self):
        base = SimulationConfig(arrival_steps=100)
        scenario = dict(prob=.4)
        low = generate_schedule(scenario, ["a", "b"], 42, replace(base, demand_scale=.1))
        high = generate_schedule(scenario, ["a", "b"], 42, base)
        self.assertTrue(set(low).issubset(high))
        self.assertTrue(low)
        for step in low:
            self.assertEqual(low[step], high[step])

    def test_validation_preserves_all_random_streams(self):
        seed_everything(123)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        expected = (random.random(), np.random.rand(), torch.rand(1))
        random.setstate(before[0]); np.random.set_state(before[1]); torch.set_rng_state(before[2])
        with isolated_random_state():
            seed_everything(456)
            random.random(); np.random.rand(); torch.rand(2)
        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_observation_distinguishes_wait_queue_and_inbound_reservations(self):
        h = hospital()
        empty = build_state_vector((0, 0), 2, [h])
        manager = MissionManager()
        manager.waiting.append(dict(ktas=1, occurrence_time=0))
        h["reserved_count"] = 1
        occupied = build_state_vector((0, 0), 2, [h],
                                     patient=dict(occurrence_time=0), manager=manager, now=10)
        self.assertEqual(len(occupied), state_dimension(1))
        self.assertFalse(np.array_equal(empty, occupied))
        self.assertTrue(np.isfinite(occupied).all())

    def test_shortest_distance_uses_road_length(self):
        h1, h2 = hospital(), hospital()
        h1.update(sumo_x=1, route_length=1000)
        h2.update(sumo_x=10, route_length=20)
        self.assertEqual(HospitalRouters([h1, h2]).shortest_distance_strategy([0, 1]), 1)

    def test_failed_mission_cannot_earn_a_golden_time_bonus(self):
        parts = reward_components(1, 1, hospital(), route_failed=True)
        self.assertEqual(parts["golden"], 0)

    def test_manifest_round_trip_and_tampering(self):
        config = SimulationConfig()
        with patch("simulation_setup.network_digest", return_value="abc"):
            m = make_manifest(None, [hospital()], ["e"], [["h"]], config, 10, "severity", 300, ())
        restored = json.loads(json.dumps(m))
        self.assertEqual(m, restored)
        self.assertEqual(manifest_signature(restored), m["signature"])
        restored["hospitals"][0]["type"] = "changed"
        self.assertNotEqual(manifest_signature(restored), m["signature"])

    def test_invalid_config_values_are_rejected(self):
        for options in (dict(demand_scale=-1), dict(treatment_seconds=0),
                        dict(bed_fraction=.9), dict(max_wait=float("nan")),
                        dict(arrival_steps=0), dict(arrival_steps=1.5)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SimulationConfig(**options)


class FakeTraCI:
    class TraCIException(Exception):
        pass

    def __init__(self, duration=2):
        self.now, self.duration, self.loaded = 0., duration, False
        self.running, self.arrived = {}, []
        self.simulation = SimpleNamespace(getTime=lambda: self.now,
                                          getArrivedIDList=lambda: self.arrived)
        self.vehicle = SimpleNamespace(getIDList=lambda: list(self.running), remove=self.remove)

    def start(self, *args, **kwargs):
        self.loaded = True

    def close(self):
        self.loaded = False

    def isLoaded(self):
        return self.loaded

    def simulationStep(self):
        self.now += 1
        self.arrived = [vid for vid, end in self.running.items() if end <= self.now]
        for vid in self.arrived:
            self.running.pop(vid)

    def remove(self, vid):
        self.running.pop(vid, None)

    def spawn(self, edge, destination, vid, *args, **kwargs):
        self.running[vid] = self.now + self.duration
        return True, "created"


class FakeModel:
    def __init__(self):
        self.training = True

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)


class FakeAgent:
    def __init__(self, state_dim=24, action_dim=1):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.records, self.saves, self.memory = [], [], []
        self.epsilon, self.metadata, self.model = .5, {}, FakeModel()

    def select_action(self, state, valid_actions):
        return valid_actions[0]

    def store_transition(self, *args, **kwargs):
        self.records.append((args, kwargs))
        self.memory.append(args)

    def train_step(self, batch_size):
        return .5

    def update_target_network(self):
        pass

    def save_model(self, path, **kwargs):
        self.saves.append((path, kwargs, copy.deepcopy(self.metadata)))


class SharedEngineTests(unittest.TestCase):
    def run_case(self, config=None, duration=2, spawn_failure=False, training=True):
        import ambulance_sim
        fake = FakeTraCI(duration)
        config = config or SimulationConfig(arrival_steps=2, max_wait=20, mission_timeout=10,
                                             treatment_seconds=3, bed_fraction=.1)
        schedule = {
            i:dict(patient_id=f"p{i}", occurrence_time=float(i), edge_id="e", ktas=1, pre_ktas=1)
            for i in (1, 2)
        }
        edge = SimpleNamespace(getFromNode=lambda: SimpleNamespace(getCoord=lambda:(0., 0.)))
        env = SimpleNamespace(ambulance_edges=["e"], hospital_access_edges=[["h"]],
                              net=SimpleNamespace(getEdge=lambda edge_id: edge))
        agent, rows = FakeAgent(), []
        def capacity(hospitals, config):
            for h in hospitals:
                h["occupancy"] = .7
            return HospitalCapacity(hospitals, config)
        def routes(env, edge, hospitals, access):
            result = copy.deepcopy(hospitals)
            result[0].update(reachable=True, hospital_edge="h", route_edges=["e", "h"],
                             route_length=10., estimated_travel_time=2.)
            return result, 0
        with patch.object(simulation, "traci", fake), patch.object(ambulance_sim, "traci", fake), \
             patch.object(simulation, "start_sumo", fake.start), \
             patch.object(simulation, "TrafficRouter", return_value=SimpleNamespace(reset=lambda:None, refresh=lambda:None)), \
             patch.object(simulation, "get_reachable_hospital_routes", side_effect=routes), \
             patch.object(simulation, "generate_schedule", return_value=schedule), \
             patch.object(simulation, "HospitalCapacity", side_effect=capacity), \
             patch.object(ambulance_sim, "spawn_at_patient",
                          side_effect=(lambda *a, **k:(False, "spawn_failed")) if spawn_failure else fake.spawn):
            result = simulation.run_episode(
                env, [hospital()], dict(id=1, prob=.2, scale=.3), 42, "dqn", config,
                agent=agent, training=training, ambulance_count=1, on_outcome=rows.append)
        return result, rows, agent

    def test_temporary_capacity_shortage_is_retried_until_discharge(self):
        result, rows, agent = self.run_case()
        self.assertEqual((result["arrived"], result["failed"], result["censored"]), (2, 0, 0))
        self.assertGreater(result["route_diagnostics"]["capacity_deferrals"], 0)
        self.assertLessEqual(max(result["peak_hospital_occupancy"]), .85 + 1e-9)
        self.assertEqual(result["remaining_reservations"], 0)
        self.assertGreater(rows[1]["dispatch_time"], rows[0]["finish_time"])
        self.assertEqual([r[0][4] for r in agent.records], [False, True])
        self.assertAlmostEqual(sum(r[0][2] for r in agent.records), result["reward"])

    def test_waiting_failure_is_in_reward_and_all_patient_denominators(self):
        config = SimulationConfig(arrival_steps=2, max_wait=2, mission_timeout=10,
                                  treatment_seconds=10, bed_fraction=.1)
        result, rows, agent = self.run_case(config)
        self.assertEqual(result["generated_patients"], 2)
        self.assertEqual((result["arrived"], result["failed"]), (1, 1))
        self.assertEqual(result["arrival_rate"], 50.)
        self.assertEqual(result["golden_rate"], 50.)
        self.assertEqual(result["failure_reasons"], {"waiting_timeout": 1})
        self.assertAlmostEqual(result["avg_reward"], sum(r["reward"] for r in rows) / 2)
        self.assertEqual(rows[1]["reward"], -20)
        self.assertAlmostEqual(sum(r[0][2] for r in agent.records), result["learning_reward"])

    def test_spawn_failure_rolls_back_reservations_without_fake_censoring(self):
        result, rows, agent = self.run_case(spawn_failure=True)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["remaining_reservations"], 0)
        self.assertEqual(result["censored"], 0)
        self.assertEqual(result["failure_reasons"], {"spawn_failed": 2})

    def test_late_dispatch_gets_full_mission_timeout(self):
        config = SimulationConfig(arrival_steps=2, max_wait=20, mission_timeout=10,
                                  treatment_seconds=3, bed_fraction=.1)
        result, rows, _ = self.run_case(config, duration=9)
        self.assertGreater(rows[-1]["finish_time"], config.arrival_steps + config.mission_timeout)
        self.assertEqual(result["censored"], 0)
        self.assertEqual(result["arrived"], 2)

    def test_training_and_evaluation_have_identical_dynamics(self):
        trained, training_rows, _ = self.run_case(training=True)
        evaluated, evaluation_rows, _ = self.run_case(training=False)
        self.assertEqual(training_rows, evaluation_rows)
        for field in ("arrived", "failed", "reward", "golden_rate", "avg_wait", "peak_hospital_occupancy"):
            self.assertEqual(trained[field], evaluated[field])

    def test_censoring_is_not_recorded_as_operational_failure(self):
        import ambulance_sim
        fake, rows = FakeTraCI(), []
        manager = MissionManager()
        patient = dict(occurrence_time=0., ktas=1, edge_id="e")
        manager.active["a"] = Mission(patient, "a", 0., (0,))
        with patch.object(ambulance_sim, "traci", fake):
            manager.close_unfinished(lambda *args:rows.append(args))
        self.assertEqual((manager.completed, manager.failed, manager.censored), (0, 0, 1))
        self.assertEqual(dict(manager.failure_reasons), {})
        self.assertEqual(len(rows), 1)

    def test_unexpected_dispatch_error_propagates(self):
        manager = MissionManager()
        manager.enqueue(dict(occurrence_time=0., ktas=1, edge_id="e"))
        def dispatch(*args):
            raise RuntimeError("optimizer failed")
        with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
            manager._dispatch(0., dispatch, lambda *a:None, "a")

    def test_aggregate_uses_all_patients_and_distinguishes_censoring(self):
        result, _, _ = self.run_case()
        summary = aggregate_results([result, result])
        self.assertEqual(summary["generated_patients"], 4)
        self.assertEqual(summary["arrival_rate"], 100.)
        self.assertEqual(summary["censored"], 0)
        self.assertEqual(summary["reward_seed_std"], 0.)


class ModelSelectionTests(unittest.TestCase):
    def test_best_validation_model_is_not_overwritten_by_last_episode(self):
        agent = FakeAgent()
        result = dict(generated_patients=2, missions=2, arrived=1, failed=1, censored=0,
                      golden_rate=50., avg_wait=1., reward=-40., avg_reward=-20., avg_loss=.1)
        panel = [dict(avg_reward=-5., arrival_rate=50., golden_rate=50.),
                 dict(avg_reward=-10., arrival_rate=50., golden_rate=50.)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(train_agent, "prepare_environment", return_value=(None, [hospital()], ["e"], [["h"]])), \
             patch.object(train_agent, "make_manifest", return_value={"version":3}), \
             patch.object(train_agent, "DQNAmbulanceAgent", return_value=agent), \
             patch.object(train_agent, "run_episode", side_effect=[copy.deepcopy(result),copy.deepcopy(result)]), \
             patch.object(train_agent, "validate_panel", side_effect=panel), \
             patch.object(train_agent, "append_record"), patch.object(train_agent, "plot_training_metrics"):
            train_agent.train(num_episodes=2, eval_every=1, config=SimulationConfig(arrival_steps=2),
                              validation_seeds=(30001,), output_root=directory)
        best = [s for s in agent.saves if s[0].endswith("_best.pth")]
        last = [s for s in agent.saves if s[0].endswith("_last.pth")]
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0][1]["episode"], 1)
        self.assertEqual(last[-1][1]["episode"], 2)
        self.assertEqual(last[-1][2]["best_validation_reward"], -5.)


if __name__ == "__main__":
    unittest.main()
