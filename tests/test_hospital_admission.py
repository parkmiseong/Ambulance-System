"""Actual admission refusals, alternative selection, and one-time reward accounting."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
import ambulance_sim
import simulation
from baseline import state_dimension
from experiment_config import SimulationConfig
from hospital_capacity import HospitalCapacity
from reward import failure_penalty, reward_components
from test_simulation_system import FakeAgent, FakeTraCI


class AdmissionFlowTests(unittest.TestCase):
    def run_case(self, occupancy=(.8, .2), patients=(1,), *, training=True, mode="request",
                 accepting=None, types=None, duration=2, timeout=20, congestion_reward_mode="penalty"):
        config = SimulationConfig(arrival_steps=max(patients), max_wait=30, mission_timeout=timeout,
                                  treatment_seconds=3, bed_fraction=.1, admission_mode=mode,
                                  congestion_reward_mode=congestion_reward_mode)
        hospitals = [dict(id=f"h{i}", name=f"Hospital {i}", type=(types or ["권역응급의료센터"]*len(occupancy))[i],
                          occupancy=occ, sumo_x=i, sumo_y=0., accepting=(accepting or [True]*len(occupancy))[i])
                     for i, occ in enumerate(occupancy)]
        config = replace(config, candidate_limit=1)
        fake = FakeTraCI(duration)
        edge = SimpleNamespace(getFromNode=lambda: SimpleNamespace(getCoord=lambda: (0., 0.)))
        env = SimpleNamespace(ambulance_edges=["e"], hospital_access_edges=[[h["id"]] for h in hospitals],
                              net=SimpleNamespace(getEdge=lambda _: edge))
        agent = FakeAgent(state_dimension(len(hospitals)), len(hospitals))
        rows, events, snapshots = [], [], []
        schedule = {tick: dict(patient_id=f"p{tick}", edge_id="e", occurrence_time=float(tick), ktas=1, pre_ktas=1) for tick in patients}
        def capacity(actual, settings):
            for h, value in zip(actual, occupancy):
                h["occupancy"] = value
            return HospitalCapacity(actual, settings)
        def routes(_env, _edge, actual, _access):
            routed = copy.deepcopy(actual)
            for i, h in enumerate(routed):
                h.update(reachable=True, hospital_edge=h["id"], route_edges=["e", h["id"]],
                         route_length=10.+i, estimated_travel_time=2.+i)
            return routed, 0
        with patch.object(simulation, "traci", fake), patch.object(ambulance_sim, "traci", fake), \
             patch.object(simulation, "start_sumo", fake.start), \
             patch.object(simulation, "TrafficRouter", return_value=SimpleNamespace(reset=lambda:None, refresh=lambda:None)), \
             patch.object(simulation, "HospitalCapacity", side_effect=capacity), \
             patch.object(simulation, "generate_schedule", return_value=schedule), \
             patch.object(simulation, "get_reachable_hospital_routes", side_effect=routes), \
             patch.object(ambulance_sim, "spawn_at_patient", side_effect=fake.spawn):
            result = simulation.run_episode(env, hospitals, dict(id=1, prob=.2, scale=.3), 42,
                "dqn", config, agent=agent, training=training, ambulance_count=2,
                on_outcome=rows.append, on_rejection=events.append,
                on_step=lambda snap: snapshots.append((snap["time"],snap["rejection_count"],snap["rejection_penalty"])))
        return result, rows, agent, events, snapshots

    def assert_accounting(self, result, agent):
        self.assertAlmostEqual(result["reward"], sum(result["reward_components"].values()))
        self.assertAlmostEqual(result["reward"], result["learning_reward"])
        self.assertAlmostEqual(result["reward"], sum(record[0][2] for record in agent.records))
        self.assertEqual(result["remaining_reservations"], 0)

    def test_balanced_rewards_include_other_patients_reservations_but_not_own_reservation(self):
        result, rows, agent, _, _ = self.run_case(occupancy=(.2, .75), patients=(1, 2),
                                                congestion_reward_mode="balanced")
        self.assertEqual(result["congestion_reward_mode"], "balanced")
        self.assertEqual([row["hospital_id"] for row in rows], ["h0", "h0"])
        self.assertAlmostEqual(rows[0]["occupancy_at_selection"], .2)
        self.assertAlmostEqual(rows[1]["occupancy_at_selection"], .3)
        self.assertAlmostEqual(rows[0]["reward_components"]["occupancy"], 4.8)
        self.assertAlmostEqual(rows[1]["reward_components"]["occupancy"], 3.2)
        self.assert_accounting(result, agent)

    def test_refused_hospital_is_penalized_then_an_alternative_is_selected(self):
        result, rows, agent, events, snapshots = self.run_case()
        self.assertEqual(rows[0]["hospital_id"], "h1")
        self.assertTrue(rows[0]["arrived"])
        self.assertEqual(rows[0]["rejection_count"], 1)
        self.assertEqual(events[0]["hospital_id"], "h0")
        self.assertEqual(events[0]["penalty"], -12.)
        self.assertEqual(rows[0]["reward_components"]["rejection"], -12.)
        self.assertEqual(result["admission_requests"], 2)
        self.assertEqual(result["admission_rejection_reasons"], {"hospital_full":1})
        self.assertEqual([record[0][1] for record in agent.records], [0,1])
        self.assertEqual(agent.records[0][0][2], -12.)
        self.assertTrue(any(count == 1 and penalty == -12. for _,count,penalty in snapshots))
        self.assert_accounting(result, agent)

    def test_waiting_polls_do_not_repeat_refusal_penalties_and_failure_keeps_both_components(self):
        result, rows, agent, events, _ = self.run_case(occupancy=(.8,.8))
        self.assertEqual(len(events), 2)
        self.assertGreater(result["route_diagnostics"]["capacity_deferrals"], 2)
        self.assertEqual(rows[0]["reason"], "waiting_timeout")
        self.assertEqual(rows[0]["reward_components"]["route_failure"], -85.)
        self.assertEqual(rows[0]["reward_components"]["rejection"], -24.)
        self.assertEqual(result["reward"], -109.)
        self.assert_accounting(result, agent)

    def test_previously_refused_hospital_can_accept_after_a_bed_is_released(self):
        result, rows, agent, events, _ = self.run_case(occupancy=(.7,), patients=(1,2))
        self.assertEqual(result["arrived"], 2)
        self.assertEqual(len(events), 1)
        self.assertEqual([row["rejection_count"] for row in rows], [0,1])
        self.assertGreater(rows[1]["dispatch_time"], rows[0]["finish_time"])
        self.assert_accounting(result, agent)

    def test_capability_exclusion_is_not_an_actual_refusal(self):
        result, rows, agent, events, _ = self.run_case(occupancy=(.2,.2), types=["지역응급의료기관","권역응급의료센터"])
        self.assertEqual(rows[0]["hospital_id"], "h1")
        self.assertEqual(events, [])
        self.assertEqual(result["rejection_count"], 0)
        self.assertEqual(len(agent.records), 1)

    def test_not_accepting_is_a_refusal_even_with_a_free_bed(self):
        result, rows, agent, events, _ = self.run_case(occupancy=(.2,.2), accepting=[False,True])
        self.assertEqual(events[0]["reason"], "not_accepting")
        self.assertEqual(rows[0]["hospital_id"], "h1")
        self.assert_accounting(result, agent)

    def test_transport_failure_does_not_double_charge_previous_refusals(self):
        result, rows, agent, _, _ = self.run_case(duration=100,timeout=3)
        self.assertEqual(rows[0]["reward_components"]["route_failure"], -85.)
        self.assertEqual(rows[0]["reward_components"]["rejection"], -12.)
        self.assertEqual(result["reward"], -97.)
        self.assert_accounting(result, agent)

    def test_censoring_retains_incurred_refusals_without_a_failure_penalty(self):
        with patch.object(SimulationConfig,"drain_deadline",new_callable=PropertyMock,return_value=3):
            result, rows, agent, _, _ = self.run_case(occupancy=(.8,.8))
        self.assertEqual(result["censored"], 1)
        self.assertEqual(rows[0]["reward_components"]["route_failure"], 0.)
        self.assertEqual(result["reward"], -24.)
        self.assert_accounting(result, agent)

    def test_old_precheck_mode_and_evaluation_are_reproducible(self):
        old, old_rows, _, old_events, _ = self.run_case(mode="precheck")
        new, rows, _, _, _ = self.run_case()
        evaluated, eval_rows, _, _, _ = self.run_case(training=False)
        self.assertEqual(old_events, [])
        self.assertAlmostEqual(old["reward"] - 12., new["reward"])
        self.assertEqual(new["reward"], evaluated["reward"])
        self.assertEqual(old_rows[0]["dispatch_time"], rows[0]["dispatch_time"])
        self.assertEqual(eval_rows[0]["rejection_count"], rows[0]["rejection_count"])

    def test_failure_component_split_preserves_existing_total_and_old_config_signatures(self):
        parts = reward_components(1,1,dict(type="권역응급의료센터"),route_failed=True,rejection_count=2)
        self.assertEqual(sum(parts.values()), -failure_penalty(4,rejection_count=2))
        self.assertEqual(parts["rejection"], -24.)
        self.assertNotIn("admission_mode", SimulationConfig().to_dict())
        self.assertEqual(SimulationConfig(admission_mode="request").to_dict()["admission_mode"], "request")
        with self.assertRaises(ValueError):
            SimulationConfig(admission_mode="unknown")
