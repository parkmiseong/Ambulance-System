"""Admission after physical arrival, same-vehicle redirects, no advance bed hold."""
import copy
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
from test_simulation_system import FakeAgent, FakeTraCI


class StoppingTraCI(FakeTraCI):
    def __init__(self, duration):
        super().__init__(duration)
        self.stopped, self.destinations = set(), {}
        self.spawns, self.redirects = [], []
        self.vehicle.isStopped = lambda vid: vid in self.stopped
        self.vehicle.getRoadID = lambda vid: self.destinations[vid] if vid in self.stopped else "e"
        self.vehicle.getPosition = lambda vid: (100., 0.)
        self.vehicle.resume = self.resume
        self.vehicle.setRoute = self.set_route
        self.vehicle.setStop = lambda *a, **kw: None

    def simulationStep(self):
        self.now += 1
        self.arrived = []
        self.stopped.update(vid for vid, end in self.running.items() if end <= self.now)

    def spawn(self, edge, destination, vid, *args, **kwargs):
        assert kwargs.get("arrival_stop_pos") == 99.5
        self.spawns.append((vid, self.now, edge, destination))
        self.destinations[vid] = destination
        return super().spawn(edge, destination, vid, *args, **kwargs)

    def resume(self, vid):
        self.stopped.remove(vid)

    def set_route(self, vid, edges):
        self.redirects.append((vid, self.now, list(edges)))
        self.destinations[vid] = edges[-1]
        self.running[vid] = self.now + self.duration

    def remove(self, vid):
        super().remove(vid)
        self.stopped.discard(vid)


class ArrivalAdmissionTests(unittest.TestCase):
    def run_case(self, occupancy=(.7, .2), patients=(1, 2), duration=2, timeout=20,
                 accepting=None, training=True, mutate=None):
        config = SimulationConfig(arrival_steps=max(patients), max_wait=12, mission_timeout=timeout,
                                  bed_fraction=.1, treatment_seconds=3, admission_mode="arrival",
                                  congestion_reward_mode="balanced")
        hospitals = [dict(id=f"h{i}", name=f"Hospital {i}", type="권역응급의료센터", occupancy=occ,
                          accepting=(accepting or [True] * len(occupancy))[i], sumo_x=i, sumo_y=0.)
                     for i, occ in enumerate(occupancy)]
        fake = StoppingTraCI(duration)
        edge = SimpleNamespace(getFromNode=lambda: SimpleNamespace(getCoord=lambda: (0., 0.)),
                               getLength=lambda: 100., getLanes=lambda: [SimpleNamespace(allows=lambda _: True)])
        env = SimpleNamespace(ambulance_edges=["e"], hospital_access_edges=[[h["id"]] for h in hospitals],
                              net=SimpleNamespace(getEdge=lambda _: edge))
        agent = FakeAgent(state_dimension(len(hospitals)), len(hospitals))
        rows, events, frames, holder = [], [], [], []
        schedule = {tick: dict(patient_id=f"p{tick}", edge_id="e", occurrence_time=float(tick), ktas=1, pre_ktas=1) for tick in patients}
        def capacity(actual, settings):
            for hospital, value in zip(actual, occupancy):
                hospital["occupancy"] = value
            beds = HospitalCapacity(actual, settings)
            holder.append(beds)
            return beds
        def routes(_env, origin, actual, access):
            routed = copy.deepcopy(actual)
            for i, h in enumerate(routed):
                h.update(reachable=bool(access[i]), hospital_edge=h["id"],
                         route_edges=[origin, h["id"]] if origin != h["id"] else [origin],
                         route_length=100., estimated_travel_time=2.+i)
            return routed, 0
        def frame(snap):
            frames.append(dict(time=snap["time"], rejections=snap["rejection_count"],
                               reservations=sum(h["reserved_count"] for h in snap["hospitals"]),
                               active=len(snap["manager"].active)))
            if mutate:
                mutate(snap, holder[0])
        with patch.object(simulation, "traci", fake), patch.object(ambulance_sim, "traci", fake), \
             patch.object(simulation, "start_sumo", fake.start), \
             patch.object(simulation, "TrafficRouter", return_value=SimpleNamespace(reset=lambda: None, refresh=lambda **kw: None)), \
             patch.object(simulation, "HospitalCapacity", side_effect=capacity), \
             patch.object(simulation, "generate_schedule", return_value=schedule), \
             patch.object(simulation, "get_reachable_hospital_routes", side_effect=routes), \
             patch.object(ambulance_sim, "spawn_at_patient", side_effect=fake.spawn):
            result = simulation.run_episode(env, hospitals, dict(id=1, prob=1., scale=.3), 42,
                "dqn", config, agent=agent, training=training, ambulance_count=2,
                on_outcome=rows.append, on_rejection=events.append, on_step=frame)
        self.assertEqual(result["remaining_reservations"], 0)
        self.assertTrue(all(frame["reservations"] == 0 for frame in frames))
        self.assertEqual(result["generated_patients"], len(rows))
        self.assertAlmostEqual(result["reward"], sum(row["reward"] for row in rows))
        self.assertAlmostEqual(result["reward"], sum(result["reward_components"].values()))
        self.assertAlmostEqual(result["learning_reward"], result["reward"])
        return result, {row["patient_id"]: row for row in rows}, events, fake, frames, agent

    def test_no_refusal_before_arrival_and_redirect_keeps_the_same_ambulance(self):
        result, rows, events, fake, frames, _ = self.run_case()
        row = rows["p2"]
        self.assertEqual(result["arrived"], 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["stage"], "after_arrival")
        self.assertEqual(events[0]["time"], 4.)
        self.assertGreater(events[0]["time"], events[0]["leg_dispatch_time"])
        self.assertEqual([frame["rejections"] for frame in frames if frame["time"] < 4], [0, 0])
        self.assertEqual([visit["hospital_id"] for visit in row["hospital_visits"]], ["h0", "h1"])
        self.assertEqual([visit["accepted"] for visit in row["hospital_visits"]], [False, True])
        self.assertEqual({visit["vehicle_id"] for visit in row["hospital_visits"]}, {row["vehicle_id"]})
        self.assertEqual(fake.redirects, [(row["vehicle_id"], 4., ["h0", "h1"])])
        self.assertEqual(len(fake.spawns), 2)
        self.assertEqual(row["wait_seconds"], 0.)
        self.assertEqual(row["travel_seconds"], 4.)
        self.assertEqual(row["reward_components"]["rejection"], -12.)

    def test_known_full_hospital_is_skipped_without_penalty_during_predeparture_check(self):
        _, rows, events, _, _, _ = self.run_case(occupancy=(.8, .2), patients=(1,))
        self.assertEqual(rows["p1"]["hospital_id"], "h1")
        self.assertEqual(events, [])

    def test_all_known_full_hospitals_delay_departure_without_refusal_cost(self):
        _, rows, events, fake, _, _ = self.run_case(occupancy=(.8, .8), patients=(1,))
        self.assertEqual(fake.spawns, [])
        self.assertEqual(events, [])
        self.assertEqual(rows["p1"]["reason"], "waiting_timeout")
        self.assertEqual(rows["p1"]["reward"], -85.)

    def test_known_not_accepting_is_skipped_without_a_refusal_cost(self):
        _, rows, events, _, _, _ = self.run_case(occupancy=(.2, .2), patients=(1,), accepting=[False, True])
        self.assertEqual(rows["p1"]["hospital_id"], "h1")
        self.assertEqual(events, [])

    def test_sudden_capacity_change_is_checked_at_the_actual_hospital(self):
        def fill_during_transport(snap, beds):
            if snap["time"] == 2:
                beds.background[0] = .8
                beds.refresh(2.)
        _, rows, events, _, _, _ = self.run_case(occupancy=(.2, .2), patients=(1,), mutate=fill_during_transport)
        self.assertEqual(rows["p1"]["hospital_id"], "h1")
        self.assertEqual(events[0]["time"], 3.)
        self.assertAlmostEqual(events[0]["occupancy_at_arrival"], .8)

    def test_repeated_waiting_at_a_full_hospital_is_not_repeated_refusal(self):
        _, rows, events, fake, _, _ = self.run_case(occupancy=(.7,))
        self.assertEqual(len(events), 1)
        self.assertEqual(fake.redirects, [])
        self.assertTrue(rows["p2"]["arrived"])
        self.assertEqual(rows["p2"]["hospital_visits"][0]["reason"], "accepted_after_wait")
        self.assertEqual(rows["p2"]["finish_time"], 6.)

    def test_timeout_covers_the_entire_transport_including_redirects(self):
        _, rows, events, _, _, _ = self.run_case(timeout=3)
        self.assertEqual(len(events), 1)
        self.assertEqual(rows["p2"]["reason"], "mission_timeout")
        self.assertEqual(rows["p2"]["reward"], -97.)

    def test_censored_redirect_keeps_refusal_cost_without_failure_cost(self):
        with patch.object(SimulationConfig, "drain_deadline", new_callable=PropertyMock, return_value=3):
            _, rows, events, _, _, _ = self.run_case()
        self.assertEqual(len(events), 1)
        self.assertTrue(rows["p2"]["censored"])
        self.assertEqual(rows["p2"]["reward"], -12.)

    def test_never_reaching_a_hospital_does_not_generate_a_refusal(self):
        _, rows, events, _, _, _ = self.run_case(patients=(1,), duration=100, timeout=3)
        self.assertEqual(events, [])
        self.assertEqual(rows["p1"]["reward"], -85.)

    def test_training_and_evaluation_have_identical_arrival_costs(self):
        trained, _, _, _, _, agent = self.run_case()
        evaluated, _, _, _, _, _ = self.run_case(training=False)
        self.assertEqual(trained["reward"], evaluated["reward"])
        self.assertAlmostEqual(sum(args[2] for args, _ in agent.records), trained["reward"])

    def test_arrival_profile_is_explicit_without_changing_legacy_defaults(self):
        self.assertNotIn("admission_mode", SimulationConfig().to_dict())
        self.assertEqual(SimulationConfig(admission_mode="arrival").to_dict()["admission_mode"], "arrival")
