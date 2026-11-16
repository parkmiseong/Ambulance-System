"""Directed emergency connectivity and lane-specific stop/turn safety."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from route_safety import emergency_component, arrival_lane, prepare_access, lane_constrained_route


class Edge:
    def __init__(self, name, count=2, length=60.):
        self.name, self.length, self.connections = name, length, {}
        # Lane objects are keys in lane reachability sets.
        self.lanes = [Lane(f"{name}_{i}") for i in range(count)]
    def getID(self): return self.name
    def allows(self, _): return True
    def getLanes(self): return self.lanes
    def getLength(self): return self.length
    def getSpeed(self): return 20.
    def getAllowedOutgoing(self, _): return self.connections
    def connect(self, other, before=0, after=0):
        self.connections.setdefault(other, []).append(SimpleNamespace(
            getFromLane=lambda: self.lanes[before], getToLane=lambda: other.lanes[after]))


class Lane:
    def __init__(self, name): self.name = name
    def getID(self): return self.name
    def allows(self, _): return True


def network(*edges):
    return SimpleNamespace(getEdges=lambda: edges, getEdge=lambda key: next(e for e in edges if e.name == key))


class RouteSafetyTests(unittest.TestCase):
    def test_directed_disconnected_and_dead_end_roads_are_outside_mutual_component(self):
        a, b, isolated, dead_end = [Edge(name) for name in ('a','b','island','dead_end')]
        a.connect(b); b.connect(a); b.connect(dead_end)
        core, audit = emergency_component(network(a,b,isolated,dead_end))
        self.assertEqual(core, {'a','b'})
        self.assertEqual(audit['excluded_edges'], 2)

    def test_stop_uses_incoming_lane_instead_of_forcing_lane_zero(self):
        a, b = Edge('a'), Edge('b')
        a.connect(b, 0, 1)
        self.assertEqual(arrival_lane(network(a,b), ['a','b']), 1)

    def test_blue_light_stop_prefers_leftmost_of_reachable_lanes(self):
        a, b = Edge('a'), Edge('b')
        a.connect(b, 0, 0); a.connect(b, 1, 1)
        self.assertEqual(arrival_lane(network(a,b), ['a','b']), 1)

    def test_reroute_does_not_assume_last_second_lane_change_on_source_edge(self):
        a, b = Edge('a'), Edge('b')
        a.connect(b, 1, 1)
        self.assertIsNone(arrival_lane(network(a,b), ['a','b'], 'a_0'))
        self.assertEqual(arrival_lane(network(a,b), ['a','b'], 'a_1'), 1)

    def test_alternative_first_turn_is_used_when_shortest_edge_route_is_lane_incompatible(self):
        a, b, c = Edge('a'), Edge('b'), Edge('c')
        a.connect(b, 1, 1); a.connect(c, 0, 0); c.connect(b, 0, 1)
        with patch('traci.simulation.findRoute', return_value=SimpleNamespace(edges=['c','b'], length=120., travelTime=6.)):
            route = lane_constrained_route(network(a,b,c), 'a','b','a_0')
        self.assertEqual(route.edges, ['a','c','b'])

    def test_access_and_patient_sources_are_filtered_without_modifying_model_data(self):
        a,b,c = Edge('a'),Edge('b'),Edge('island')
        a.connect(b);b.connect(a)
        env=SimpleNamespace(net=network(a,b,c),ambulance_edges=['a','island'])
        hospitals=[dict(id='H',type='권역응급의료센터')]
        access, (sources, report)=prepare_access(env,hospitals,[['b','island']])
        self.assertEqual(access,[['b']]);self.assertEqual(sources,['a'])
        self.assertEqual(env.ambulance_edges,['a','island'])
        self.assertEqual(report['hospital_access_counts'],{'H':1})

    def test_no_critical_hospital_access_stops_setup_instead_of_spawning_unreachable_patients(self):
        a,b,c=Edge('a'),Edge('b'),Edge('island');a.connect(b);b.connect(a)
        env=SimpleNamespace(net=network(a,b,c),ambulance_edges=['a'])
        with self.assertRaises(RuntimeError):
            prepare_access(env,[dict(id='H',type='권역응급의료센터')],[['island']])
