"""Emergency mode boundaries and real SUMO telemetry interpretation."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
import emergency_yield as yielding
from simulation_setup import start_sumo
from web_dashboard import validate_options


class YieldSettingsTests(unittest.TestCase):
    def test_dashboard_defaults_on_and_only_accepts_boolean(self):
        self.assertTrue(validate_options({})["emergency_yield"])
        self.assertFalse(validate_options({"emergency_yield": False})["emergency_yield"])
        for value in ("false", 0, 1, None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_options({"emergency_yield": value})

    def test_native_configuration_is_opt_in_for_legacy_callers(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "empty.sumocfg"
            config.write_text("<configuration/>")
            env = SimpleNamespace(cfg_file=config, update_hospital_coordinates=Mock(), get_hospitals=lambda: [])
            for enabled in (False, True):
                with self.subTest(enabled=enabled), patch("simulation_setup.traci") as native, \
                     patch("simulation_setup.sumolib.checkBinary", return_value="sumo"):
                    if enabled:
                        start_sumo(env, {"scale": .5}, 42, output_dir=temp, emergency_yield=True)
                    else:
                        start_sumo(env, {"scale": .5}, 42, output_dir=temp)
                    command = native.start.call_args.args[0]
                    self.assertEqual("--lateral-resolution" in command, enabled)
                    if enabled:
                        self.assertEqual(float(command[command.index("--lateral-resolution") + 1]), .4)
                        native.vehicletype.setParameter.assert_any_call("ambulance_custom", "has.bluelight.device", "true")
                        native.vehicletype.setParameter.assert_any_call("ambulance_custom", "device.bluelight.reactiondist", "60.0")
                    else:
                        native.vehicletype.setParameter.assert_not_called()

    def test_metadata_discloses_signal_priority(self):
        self.assertTrue(yielding.yield_settings(True)["signal_priority_included"])
        self.assertEqual(yielding.yield_settings(True)["reaction_distance_m"], 60.)
        self.assertFalse(yielding.yield_settings(False)["enabled"])
        self.assertIsNone(yielding.yield_settings(False)["lateral_resolution_m"])


class YieldTelemetryTests(unittest.TestCase):
    def test_only_native_reactions_to_managed_ambulances_are_counted(self):
        tc = yielding.tc
        values = {tc.VAR_POSITION: (1., 2.), tc.VAR_SPEED: 3., tc.VAR_ANGLE: 90.}
        lane = SimpleNamespace(getID=lambda: "road_0", getShape=lambda: [(0., 0.), (100., 0.)], getWidth=lambda: 3.2)
        env = SimpleNamespace(net=SimpleNamespace(getEdge=lambda _: SimpleNamespace(getLanes=lambda: [lane])))
        subscribed = {"expired"}
        with patch("emergency_yield.traci.vehicle") as vehicle:
            vehicle.getContextSubscriptionResults.return_value = {key: values for key in ("ambulance", "car1", "car2")}
            vehicle.getParameter.side_effect = lambda vid, _: "ambulance external" if vid == "car1" else "external"
            vehicle.getRoadID.return_value = "road"
            traffic, lanes = yielding.nearby_traffic(env, ["ambulance"], subscribed, True)
            self.assertEqual(subscribed, {"ambulance"})
            self.assertEqual([car["id"] for car in traffic if car["yielding"]], ["car1"])
            self.assertEqual(traffic[0]["yielding_to"], ["ambulance"])
            self.assertEqual(lanes[0]["width"], 3.2)
            yielding.nearby_traffic(env, ["ambulance"], subscribed, True)
            vehicle.subscribeContext.assert_called_once()
            vehicle.getParameter.reset_mock()
            traffic, _ = yielding.nearby_traffic(env, ["ambulance"], subscribed, False, limit=1)
            self.assertEqual(len(traffic), 1)
            self.assertFalse(traffic[0]["yielding"])
            vehicle.getParameter.assert_not_called()

    def test_no_active_mission_clears_subscriptions_and_traffic(self):
        subscribed = {"old_ambulance"}
        with patch("emergency_yield.traci.vehicle") as vehicle:
            self.assertEqual(yielding.nearby_traffic(None, [], subscribed, True), ([], []))
            self.assertEqual(subscribed, set())
            vehicle.subscribeContext.assert_not_called()


if __name__ == "__main__":
    unittest.main()
