"""Lane probes stay opt-in, bounded and observational during crash replay."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
import traci
from sumo_lane_snapshot import LaneSnapshotProbe


def fake_native():
    lanes = {"amb_e_5": "a_0", "amb_static": "b_0", "car_internal": ":j_0_0", "car_normal": "c_0"}
    types = {"amb_e_5": "ambulance_custom", "amb_static": "ambulance",
             "car_internal": "DEFAULT_VEHTYPE", "car_normal": "DEFAULT_VEHTYPE"}
    vehicle = SimpleNamespace(
        getIDList=Mock(side_effect=lambda: list(lanes)),
        getLaneID=Mock(side_effect=lanes.__getitem__), getTypeID=Mock(side_effect=types.__getitem__),
        getBestLanes=Mock(side_effect=lambda vid: [(lanes[vid], 100., 0., 0, True, ())]),
        getRouteIndex=Mock(return_value=1), getRouteID=Mock(return_value="route"),
        getAllowedSpeed=Mock(return_value=13.9), getSpeed=Mock(return_value=6.),
        getLanePosition=Mock(return_value=14.), getNextLinks=Mock(return_value=()),
        getParameter=Mock(return_value=""))
    return SimpleNamespace(vehicle=vehicle, lane=SimpleNamespace(getLinks=Mock(return_value=())),
                           simulation=SimpleNamespace(saveState=Mock())), lanes, types


class LaneSnapshotTests(unittest.TestCase):
    def test_default_has_no_queries_files_or_native_state_export(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Mock()
            probe = LaneSnapshotProbe(directory, native=native)
            for time in (0., 1100., 1140., 4000.):
                probe.observe(time)
            self.assertEqual(native.mock_calls, [])
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertFalse(probe.settings()["enabled"])

    def test_window_filters_normal_cars_caches_types_and_does_not_duplicate_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = fake_native()
            probe = LaneSnapshotProbe(directory, 1100., 1140., native=native)
            for time in (1099., 1100., 1100., 1128., 1140., 1141.):
                probe.observe(time, ("amb_e_5",))
            rows = [json.loads(line) for line in (Path(directory) / "lane_snapshots.jsonl").read_text().splitlines()]
            self.assertEqual([row["time"] for row in rows], [1100., 1128., 1140.])
            self.assertEqual(native.vehicle.getIDList.call_count, 3)
            self.assertEqual(native.vehicle.getLaneID.call_count, 12)
            self.assertEqual(native.vehicle.getTypeID.call_count, 4)
            self.assertEqual({row["vehicle_id"] for row in rows[0]["vehicles"]},
                             {"amb_e_5", "amb_static", "car_internal"})
            self.assertEqual({row["category"] for row in rows[0]["vehicles"]},
                             {"managed_ambulance", "background_ambulance", "internal_lane"})
            self.assertNotIn("car_normal", [call.args[0] for call in native.vehicle.getBestLanes.call_args_list])
            native.simulation.saveState.assert_not_called()

    def test_internal_lane_mismatch_is_not_flagged_as_invalid_and_other_flags_are_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = fake_native()
            native.vehicle.getBestLanes.side_effect = lambda vid: () if vid == "amb_e_5" else (("different_0", 1., 0., 0, False, ()),)
            probe = LaneSnapshotProbe(directory, 1., 1., native=native)
            probe.observe(1.)
            row = json.loads((Path(directory) / "lane_snapshots.jsonl").read_text())
            by_id = {v["vehicle_id"]: v for v in row["vehicles"]}
            self.assertEqual(by_id["car_internal"]["observation_flags"], [])
            self.assertEqual(by_id["amb_e_5"]["observation_flags"], ["empty_best_lanes"])
            self.assertEqual(by_id["amb_static"]["observation_flags"], ["current_normal_lane_absent_from_best_lanes"])
            self.assertTrue(row["flags_are_observations_only"])
            self.assertNotIn("invalid", row)

    def test_save_state_is_once_at_the_first_observed_tick_inside_the_window(self):
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = fake_native()
            def save(path):
                Path(path).write_bytes(b"native-state")
            native.simulation.saveState.side_effect = save
            probe = LaneSnapshotProbe(directory, 1100., 1140., native=native, save_native_state=True)
            for time in (1099., 1101., 1102., 1141.):
                probe.observe(time)
            native.simulation.saveState.assert_called_once()
            path = Path(native.simulation.saveState.call_args.args[0])
            self.assertEqual(path.parent, Path(directory))
            self.assertEqual(probe.statistics()["native_state_path"], str(path))

    def test_save_state_failure_is_nonblocking_and_attempted_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = fake_native()
            native.simulation.saveState.side_effect = OSError("state export failed")
            errors = []
            probe = LaneSnapshotProbe(directory, 1., 2., native=native, save_native_state=True, logging_errors=errors)
            probe.observe(1.)
            probe.observe(2.)
            native.simulation.saveState.assert_called_once()
            self.assertEqual(probe.rows_written, 2)
            self.assertTrue(any("state export failed" in value for value in errors))

    def test_json_and_write_failures_do_not_interrupt_observation(self):
        for error in (TypeError("JSON cannot encode value"), PermissionError("locked JSONL")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                native, _, _ = fake_native()
                probe = LaneSnapshotProbe(directory, 1., 2., native=native)
                if isinstance(error, TypeError):
                    scope = patch("sumo_lane_snapshot.json.dumps", side_effect=error)
                else:
                    scope = patch.object(Path, "open", side_effect=error)
                with scope:
                    probe.observe(1.)
                self.assertEqual(probe.rows_written, 0)
                self.assertEqual(probe.statistics()["logging_error_count"], 1)
                self.assertIn(str(error), probe.logging_errors[0])
                probe.observe(2.)
                self.assertEqual(probe.rows_written, 1)

    def test_recoverable_getter_failure_is_logged_but_native_disconnect_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = fake_native()
            native.vehicle.getBestLanes.side_effect = traci.TraCIException("unsupported observation")
            probe = LaneSnapshotProbe(directory, 1., 2., native=native)
            probe.observe(1.)
            self.assertEqual(probe.rows_written, 1)
            self.assertIn("unsupported observation", probe.logging_errors[0])
            native.vehicle.getIDList.side_effect = traci.exceptions.FatalTraCIError("Connection closed by SUMO.")
            with self.assertRaises(traci.exceptions.FatalTraCIError):
                probe.observe(2.)
            self.assertEqual(probe.rows_written, 1)

    def test_detail_limit_prioritizes_ambulances_and_records_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = fake_native()
            probe = LaneSnapshotProbe(directory, 1., 1., native=native)
            probe.MAX_DETAILS = 2
            probe.observe(1.)
            row = json.loads((Path(directory) / "lane_snapshots.jsonl").read_text())
            self.assertEqual(row["candidate_count"], 3)
            self.assertEqual(row["truncated_vehicle_count"], 1)
            self.assertEqual([v["vehicle_id"] for v in row["vehicles"]], ["amb_e_5", "amb_static"])

    def test_invalid_window_is_rejected_before_any_observation(self):
        for start, end, save in ((None, 1., False), (1., None, False), (2., 1., False),
                                 (-1., 1., False), (1., float("nan"), False), (None, None, True)):
            with self.subTest(start=start, end=end, save=save), self.assertRaises(ValueError):
                LaneSnapshotProbe("unused", start, end, save_native_state=save)


if __name__ == "__main__":
    unittest.main()
