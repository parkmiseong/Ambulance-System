"""A diagnostic background filter may omit whole vehicles and change nothing else."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from diagnostic_route_input import prepare_diagnostic_route_input, validate_vehicle_subset
from dashboard_traffic import prepare_background


def fixture(directory):
    root = Path(directory) / "source inputs"
    root.mkdir(parents=True)
    original = ET.fromstring('''<routes><vType id="ambulance" vClass="emergency" maxSpeed="35">
      <param key="has.bluelight.device" value="true"/></vType>
      <vehicle id="a" type="ambulance" depart="0.00"><route edges="e1 e2"/></vehicle>
      <vehicle id="b" type="ambulance" depart="0.00"><route edges="e3 e4"/></vehicle>
      <vehicle id="c" type="ambulance" depart="0.00"><route edges="e5 e6"/></vehicle></routes>''')
    route = root / "ambulance_routes.rou.xml"
    ET.ElementTree(original).write(route, encoding="utf-8", xml_declaration=True)
    passenger = root / "osm.passenger.trips.xml"
    passenger.write_text('<routes><trip id="p" depart="100" from="e1" to="e2"/></routes>', encoding="utf-8")
    for file in ("net.xml", "output.add.xml", "poi.add.xml", "view.xml", "weights.xml", "state.xml"):
        (root / file).write_text('<root/>', encoding="utf-8")
    cfg = root / "osm.sumocfg"
    cfg.write_text('''<sumoConfiguration><input><net-file value="net.xml"/>
      <route-files value="osm.passenger.trips.xml,ambulance_routes.rou.xml"/>
      <additional-files value="output.add.xml,poi.add.xml"/><weight-files value="weights.xml"/>
      <load-state value="state.xml"/></input>
      <processing><ignore-route-errors value="true"/></processing>
      <gui_only><gui-settings-file value="view.xml"/></gui_only></sumoConfiguration>''', encoding="utf-8")
    candidate = Path(directory) / "candidate.rou.xml"
    filtered = copy.deepcopy(original)
    filtered.remove(filtered.findall("vehicle")[1])
    ET.ElementTree(filtered).write(candidate, encoding="utf-8", xml_declaration=True)
    return cfg, route, candidate


class DiagnosticRouteInputTests(unittest.TestCase):
    def test_copy_resolves_paths_preserves_sources_and_keeps_shared_passenger_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, original, candidate = fixture(directory)
            before = {path: path.read_bytes() for path in (cfg, original, candidate)}
            metadata = prepare_diagnostic_route_input(cfg, candidate, Path(directory) / "diagnostic")
            self.assertEqual((metadata["kept_vehicle_count"], metadata["dropped_vehicle_count"]), (2, 1))
            self.assertEqual(metadata["kept_vehicle_ids"], ["a", "c"])
            self.assertEqual(metadata["dropped_vehicle_ids"], ["b"])
            copied = ET.parse(metadata["applied_cfg"]).getroot()
            self.assertEqual(copied.find("input/net-file").get("value"), str((cfg.parent / "net.xml").resolve()))
            for tag in ("additional-files", "weight-files", "load-state"):
                self.assertTrue(all(Path(value).is_absolute() for value in copied.find("input/" + tag).get("value").split(",")))
            self.assertEqual(copied.find("gui_only/gui-settings-file").get("value"), str((cfg.parent / "view.xml").resolve()))
            self.assertEqual(copied.find("processing/ignore-route-errors").get("value"), "true")
            applied = Path(metadata["applied_route"])
            self.assertEqual(applied.read_bytes(), candidate.read_bytes())
            self.assertEqual(metadata["applied_route_sha256"], hashlib.sha256(candidate.read_bytes()).hexdigest())
            paths, background = prepare_background(Path(metadata["applied_cfg"]), Path(directory) / "repeat", 4000.)
            self.assertEqual(paths[0], str((cfg.parent / "osm.passenger.trips.xml").resolve()))
            self.assertEqual(paths[1], str(applied))
            self.assertEqual(background["source_trips"], 1)
            self.assertEqual(background["repeated_trips"], 1)
            self.assertEqual([v.get("id") for v in ET.parse(paths[2]).getroot().findall("trip")], ["dashboard_repeat_1_0_p"])
            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_candidate_rejects_additions_reordering_and_changes_to_types_routes_or_attributes(self):
        mutations = {
            "added_id": lambda root: root.findall("vehicle")[0].set("id", "new"),
            "changed_depart": lambda root: root.findall("vehicle")[0].set("depart", "1"),
            "changed_type": lambda root: root.findall("vehicle")[0].set("type", "different"),
            "changed_route": lambda root: root.find("vehicle/route").set("edges", "x y"),
            "changed_vtype": lambda root: root.find("vType").set("maxSpeed", "36"),
            "changed_param": lambda root: root.find("vType/param").set("value", "false"),
            "reordered": self._move_last_vehicle,
            "duplicated": lambda root: root.append(copy.deepcopy(root.findall("vehicle")[0])),
            "extra_definition": lambda root: ET.SubElement(root, "route", id="new", edges="x y"),
            "text": lambda root: setattr(root, "text", "different text"),
        }
        with tempfile.TemporaryDirectory() as directory:
            _, original, candidate = fixture(directory)
            baseline = candidate.read_bytes()
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    root = ET.fromstring(baseline)
                    mutate(root)
                    with self.assertRaises(ValueError):
                        validate_vehicle_subset(original.read_bytes(), ET.tostring(root))
            with self.assertRaisesRegex(ValueError, "strict subset"):
                validate_vehicle_subset(original.read_bytes(), original.read_bytes())

    @staticmethod
    def _move_last_vehicle(root):
        last = root.findall("vehicle")[-1]
        root.remove(last)
        root.insert(1, last)

    def test_missing_or_duplicate_original_ambulance_route_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, candidate = fixture(directory)
            original = cfg.read_bytes()
            for value in ("osm.passenger.trips.xml", "ambulance_routes.rou.xml,ambulance_routes.rou.xml"):
                with self.subTest(value=value):
                    tree = ET.ElementTree(ET.fromstring(original))
                    tree.getroot().find("input/route-files").set("value", value)
                    tree.write(cfg)
                    with self.assertRaisesRegex(ValueError, "Exactly one original"):
                        prepare_diagnostic_route_input(cfg, candidate, Path(directory) / "diagnostic")
                    self.assertFalse((Path(directory) / "diagnostic").exists())

    def test_existing_input_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, candidate = fixture(directory)
            diagnostic = Path(directory) / "diagnostic"
            metadata = prepare_diagnostic_route_input(cfg, candidate, diagnostic)
            saved = Path(metadata["applied_cfg"]).read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_diagnostic_route_input(cfg, candidate, diagnostic)
            self.assertEqual(Path(metadata["applied_cfg"]).read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
