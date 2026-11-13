"""Keep ordinary traffic available after the source dataset's first hour."""
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dashboard_traffic import prepare_background


class BackgroundTrafficTests(unittest.TestCase):
    def test_repeated_departures_are_sorted_unique_bounded_and_keep_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = '<routes><vType id="car" vClass="passenger"/><trip id="first" type="car" depart="0" from="a" to="b"/><trip id="last" type="car" depart="3599.5" from="b" to="a"/></routes>'
            (root / "trips.xml").write_text(original)
            (root / "demo.xml").write_text('<routes><vType id="ambulance" vClass="emergency"/><vehicle id="demo" depart="0"/></routes>')
            config = root / "run.sumocfg"
            config.write_text('<configuration><input><route-files value="trips.xml,demo.xml"/></input></configuration>')
            paths, metadata = prepare_background(config, root / "output", 7201)
            generated = ET.parse(paths[-1]).getroot().findall("trip")
            self.assertEqual([float(row.get("depart")) for row in generated], [3600., 7199.5, 7200.])
            self.assertEqual(len({row.get("id") for row in generated}), 3)
            self.assertTrue(all(row.get("type") == "car" for row in generated))
            self.assertEqual((root / "trips.xml").read_text(), original)
            self.assertEqual(metadata["repeated_trips"], 3)
            self.assertEqual(metadata["source_trips"], 2)
            self.assertEqual(len(metadata["sources"][0]["sha256"]), 64)
            paths, metadata = prepare_background(config, root / "short", 3600)
            self.assertEqual(len(paths), 2)
            self.assertEqual(metadata["repeated_trips"], 0)

    def test_repeat_generation_can_be_cancelled_before_reading_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "run.sumocfg"
            config.write_text('<configuration><input><route-files value="missing.xml"/></input></configuration>')
            def stopped():
                raise InterruptedError("stop requested")
            with self.assertRaises(InterruptedError):
                prepare_background(config, root, 7200, stopped)
