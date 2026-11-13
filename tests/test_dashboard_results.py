"""Patient attribution, complete result retention, cohorts, filters, and exports."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dashboard_results import (hospital_catalog, load_saved_patients, parse_filters, patient_record, patient_report,
                               patients_csv, select_patients)


HOSPITALS = {"H1": dict(name="서울 병원", type="센터"), "H2": dict(name="강남 병원", type="기관")}


def record(patient="p1", ktas=1, hospital="H1", episode=1, status="arrived"):
    return patient_record(dict(patient_id=patient, ktas=ktas, pre_ktas=ktas, hospital_id=hospital,
                               arrived=status == "arrived", censored=status == "censored",
                               golden_success=status == "arrived", reward=-12.5,
                               total_seconds=120, wait_seconds=10, travel_seconds=110,
                               vehicle_id="amb_eval_1" if hospital else None), HOSPITALS, episode, "dqn")


class PatientReportingTests(unittest.TestCase):
    def test_arrival_visits_and_penalty_timing_survive_csv_and_json_reporting(self):
        row = record()
        visit = dict(hospital_id="H2", hospital_name="강남 병원", vehicle_id="amb_eval_1",
                     leg_dispatch_time=2., arrival_time=30., accepted=False, reason="hospital_full")
        row.update(hospital_admission_mode="arrival", hospital_visits=[visit],
                   rejection_history=[dict(time=30., stage="after_arrival", vehicle_id="amb_eval_1", penalty=-12.)])
        exported = next(csv.DictReader(io.StringIO(patients_csv([row]).decode("utf-8-sig"))))
        self.assertEqual(exported["병원 수용 판정 방식"], "arrival")
        self.assertEqual(json.loads(exported["실제 병원 방문 이력(JSON)"]), [visit])
        self.assertEqual(json.loads(exported["병원 거절 이력(JSON)"])[0]["stage"], "after_arrival")

    def test_rejection_count_and_exact_history_are_retained_in_csv(self):
        row = record()
        row.update(rejection_count=1, rejection_history=[dict(hospital_id="H2",hospital_name="강남 병원",time=3.,reason="hospital_full",penalty=-12.)],
                   reward_components={"rejection":-12.})
        csv_row = next(csv.DictReader(io.StringIO(patients_csv([row]).decode("utf-8-sig"))))
        self.assertEqual(csv_row["병원 수용 거절 횟수"], "1")
        self.assertEqual(json.loads(csv_row["병원 거절 이력(JSON)"]), row["rejection_history"])
        self.assertEqual(csv_row["보상_거절"], "-12.0")

    def test_same_patient_id_in_two_episodes_is_distinct(self):
        one, two = record(episode=1), record(episode=2)
        self.assertNotEqual(one["record_id"], two["record_id"])
        self.assertEqual(one["hospital_name"], "서울 병원")
        self.assertEqual(one["vehicle_id"], "amb_eval_1")

    def test_all_records_are_retained_and_pagination_has_no_latest_12_limit(self):
        records = [record(patient=f"p{i}") for i in range(37)]
        report = patient_report(records, parse_filters({"page": ["3"]}))
        self.assertEqual((report["total"], report["pages"], len(report["rows"])), (37, 3, 7))
        self.assertEqual(report["rows"][-1]["patient_id"], "p0")
        self.assertEqual(sum(s["total"] for s in report["severities"]), 37)

    def test_matrix_counts_actual_arrivals_and_keeps_unassigned_failures_in_severity_totals(self):
        records = [record(), record("p2", ktas=2, status="failed"),
                   record("p3", ktas=3, status="censored"),
                   record("p4", ktas=4, hospital=None, status="failed")]
        report = patient_report(records, parse_filters({}))
        self.assertEqual(report["matrix"][0]["counts"], [1, 0, 0, 0, 0])
        self.assertEqual(report["matrix"][0]["assigned"], 3)
        self.assertEqual(sum(s["total"] for s in report["severities"]), 4)
        self.assertEqual(sum(s["censored"] for s in report["severities"]), 1)

    def test_filters_match_the_table_without_changing_the_episode_summary(self):
        records = [record(), record("p2", ktas=2), record("p3", ktas=2, episode=2),
                   record("p4", ktas=2, hospital="H2")]
        filters = parse_filters(dict(episode=["1"], ktas=["2"], hospital=["H1"], status=["arrived"], search=["서울"]))
        report = patient_report(records, filters)
        self.assertEqual([r["patient_id"] for r in report["rows"]], ["p2"])
        self.assertEqual(report["cohort_count"], 3)
        self.assertEqual(len(select_patients(records, filters)), 1)

    def test_bad_filters_are_rejected(self):
        for query in (dict(ktas=["6"]), dict(page=["0"]), dict(limit=["101"]),
                      dict(episode=["nan"]), dict(status=["survived"])):
            with self.subTest(query=query), self.assertRaises(ValueError):
                parse_filters(query)

    def test_csv_exports_actual_hospital_names_and_keeps_negative_numeric_rewards(self):
        data = patients_csv([record(patient="=SUM(A1)")])
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        self.assertEqual(rows[0]["배정 병원"], "서울 병원")
        self.assertEqual(rows[0]["이송 결과"], "도착")
        self.assertEqual(rows[0]["보상"], "-12.5")
        self.assertTrue(rows[0]["환자 ID"].startswith("'="))

    def test_older_saved_logs_use_manifest_names_without_inventing_missing_vehicle_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "experiment.json").write_text(json.dumps(dict(hospitals=[dict(id="V", name="가상 병원", type="센터")])), encoding="utf-8")
            with (folder / "experiment_history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["record_type", "episode", "strategy", "metrics_json"])
                writer.writeheader()
                writer.writerow(dict(record_type="mission", episode=2, strategy="dqn", metrics_json=json.dumps(
                    dict(patient_id="p1", ktas=1, hospital_id="V", arrived=True, censored=False, reward=3))))
            records = load_saved_patients(folder, HOSPITALS)
            self.assertEqual(records[0]["hospital_name"], "가상 병원")
            self.assertEqual(records[0]["record_id"], "2:p1")
            self.assertNotIn("vehicle_id", records[0])

    def test_a_broken_unrelated_manifest_does_not_break_all_saved_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "DATASET").mkdir()
            (root / "DATASET" / "서울시 응급실 위치 정보.json").write_text(
                json.dumps({"DATA": [{"hpid": "real", "dutyname": "Hospital"}]}), encoding="utf-8")
            broken = root / "results/web_dashboard/broken"
            broken.mkdir(parents=True)
            (broken / "experiment.json").write_text('{"hospitals":')
            valid = root / "results/web_dashboard/valid"
            valid.mkdir()
            (valid / "experiment.json").write_text(json.dumps({"hospitals": [{"id": "virtual", "name": "Virtual"}]}))
            self.assertEqual(set(hospital_catalog(str(root))), {"real", "virtual"})
            with self.assertRaisesRegex(ValueError, "experiment.json"):
                load_saved_patients(broken, {})
