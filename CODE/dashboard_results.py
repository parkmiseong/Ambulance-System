"""Patient-level reporting for live and saved dashboard runs; no simulator imports."""
import csv
from functools import lru_cache
import io
import json
import math
from pathlib import Path

STATUSES = {"arrived": "도착", "failed": "이송 실패", "censored": "관측 종료"}
POLICIES = {"dqn": "DQN", "shortest_distance": "최단 거리", "shortest_time": "최단 시간",
            "heuristic": "종합 휴리스틱", "greedy_reward": "예상 보상 그리디"}


def patient_record(row, hospitals, episode, policy):
    """Keep original engine values and attach readable hospital/episode identities."""
    hospital = hospitals.get(row.get("hospital_id"), {})
    parts = row.get("reward_components") or {}
    congestion = (parts["occupancy"] + parts["overcrowding"]
                  if all(isinstance(parts.get(key), (int, float)) for key in ("occupancy", "overcrowding"))
                  else None)
    return {**row, "record_id": f"{episode}:{row['patient_id']}", "episode": int(episode),
            "hospital_congestion_reward": congestion,
            "strategy": policy, "strategy_name": POLICIES.get(policy, policy),
            "hospital_name": row.get("hospital_name") or hospital.get("name") or row.get("hospital_id"),
            "hospital_type": row.get("hospital_type") or hospital.get("type"),
            "status": "censored" if row.get("censored") else ("arrived" if row.get("arrived") else "failed")}


@lru_cache(maxsize=1)
def hospital_catalog(root):
    path = Path(root) / "DATASET" / "서울시 응급실 위치 정보.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    hospitals = {h["hpid"]: dict(id=h["hpid"], name=h.get("dutyname"), type=h.get("dutyemclsname"))
                 for h in data["DATA"] if h.get("hpid")}
    # Older simulation CSVs did not include a manifest; recover known virtual
    # hospital names from actual saved experiments with the same hospital IDs.
    for manifest in (Path(root) / "results" / "web_dashboard").glob("*/experiment.json"):
        try:
            hospitals.update({h["id"]: h for h in json.loads(manifest.read_text(encoding="utf-8"))["hospitals"]})
        except (OSError, ValueError, KeyError, TypeError):
            # An interrupted write in an unrelated run must not break all history.
            continue
    return hospitals


def load_saved_patients(directory, catalog):
    hospitals = dict(catalog)
    for name in ("run.json", "experiment.json"):
        manifest = directory / name
        if manifest.exists():
            try:
                hospitals.update({h["id"]: h for h in json.loads(manifest.read_text(encoding="utf-8"))["hospitals"]})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"저장된 실행의 {name} 파일을 읽을 수 없습니다.") from exc
    records, seen = [], set()
    with (directory / "experiment_history.csv").open(encoding="utf-8-sig", newline="") as stream:
        for line, entry in enumerate(csv.DictReader(stream), 2):
            if entry.get("record_type") != "mission":
                continue
            try:
                row = json.loads(entry["metrics_json"])
                record = patient_record(row, hospitals, int(entry["episode"]), entry.get("strategy", ""))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"저장된 환자 결과의 {line}행을 읽을 수 없습니다.") from exc
            if record["record_id"] not in seen:
                records.append(record)
                seen.add(record["record_id"])
    return records


def parse_filters(query):
    def integer(key, default, low, high):
        raw = query.get(key, [""])[0]
        if raw == "":
            return default
        try:
            value = int(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{key} 값이 올바르지 않습니다.") from exc
        if not low <= value <= high:
            raise ValueError(f"{key} 범위가 올바르지 않습니다.")
        return value
    status = query.get("status", [""])[0]
    if status and status not in STATUSES:
        raise ValueError("결과 상태가 올바르지 않습니다.")
    return dict(episode=integer("episode", None, 1, 100), ktas=integer("ktas", None, 1, 5),
                page=integer("page", 1, 1, 1000000), limit=integer("limit", 15, 1, 100),
                status=status, hospital=query.get("hospital", [""])[0],
                search=query.get("search", [""])[0].strip().casefold()[:200])


def select_patients(records, filters):
    return [row for row in records
            if (filters["episode"] is None or row["episode"] == filters["episode"])
            and (filters["ktas"] is None or row["ktas"] == filters["ktas"])
            and (not filters["hospital"] or row.get("hospital_id") == filters["hospital"])
            and (not filters["status"] or row["status"] == filters["status"])
            and (not filters["search"] or filters["search"] in " ".join(str(row.get(key) or "")
                 for key in ("patient_id", "vehicle_id", "hospital_name", "hospital_id")).casefold())]


def patient_report(records, filters):
    # Summaries describe the chosen episode(s), independent of table filters.
    cohort = [r for r in records if filters["episode"] is None or r["episode"] == filters["episode"]]
    severities = []
    for ktas in range(1, 6):
        rows = [r for r in cohort if r["ktas"] == ktas]
        severities.append(dict(ktas=ktas, total=len(rows), arrived=sum(r["status"] == "arrived" for r in rows),
                               failed=sum(r["status"] == "failed" for r in rows),
                               censored=sum(r["status"] == "censored" for r in rows),
                               golden=sum(bool(r.get("golden_success")) for r in rows),
                               avg_reward=sum(r.get("reward", 0) for r in rows) / len(rows) if rows else None))
    hospitals = {}
    for row in cohort:
        hid = row.get("hospital_id")
        if not hid:
            continue
        hospital = hospitals.setdefault(hid, dict(id=hid, name=row.get("hospital_name") or hid,
                                                  type=row.get("hospital_type"), counts=[0] * 5,
                                                  arrived=0, assigned=0, failed=0, censored=0))
        hospital["assigned"] += 1
        hospital[row["status"]] += 1
        if row["status"] == "arrived":
            hospital["counts"][row["ktas"] - 1] += 1
    selected = select_patients(records, filters)
    pages = max(1, math.ceil(len(selected) / filters["limit"]))
    page = min(filters["page"], pages)
    newest_first = list(reversed(selected))
    start = (page - 1) * filters["limit"]
    options = {r["hospital_id"]: r.get("hospital_name") or r["hospital_id"]
               for r in records if r.get("hospital_id")}
    return dict(rows=newest_first[start:start + filters["limit"]], total=len(selected), page=page,
                pages=pages, limit=filters["limit"], cohort_count=len(cohort), record_count=len(records),
                episodes=sorted({r["episode"] for r in records}),
                strategies=sorted({r.get("strategy_name", "") for r in records if r.get("strategy_name")}),
                hospitals=[dict(id=hid, name=name) for hid, name in sorted(options.items(), key=lambda x: x[1])],
                severities=severities, matrix=sorted(hospitals.values(), key=lambda h: (-h["arrived"], h["name"])))


def patients_csv(records):
    fields = [("episode", "에피소드"), ("patient_id", "환자 ID"), ("pre_ktas", "보고 중증도 Pre-KTAS"),
              ("ktas", "배정 중증도 KTAS"), ("vehicle_id", "구급차 ID"), ("hospital_id", "병원 ID"),
              ("hospital_name", "배정 병원"), ("hospital_type", "병원 유형"), ("status", "이송 결과"),
              ("reason", "종료 사유"), ("strategy_name", "선택 정책"), ("occurrence_time", "환자 발생 시각(초)"),
              ("dispatch_time", "배정 시각(초)"), ("finish_time", "결과 확정 시각(초)"),
              ("wait_seconds", "대기 시간(초)"), ("travel_seconds", "이송 시간(초)"),
              ("total_seconds", "총 소요 시간(초)"), ("golden_success", "연구용 골든타임 내 도착"),
              ("rejection_count", "병원 수용 거절 횟수"), ("rejection_history", "병원 거절 이력(JSON)"),
              ("hospital_admission_mode", "병원 수용 판정 방식"), ("hospital_visits", "실제 병원 방문 이력(JSON)"),
              ("occupancy_at_selection", "선택 시 점유율(0~1)"), ("hospital_congestion_reward", "실제 적용 병원 혼잡도 보상"),
              ("congestion_reward_mode", "병원 혼잡도 보상 방식"),
              ("hospital_suitability", "병원 적합도(0~1)"),
              ("reward", "보상")]
    fields += [(f"reward_{key}", label) for key, label in (
        ("golden", "보상_골든타임"), ("travel", "보상_이동"), ("overtime", "보상_시간초과"),
        ("mismatch", "보상_병원적합도"), ("occupancy", "보상_점유율"),
        ("overcrowding", "보상_과밀"), ("rejection", "보상_거절"), ("route_failure", "보상_이송실패"))]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(label for _, label in fields)
    for record in records:
        row = []
        for key, _ in fields:
            value = record.get(key)
            if key.startswith("reward_"):
                value = record.get("reward_components", {}).get(key[len("reward_"):])
            if key == "status":
                value = STATUSES.get(value, value)
            if key in ("rejection_history", "hospital_visits") and value is not None:
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
                value = "'" + value
            row.append(value)
        writer.writerow(row)
    return output.getvalue().encode("utf-8-sig")
