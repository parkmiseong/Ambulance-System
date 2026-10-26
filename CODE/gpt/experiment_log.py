# -*- coding: utf-8 -*-
"""학습/평가 실행 이력을 하나의 CSV에 누적 저장한다."""
import csv
import datetime as dt
import json
import os
import uuid
from pathlib import Path

HEADER = ["logged_at", "run_id", "run_type", "record_type", "episode",
          "scenario_id", "scenario_name", "strategy", "status",
          "ambulance_limit", "model_path", "duration_seconds", "metrics_json"]

LOG_PATH = Path("results") / "experiment_history.csv"

def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")

def new_run_id(kind):
    return f"{kind}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

def append_record(run_id, run_type, record_type, *, episode="",
                  scenario_id="", scenario_name="", strategy="", status="",
                  ambulance_limit="", model_path="", duration_seconds="",
                  metrics=None):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 한 번의 append는 한 행만 쓰고, 파일이 이미 있으면 헤더를 다시 쓰지 않는다.
    with LOG_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(HEADER)
        writer.writerow([
            now(), run_id, run_type, record_type, episode, scenario_id,
            scenario_name, strategy, status, ambulance_limit, model_path,
            duration_seconds, json.dumps(metrics or {}, ensure_ascii=False, default=str),
        ])
        f.flush()
        os.fsync(f.fileno())
