"""Local dashboard for the project's real SUMO/DQN engine (stdlib HTTP server)."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import secrets
import shutil
import threading
import time
import traceback
from urllib.parse import parse_qs, urlparse
import webbrowser

from dashboard_results import (hospital_catalog, load_saved_patients, parse_filters, patient_record,
                               patient_report, patients_csv, select_patients)
from dashboard_demand import DEFAULT_DURATION, PATIENTS_PER_HOUR, dashboard_scenario, demand_metadata
from dashboard_environment import SHARED_ARCHITECTURES
from virtual_hospital_mix import apply_hospital_mix, hospital_plan, mix_controls
from reward import WEIGHTS, hospital_congestion_components, congestion_reward_profile

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web"
OUTPUT = ROOT / "results" / "web_dashboard"
LEVELS = {"traffic": (.3, .5, .8), "demand": PATIENTS_PER_HOUR,
          "occupancy": ((.15, .30), (.35, .50), (.60, .75))}
ACTIVE = {"preparing", "running", "paused", "stopping"}


def ensure_sumo():
    if not os.environ.get("SUMO_HOME"):
        binary = shutil.which("sumo")
        if binary:
            os.environ["SUMO_HOME"] = str(Path(binary).resolve().parents[1])


def hospital_rows(hospitals, congestion_reward_mode="balanced"):
    rows = []
    for hospital in hospitals:
        row = {k: hospital.get(k) for k in ("id", "name", "type", "sumo_x", "sumo_y", "occupancy",
                                           "virtual", "reserved_count", "admitted_count", "can_admit")}
        occupancy = row["occupancy"]
        row["congestion_reward_preview"] = (sum(hospital_congestion_components(occupancy, mode=congestion_reward_mode).values())
                                            if occupancy is not None and math.isfinite(occupancy) else None)
        rows.append(row)
    return rows


def network_data():
    from environment import SumoMedicalEnvironment
    env = SumoMedicalEnvironment()
    roads, labels, names = [], [], set()
    for edge in env.net.getEdges():
        if not edge.allows("emergency"):
            continue
        shape = [[round(x, 1), round(y, 1)] for x, y in edge.getShape()]
        if len(shape) < 2:
            continue
        major = edge.getLaneNumber() >= 3 or edge.getSpeed() >= 19
        roads.append([1 if major else 0, shape])
        name = edge.getName()
        if name and name not in names and major and edge.getLength() > 200:
            names.add(name)
            labels.append([name, *shape[len(shape) // 2]])
    for h in env.hospitals:
        h["sumo_x"], h["sumo_y"] = env.net.convertLonLat2XY(h["lon"], h["lat"])
        h["occupancy"] = None  # No simulated values before a run starts.
    districts = []
    for name, lon, lat in [("반포동", 127.005, 37.505), ("논현동", 127.027, 37.513),
                            ("역삼동", 127.040, 37.500), ("삼성동", 127.060, 37.514),
                            ("서초동", 127.015, 37.484), ("대치동", 127.062, 37.493),
                            ("압구정동", 127.030, 37.530)]:
        x, y = env.net.convertLonLat2XY(lon, lat)
        districts.append([name, x, y])
    import xml.etree.ElementTree as ET
    centers = []
    boundary = env.net.getBoundary()
    for poi in ET.parse(env.cfg_file.parent / "fire_stations.add.xml").getroot().iter("poi"):
        if poi.get("type") != "fire_station":
            continue
        x, y = env.net.convertLonLat2XY(float(poi.get("lon")), float(poi.get("lat")))
        if boundary[0] <= x <= boundary[2] and boundary[1] <= y <= boundary[3]:
            centers.append(dict(id=poi.get("id"), name=poi.get("name"), x=x, y=y))
    return {"bounds": boundary, "roads": roads, "labels": labels[:55],
            "districts": districts, "hospitals": hospital_rows(env.hospitals), "centers": centers}


def available_models():
    # Only best models are offered, in keeping with the project's evaluation policy.
    roots = [ROOT / "results" / "models"]
    roots += [p / "models" for p in (ROOT / "results").iterdir()
              if p.is_dir() and not p.name.startswith(("backup", "verification"))] if (ROOT / "results").exists() else []
    roots += list(OUTPUT.glob("*/models")) if OUTPUT.exists() else []
    paths = {p.resolve() for folder in roots for p in folder.glob("dqn_v*_best.pth")}
    return sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True)


def model_manifest(model_path):
    """Read display metadata without loading Torch or reconstructing the network."""
    path = Path(model_path)
    tag = path.stem.removeprefix("dqn_").removesuffix("_best")
    candidates = [path.parent.parent / "experiment.json",
                  path.parent.parent / "training" / f"experiment_{tag}.json"]
    for candidate in candidates:
        try:
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
            if tag != f"v{manifest['version']}_{manifest['signature']}":
                continue
            if not isinstance(manifest.get("hospitals"), list):
                continue
            return manifest
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def model_hospital_counts(model_path):
    manifest = model_manifest(model_path)
    if manifest is None:
        return {"hospital_count": None, "virtual_count": None}
    return {"hospital_count": len(manifest["hospitals"]),
            "virtual_count": sum(bool(h.get("virtual")) for h in manifest["hospitals"])}


def dashboard_configuration(dashboard):
    models = available_models()
    return {"token": dashboard.token, "demand_profile": demand_metadata(),
            "hospital_admission": {"enabled_by_default": True, "rejection_penalty": -WEIGHTS.rejection,
                                   "stage": "after_arrival", "pre_departure_check": True,
                                   "reserve_during_transport": False},
            "hospital_congestion": congestion_reward_profile("balanced"),
            "hospital_controls": {"simulation_editable": True, "virtual_mix": mix_controls()},
            "models": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                        "name": p.name, "group": p.parent.parent.name,
                        "hospital_resize_supported": (model_manifest(p) or {}).get("agent_architecture") in SHARED_ARCHITECTURES,
                        "virtual_hospital_types": [h.get("type") for h in (model_manifest(p) or {}).get("hospitals", []) if h.get("virtual")],
                        **model_hospital_counts(p)} for p in models]}


def validate_options(body):
    if not isinstance(body, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")
    options = {}
    for key, default, low, high in [("traffic", 1, 0, 2), ("demand", 1, 0, 2),
                                    ("occupancy", 1, 0, 2), ("episodes", 3, 1, 100),
                                    ("duration", DEFAULT_DURATION, 30, 86400), ("ambulances", 10, 1, 10),
                                    ("virtual_count", 2, 0, 10),
                                    ("seed", 42, 0, 2**31 - 100001)]:
        value = body.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key}: {low}~{high} 범위의 정수를 입력하세요.")
        options[key] = value
    options["virtual_mix"] = hospital_plan(options["virtual_count"], body.get("virtual_mix"))["weights"]
    options["mode"] = body.get("mode", "simulation")
    options["policy"] = body.get("policy", "dqn")
    options["demand_mode"] = body.get("demand_mode", "preset")
    if options["demand_mode"] not in ("preset", "rate", "count"):
        raise ValueError("환자 발생 방식은 강남 기준, 시간당 직접 입력, 총 환자 수 중 선택하세요.")
    if options["demand_mode"] == "rate":
        rate = body.get("custom_patients_per_hour")
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 3600:
            raise ValueError("시간당 환자 수는 0~3600 사이의 숫자를 입력하세요.")
        options["custom_patients_per_hour"] = rate
    elif options["demand_mode"] == "count":
        count = body.get("patient_count")
        if type(count) is not int or not 0 <= count <= options["duration"]:
            raise ValueError(f"총 환자 수는 0~{options['duration']} 사이의 정수를 입력하세요 (초당 최대 1명).")
        options["patient_count"] = count
    options["emergency_yield"] = body.get("emergency_yield", True)
    if type(options["emergency_yield"]) is not bool:
        raise ValueError("구급차 양보 설정은 켜기/끄기로 지정하세요.")
    options["hospital_rejection"] = body.get("hospital_rejection", True)
    if type(options["hospital_rejection"]) is not bool:
        raise ValueError("병원 수용 요청·거절 패널티는 켜기/끄기로 지정하세요.")
    if options["mode"] not in ("simulation", "training"):
        raise ValueError("실행 모드가 올바르지 않습니다.")
    if options["policy"] not in ("dqn", "shortest_distance", "shortest_time", "heuristic", "greedy_reward"):
        raise ValueError("병원 선택 정책이 올바르지 않습니다.")
    return options


class RunStopped(Exception):
    pass


def run_worker(options, model_path, events, stop, paused, speed):
    """Own TraCI and all random state in an isolated process."""
    ensure_sumo()
    os.environ["PYTHONIOENCODING"] = "utf-8"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    directory = OUTPUT / run_id
    directory.mkdir(parents=True, exist_ok=True)

    def emit(kind, **data):
        message = {"kind": kind, **data}
        if kind == "frame":
            try:
                events.put_nowait(message)
            except queue.Full:
                pass
        else:
            events.put(message, timeout=5)

    def checkpoint():
        if stop.is_set():
            raise RunStopped()
        while paused.is_set():
            if stop.wait(.05):
                raise RunStopped()

    try:
        from dataclasses import replace
        import torch
        import traci
        from baseline import state_dimension
        from dqn_agent import DQNAmbulanceAgent
        from experiment_config import (SimulationConfig, LearningConfig, SCENARIOS,
                                       network_architecture, isolated_random_state)
        from experiment_log import append_record
        from simulation import run_episode
        from emergency_yield import nearby_traffic, yield_settings
        from dashboard_traffic import prepare_background
        from dashboard_environment import resize_inference_agent
        from simulation_setup import prepare_environment, make_manifest, manifest_signature, load_experiment

        torch.set_num_threads(2)
        config = SimulationConfig(arrival_steps=options["duration"], seed=options["seed"],
                                  congestion_reward_mode="balanced",
                                  admission_mode="arrival" if options["hospital_rejection"] else "precheck")
        scenario = dashboard_scenario(options, SCENARIOS)
        options["patients_per_hour"] = scenario["patients_per_hour"]
        options["demand_profile_id"] = scenario["demand_profile"]["profile_id"]
        occupancy = LEVELS["occupancy"][options["occupancy"]]
        training = options["mode"] == "training"
        policy = "dqn" if training else options["policy"]
        emit("status", status="preparing", message="도로망과 병원 접근 경로를 준비하고 있습니다.",
             run_id=run_id, output=str(directory.relative_to(ROOT)))
        checkpoint()
        hospital_transfer = None
        if policy == "dqn" and not training:
            if not model_path:
                raise ValueError("사용 가능한 best 모델이 없습니다. 먼저 학습하거나 다른 정책을 선택하세요.")
            env, hospitals, agent, original_config, manifest = load_experiment(model_path)
            config = replace(original_config, arrival_steps=options["duration"],
                             demand_scale=1., seed=options["seed"],
                             congestion_reward_mode="balanced",
                             admission_mode="arrival" if options["hospital_rejection"] else "precheck")
            checkpoint()
            saved_count = sum(bool(h.get("virtual")) for h in hospitals)
            saved_types = {h["id"]: h.get("type") for h in hospitals if h.get("virtual")}
            saved_ids = [h["id"] for h in hospitals]
            if options["virtual_count"] != saved_count:
                if agent.architecture not in SHARED_ARCHITECTURES:
                    raise ValueError("이전 v3 모델은 병원 수가 고정되어 있습니다. v4 이상 모델이나 다른 병원 선택 정책을 사용하세요.")
                hospital_transfer = dict(method="shared_scorer_weights_unchanged", trained_virtual_count=saved_count,
                    trained_hospital_count=len(hospitals), requested_virtual_count=options["virtual_count"],
                    source_signature=manifest.get("signature"), original_hospital_ids=[h["id"] for h in hospitals])
                emit("status", status="preparing", message="요청한 가상 병원 수로 도로망과 병원 접근 경로를 다시 준비하고 있습니다.")
                checkpoint()
                env, hospitals, edges, access = prepare_environment(
                    config, virtual_count=options["virtual_count"], output_dir=directory / "setup", quiet=True,
                    emergency_yield=options["emergency_yield"], virtual_mix=options["virtual_mix"])
                agent = resize_inference_agent(agent, len(hospitals), config)
        else:
            env, hospitals, edges, access = prepare_environment(
                config, virtual_count=options["virtual_count"], output_dir=directory / "setup", quiet=True,
                emergency_yield=options["emergency_yield"], virtual_mix=options["virtual_mix"])
            agent = None
        virtual_mix_plan = apply_hospital_mix(hospitals, options["virtual_mix"])
        env.hospitals = copy.deepcopy(hospitals)
        if policy == "dqn" and not training:
            types_changed = any(saved_types.get(h["id"]) != h.get("type") for h in hospitals if h.get("virtual"))
            if types_changed and hospital_transfer is None:
                hospital_transfer = dict(method="shared_scorer_weights_unchanged" if agent.architecture in SHARED_ARCHITECTURES else "same_slots_type_features_changed",
                    trained_virtual_count=saved_count, trained_hospital_count=len(saved_ids),
                    requested_virtual_count=options["virtual_count"], source_signature=manifest.get("signature"),
                    original_hospital_ids=saved_ids)
            if hospital_transfer is not None:
                hospital_transfer.update(hospital_types_changed=types_changed, original_virtual_types=saved_types)
        options["virtual_count"] = sum(bool(h.get("virtual")) for h in hospitals)
        checkpoint()
        route_files, background = prepare_background(env.cfg_file, directory, config.drain_deadline, checkpoint)
        scenario["background_traffic"] = background
        if training:
            learning = LearningConfig()
            manifest = make_manifest(env, hospitals, edges, access, config,
                                     options["ambulances"], "severity", 300, (), learning)
            manifest["validation_seeds"] = [options["seed"] + 50000]
            manifest["web_scenario"] = {**scenario, "initial_occupancy_range": list(occupancy)}
            manifest["web_emergency_yield"] = yield_settings(options["emergency_yield"])
            manifest["web_virtual_hospital_mix"] = virtual_mix_plan
            manifest["signature"] = manifest_signature(manifest)
            manifest_path = directory / "experiment.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            agent = DQNAmbulanceAgent(state_dimension(len(hospitals), config.observation_version),
                                      len(hospitals), architecture=network_architecture(config),
                                      learning_config=learning)
            agent.metadata = {"manifest_path": str(manifest_path.resolve()),
                              "experiment_signature": manifest["signature"],
                              "selection_metric": "validation_reward_per_generated_patient"}
        # Runtime input paths do not enter portable experiment manifests.
        scenario["_route_files"] = route_files
        public_scenario = {key: value for key, value in scenario.items() if not key.startswith("_")}
        # The source checkpoint stays unchanged; changed sets are recorded for evaluation.
        checkpoint()
        hospital_lookup = {h["id"]: h for h in hospitals}
        (directory / "run.json").write_text(json.dumps({"run_id": run_id, "options": options,
            "hospitals": hospital_rows(hospitals), "scenario": public_scenario,
            "hospital_transfer": hospital_transfer,
            "virtual_hospital_mix": virtual_mix_plan,
            "hospital_congestion": congestion_reward_profile(config.congestion_reward_mode),
            "hospital_admission": {"mode": config.admission_mode, "rejection_penalty": -WEIGHTS.rejection,
                                   "stage": "after_arrival" if config.admission_mode == "arrival" else "before_departure",
                                   "pre_departure_check": True,
                                   "reserve_during_transport": config.admission_mode != "arrival"},
            "emergency_yield": yield_settings(options["emergency_yield"])}, ensure_ascii=False, indent=2), encoding="utf-8")
        emit("ready", hospitals=hospital_rows(hospitals), policy=policy,
             model=Path(model_path).name if model_path and not training else None,
             scenario=public_scenario, occupancy_range=occupancy,
             patients_per_hour=scenario["patients_per_hour"],
             expected_patients=scenario.get("fixed_patient_count", scenario["patients_per_hour"] * options["duration"] / 3600.),
             requested_patient_count=scenario.get("fixed_patient_count"),
             emergency_yield=yield_settings(options["emergency_yield"]),
             demand_scale=config.demand_scale, observation_version=config.observation_version,
             reward_version=config.reward_version, virtual_count=options["virtual_count"],
             hospital_admission_mode=config.admission_mode,
             congestion_reward_mode=config.congestion_reward_mode,
             hospital_count=len(hospitals), hospital_transfer=hospital_transfer,
             virtual_hospital_mix=virtual_mix_plan, options=dict(options))
        count = options["episodes"] if training else 1
        best_score = None
        for episode in range(1, count + 1):
            checkpoint()
            emit("episode_start", episode=episode, total_episodes=count)
            last_frame, last_tick = 0., time.monotonic()
            episode_patients = []
            subscribed = set()

            def on_step(snapshot):
                nonlocal last_frame, last_tick
                checkpoint()
                # Pace SUMO's own clock; positions and rewards remain engine outputs.
                interval = 1. / max(1., speed.value)
                remaining = interval - (time.monotonic() - last_tick)
                if remaining > 0 and stop.wait(remaining):
                    raise RunStopped()
                checkpoint()
                last_tick = time.monotonic()
                if last_tick - last_frame < .09:
                    return
                last_frame = last_tick
                manager, rows = snapshot["manager"], snapshot["outcomes"]
                vehicles = []
                visible = set(traci.vehicle.getIDList())
                for vid, mission in list(manager.active.items()):
                    if vid not in visible:
                        continue
                    x, y = traci.vehicle.getPosition(vid)
                    selected = mission.payload[1]
                    vehicles.append({"id": vid, "x": x, "y": y,
                                     "angle": traci.vehicle.getAngle(vid),
                                     "speed": traci.vehicle.getSpeed(vid),
                                     "hospital_id": selected["id"], "hospital_name": selected.get("name"),
                                     "patient_id": mission.patient["patient_id"], "ktas": mission.patient["ktas"],
                                     "dispatch_time": mission.dispatch_time,
                                     "elapsed_seconds": snapshot["time"] - mission.dispatch_time,
                                     "rejection_count": len(mission.patient.get("rejection_history", [])),
                                     "transport_stage": ("awaiting_hospital" if selected.get("admission_checked")
                                                         else "redirecting" if mission.patient.get("rejection_history")
                                                         else "transporting")})
                traffic, lanes = nearby_traffic(env, [v["id"] for v in vehicles], subscribed,
                                                options["emergency_yield"])
                patients = []
                for patient in list(manager.waiting)[:150]:
                    x, y = env.net.getEdge(patient["edge_id"]).getFromNode().getCoord()[:2]
                    patients.append({"id": patient["patient_id"], "x": x, "y": y, "ktas": patient["ktas"]})
                components = {}
                for row in rows:
                    for key, value in row["reward_components"].items():
                        components[key] = components.get(key, 0.) + value
                settled_reward = sum(row["reward"] for row in rows)
                pending_penalty = snapshot.get("rejection_penalty", 0.) - components.get("rejection", 0.)
                components["rejection"] = components.get("rejection", 0.) + pending_penalty
                emit("frame", time=snapshot["time"], vehicles=vehicles, patients=patients,
                     traffic=traffic, lanes=lanes, yielding_count=sum(v["yielding"] for v in traffic),
                     hospitals=hospital_rows(snapshot["hospitals"]),
                     active=len(manager.active), waiting=len(manager.waiting), completed=len(rows),
                     arrived=sum(r["arrived"] for r in rows),
                     failed=sum(not r["arrived"] and not r["censored"] for r in rows),
                     generated=len(rows) + len(manager.active) + len(manager.waiting),
                     reward=settled_reward + pending_penalty, completed_reward=settled_reward,
                     rejection_count=snapshot.get("rejection_count", 0), reward_components=components,
                     golden_success=sum(r["golden_success"] for r in rows),
                     recent=rows[-12:], updates=getattr(agent, "training_updates", 0),
                     epsilon=getattr(agent, "epsilon", None))

            def on_outcome(row):
                record = patient_record(row, hospital_lookup, episode, policy)
                episode_patients.append(record)
                append_record(run_id, options["mode"], "mission", episode=episode,
                              scenario_id=scenario["id"], strategy=policy, metrics=record,
                              log_path=directory / "experiment_history.csv")
                emit("outcome", row=record)

            def on_rejection(row):
                record = {**row, "episode": episode,
                          "record_id": f"{episode}:{row['patient_id']}:{row['rejection_number']}"}
                append_record(run_id, options["mode"], "hospital_rejection", episode=episode,
                              scenario_id=scenario["id"], strategy=policy, metrics=record,
                              log_path=directory / "experiment_history.csv")
                emit("hospital_rejection", row=record)

            emit("status", status="running", message="학습 중" if training else "시뮬레이션 중")
            episode_seed = options["seed"] + 10000 + episode if training else options["seed"]
            result = run_episode(env, hospitals, scenario, episode_seed,
                                 policy, config, agent=agent, training=training,
                                 ambulance_count=options["ambulances"], output_dir=directory / f"episode{episode}",
                                 on_step=on_step, on_outcome=on_outcome, quiet=True,
                                 on_rejection=on_rejection,
                                 initial_occupancy_range=occupancy, emergency_yield=options["emergency_yield"])
            result["hospital_transfer"] = hospital_transfer
            result["virtual_hospital_mix"] = virtual_mix_plan
            checkpoint()
            (directory / f"patients_episode{episode}.json").write_text(
                json.dumps(episode_patients, ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / f"episode{episode}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            emit("episode_result", episode=episode, result=result)
            if training:
                agent.save_model(str(directory / "models" / f"dqn_v6_{manifest['signature']}_last.pth"),
                                 episode=episode)
                agent.save_model(str(directory / "checkpoint.pth"), episode=episode, full_state=True)
                emit("status", status="running", message=f"에피소드 {episode} 완료 · 별도 시드로 모델 검증 중")
                with isolated_random_state():
                    epsilon = agent.epsilon
                    agent.epsilon = 0.
                    agent.model.eval()
                    try:
                        validation = run_episode(
                            env, hospitals, scenario, options["seed"] + 50000, "dqn", config, agent=agent,
                            ambulance_count=options["ambulances"],
                            output_dir=directory / f"validation{episode}", quiet=True,
                            initial_occupancy_range=occupancy, on_step=lambda _: checkpoint(),
                            emergency_yield=options["emergency_yield"])
                    finally:
                        agent.epsilon = epsilon
                        agent.model.train()
                if agent.training_updates and validation["generated_patients"] and (best_score is None or validation["avg_reward"] > best_score):
                    best_score = validation["avg_reward"]
                    agent.metadata["best_validation_reward"] = best_score
                    agent.save_model(str(directory / "models" / f"dqn_v6_{manifest['signature']}_best.pth"),
                                     episode=episode)
                result["validation_avg_reward"] = validation["avg_reward"] if validation["generated_patients"] else None
                result["validation_generated_patients"] = validation["generated_patients"]
            (directory / f"episode{episode}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            append_record(run_id, options["mode"], "episode", episode=episode,
                          scenario_id=scenario["id"], strategy=policy, metrics=result,
                          log_path=directory / "experiment_history.csv")
            emit("episode_end", episode=episode, result=result)
        message = "실행이 완료되었습니다."
        if training and not agent.training_updates:
            message += " 가중치 업데이트는 0회입니다. 학습 기간이나 에피소드를 늘려주세요."
        if training and best_score is None:
            message += " 학습 업데이트와 검증 환자를 모두 확보하지 못해 best 모델을 선정하지 않았습니다."
        emit("status", status="completed", message=message)
    except RunStopped:
        emit("status", status="stopped", message="실행을 중지했습니다. 완료된 에피소드는 저장되어 있습니다.")
    except Exception as exc:
        traceback.print_exc()
        emit("status", status="error", message=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            import traci
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass


class Dashboard:
    def __init__(self):
        self.lock = threading.RLock()
        self.context = mp.get_context("spawn")
        self.process = None
        self.token = secrets.token_urlsafe(24)
        self.state = {"status": "idle", "message": "시나리오를 설정하고 실행하세요.",
                      "episode": 0, "history": [], "frame": None, "speed": 30, "patient_count": 0}
        self.outcomes, self.outcome_ids, self.saved_cache = [], set(), {}
        self.network = None
        self.network_error = None
        threading.Thread(target=self.load_network, daemon=True).start()

    def load_network(self):
        try:
            self.network = network_data()
        except Exception as exc:
            self.network_error = str(exc)

    def snapshot(self):
        with self.lock:
            return {**copy.deepcopy(self.state), "worker_active": bool(self.process and self.process.is_alive())}

    def saved_runs(self):
        if not OUTPUT.exists():
            return []
        return [dict(id=p.name, completed_episodes=len(list(p.glob("episode[0-9]*.json"))))
                for p in sorted(OUTPUT.iterdir(), reverse=True)
                if p.is_dir() and (p / "experiment_history.csv").is_file()
                and p.name != self.state.get("run_id")]

    def patient_data(self, query):
        run = query.get("run", [""])[0]
        with self.lock:
            if not run or run == self.state.get("run_id"):
                return copy.deepcopy(self.outcomes), self.state.get("run_id"), self.state.get("patient_count", 0)
        # Use only exact directory names from the run listing, never arbitrary paths.
        if run not in {entry["id"] for entry in self.saved_runs()}:
            raise ValueError("저장된 실행을 찾을 수 없습니다.")
        directory = OUTPUT / run
        stat = (directory / "experiment_history.csv").stat()
        revision = f"{stat.st_mtime_ns}:{stat.st_size}"
        key = (run, revision)
        with self.lock:
            if key not in self.saved_cache:
                self.saved_cache = {key: load_saved_patients(directory, hospital_catalog(str(ROOT)))}
            return copy.deepcopy(self.saved_cache[key]), run, revision

    def patient_snapshot(self, query):
        records, run, revision = self.patient_data(query)
        return {**patient_report(records, parse_filters(query)), "run_id": run, "revision": revision}

    def export_snapshot(self):
        with self.lock:
            return {**copy.deepcopy(self.state), "patient_outcomes": copy.deepcopy(self.outcomes)}

    def start(self, body):
        options = validate_options(body)
        models = available_models()
        selected = body.get("model", "")
        model = next((p for p in models if str(p.relative_to(ROOT)).replace("\\", "/") == selected), None)
        if selected and model is None:
            raise ValueError("선택한 모델을 찾을 수 없습니다. 모델 목록을 새로고침하세요.")
        model = model or (models[0] if models else None)
        if options["mode"] == "simulation" and options["policy"] == "dqn" and model is None:
            raise ValueError("사용 가능한 best 모델이 없습니다. 먼저 학습하거나 다른 정책을 선택하세요.")
        options["model"] = str(model.relative_to(ROOT)).replace("\\", "/") if model else None
        if options["mode"] == "simulation" and options["policy"] == "dqn" and model is not None:
            manifest = model_manifest(model)
            if manifest:
                saved_count = sum(bool(h.get("virtual")) for h in manifest["hospitals"])
                if "virtual_count" not in body:
                    options["virtual_count"] = 2 if saved_count == 1 else saved_count
                if options["virtual_count"] != saved_count and manifest.get("agent_architecture") not in SHARED_ARCHITECTURES:
                    raise ValueError("이전 v3 모델에서 병원 수를 바꾸려면 v4 이상 모델이나 다른 병원 선택 정책을 선택하세요.")
        with self.lock:
            if self.process is not None and self.process.is_alive():
                raise ValueError("현재 실행을 중지한 후 다시 시작하세요." if self.state["status"] in ACTIVE
                                 else "직전 실행을 정리하고 있습니다. 잠시 후 다시 시작하세요.")
            self.events = self.context.Queue(maxsize=64)
            self.stop_event, self.pause_event = self.context.Event(), self.context.Event()
            self.speed = self.context.Value("d", self.state.get("speed", 30))
            self.state = {"status": "preparing", "message": "실행 환경을 불러오는 중입니다.",
                          "episode": 0, "history": [], "frame": None, "options": options,
                          "speed": self.speed.value, "patient_count": 0}
            self.outcomes, self.outcome_ids = [], set()
            self.process = self.context.Process(target=run_worker,
                                                args=(options, str(model) if model else None, self.events,
                                                      self.stop_event, self.pause_event, self.speed))
            self.process.start()
            threading.Thread(target=self.collect, args=(self.process, self.events), daemon=True).start()

    def collect(self, process, events):
        while True:
            try:
                event = events.get(timeout=.3)
            except queue.Empty:
                if process.is_alive():
                    continue
                with self.lock:
                    if self.process is not process:
                        break
                    if self.state["status"] in ACTIVE:
                        self.state.update(status="error", message=f"실행 프로세스가 종료되었습니다 (코드 {process.exitcode}).")
                        self.clear_traffic()
                break
            kind = event.pop("kind")
            with self.lock:
                if self.process is not process:
                    break
                if kind == "frame":
                    self.state["frame"] = event
                    self.state["frame_revision"] = self.state.get("frame_revision", 0) + 1
                elif kind == "outcome":
                    row = event["row"]
                    if row["record_id"] not in self.outcome_ids:
                        self.outcomes.append(row)
                        self.outcome_ids.add(row["record_id"])
                        self.state["patient_count"] = len(self.outcomes)
                elif kind == "hospital_rejection":
                    self.state["live_rejection_count"] = self.state.get("live_rejection_count", 0) + 1
                    self.state["recent_rejections"] = (self.state.get("recent_rejections", []) + [event["row"]])[-20:]
                    if self.state.get("frame"):
                        frame, penalty = self.state["frame"], event["row"]["penalty"]
                        frame["rejection_count"] = self.state["live_rejection_count"]
                        frame["reward"] = frame.get("reward", 0.) + penalty
                        parts = frame.setdefault("reward_components", {})
                        parts["rejection"] = parts.get("rejection", 0.) + penalty
                elif kind in {"episode_result", "episode_end"}:
                    if kind == "episode_end":
                        self.state["history"].append(event)
                    self.state["result"] = event["result"]
                    self.clear_traffic()
                elif kind == "episode_start":
                    self.state.update(event, frame=None, result=None, live_rejection_count=0, recent_rejections=[])
                elif kind == "status":
                    if self.state["status"] == "stopping" and event["status"] in ACTIVE:
                        event.pop("status")
                    elif self.pause_event.is_set() and event["status"] == "running":
                        event["status"] = "paused"
                    self.state.update(event)
                    if event.get("status") in {"stopped", "error", "completed"}:
                        self.clear_traffic()
                else:
                    self.state.update(event)
            if kind == "status" and event.get("status") in {"completed", "stopped", "error"}:
                process.join(timeout=5)
                break
        events.close()

    def clear_traffic(self):
        """Clear a closed SUMO session even when its last simulation time is unchanged."""
        if self.state.get("frame"):
            self.state["frame"].update(vehicles=[], patients=[], traffic=[], lanes=[], yielding_count=0,
                                       active=0, waiting=0)
        self.state["frame_revision"] = self.state.get("frame_revision", 0) + 1

    def control(self, body):
        action = body.get("action")
        with self.lock:
            if action == "speed":
                value = body.get("value")
                if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 300:
                    raise ValueError("배속은 1~300 사이여야 합니다.")
                self.state["speed"] = value
                if self.process and self.process.is_alive():
                    self.speed.value = value
                return
            if not self.process or not self.process.is_alive() or self.state["status"] not in ACTIVE:
                raise ValueError("진행 중인 실행이 없습니다.")
            if action == "stop":
                self.stop_event.set()
                self.pause_event.clear()
                self.state.update(status="stopping", message="실행을 안전하게 중지하는 중입니다.")
            elif action == "pause" and self.state["status"] == "running":
                self.pause_event.set()
                self.state["status"] = "paused"
            elif action == "resume" and self.state["status"] == "paused":
                self.pause_event.clear()
                self.state["status"] = "running"
            else:
                raise ValueError("현재 상태에서는 이 동작을 사용할 수 없습니다.")

    def close(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.pause_event.clear()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)


def handler_for(dashboard):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC), **kwargs)

        def log_message(self, *_):
            pass

        def respond(self, data, status=200):
            payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in {"/api/patients", "/api/patients.csv"}:
                try:
                    query = parse_qs(urlparse(self.path).query)
                    if path == "/api/patients":
                        return self.respond(dashboard.patient_snapshot(query))
                    records, _, _ = dashboard.patient_data(query)
                    payload = patients_csv(select_patients(records, parse_filters(query)))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="patient-transfers.csv"')
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                except ValueError as exc:
                    return self.respond({"error": str(exc)}, 400)
            if path == "/api/runs":
                return self.respond({"runs": dashboard.saved_runs()})
            if path == "/api/state":
                return self.respond(dashboard.snapshot())
            if path == "/api/config":
                return self.respond(dashboard_configuration(dashboard))
            if path == "/api/network":
                if dashboard.network_error:
                    return self.respond({"error": dashboard.network_error}, 500)
                if dashboard.network is None:
                    return self.respond({"loading": True}, 202)
                return self.respond(dashboard.network)
            if path == "/api/export":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="ambulance-results.json"')
                self.end_headers()
                self.wfile.write(json.dumps(dashboard.export_snapshot(), ensure_ascii=False, indent=2).encode("utf-8"))
                return
            if path.startswith("/api/"):
                return self.respond({"error": "API를 찾을 수 없습니다."}, 404)
            return super().do_GET()

        def do_POST(self):
            # Requests that launch work must originate from this local dashboard.
            if self.headers.get("X-Dashboard-Token") != dashboard.token:
                return self.respond({"error": "화면을 새로고침한 뒤 다시 시도하세요."}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("요청 크기가 올바르지 않습니다.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("요청 형식이 올바르지 않습니다.")
                if self.path == "/api/start":
                    dashboard.start(body)
                elif self.path == "/api/control":
                    dashboard.control(body)
                else:
                    return self.respond({"error": "API를 찾을 수 없습니다."}, 404)
                return self.respond(dashboard.snapshot())
            except (ValueError, TypeError) as exc:
                return self.respond({"error": str(exc)}, 400)
            except Exception as exc:
                traceback.print_exc()
                return self.respond({"error": f"실행을 시작하지 못했습니다: {exc}"}, 500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="구급차 강화학습 웹 대시보드")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    ensure_sumo()
    dashboard = Dashboard()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(dashboard))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Ambulance dashboard: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        server.server_close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
