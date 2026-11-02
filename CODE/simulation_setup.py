"""Verified network preparation and experiment reconstruction shared by all runners."""
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import sumolib
import traci

from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from experiment_config import SCENARIOS
from reward import GOLDEN_TIME_LIMITS, WEIGHTS

MAX_ESTIMATED_TRAVEL_TIME = 600.


def start_sumo(env, scenario, seed, gui=False, output_dir=None, quiet=False):
    """Keep all SUMO outputs in this run directory, never in source assets."""
    import xml.etree.ElementTree as ET
    output = Path(output_dir or Path(__file__).resolve().parents[1] / "results" / "sumo")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    additional = []
    input_node = ET.parse(env.cfg_file).getroot().find("input/additional-files")
    if input_node is not None:
        for value in input_node.get("value", "").split(","):
            path = (env.cfg_file.parent / value.strip()).resolve()
            tree = ET.parse(path)
            outputs = [element for element in tree.getroot().iter() if element.get("file")]
            if outputs:
                for element in outputs:
                    element.set("file", str(output / Path(element.get("file")).name))
                target = output / path.name
                tree.write(target, encoding="utf-8", xml_declaration=True)
                additional.append(str(target))
            else:
                additional.append(str(path))
    command = [
        sumolib.checkBinary("sumo-gui" if gui else "sumo"), "-c", str(env.cfg_file),
        "--scale", str(scenario["scale"]), "--seed", str(seed),
        "--no-warnings", "true", "--no-step-log", "true",
        "--duration-log.disable", "true", "--step-length", "1",
        "--tripinfo-output", str(output / "tripinfos.xml"),
        "--statistic-output", str(output / "stats.xml"), "--start",
    ]
    if quiet:
        command.extend(["--verbose", "false"])
    if additional:
        command.extend(["--additional-files", ",".join(additional)])
    with suppress_sumo_stdout():
        traci.start(command)
        traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_custom")
        traci.vehicletype.setVehicleClass("ambulance_custom", "emergency")
        traci.vehicletype.setShapeClass("ambulance_custom", "emergency")
    env.update_hospital_coordinates()
    for hospital in env.get_hospitals():
        if hospital.get("virtual"):
            traci.poi.add(hospital["id"], float(hospital["sumo_x"]), float(hospital["sumo_y"]),
                          (0, 180, 80, 255), poiType="virtual_hospital",
                          layer=20, width=25, height=25)


@contextlib.contextmanager
def suppress_sumo_stdout():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def edge_allows_ambulance(env, edge_id):
    """TraCI에서 실제 ambulance_custom 출발 가능 여부를 1회 확인."""
    try:
        route = traci.simulation.findRoute(
            edge_id, edge_id,
            vType="ambulance_custom",
            depart=-1,
        )
        return route is not None and len(route.edges) > 0
    except Exception:
        return False


def _hospital_access_edges(env, hospitals, per_hospital=12, radius=1200.0):
    """Cache drivable candidate edges near each hospital, ordered by distance."""
    result = []
    for hospital in hospitals:
        x, y = hospital.get("sumo_x"), hospital.get("sumo_y")
        if x is None or y is None:
            result.append([])
            continue
        nearby = sorted(env.net.getNeighboringEdges(x, y, radius), key=lambda pair: pair[1])
        candidates = []
        seen = set()
        for edge, _ in nearby:
            edge_id = edge.getID()
            if edge_id.startswith(":") or edge_id in seen or not edge.allows("emergency"):
                continue
            seen.add(edge_id)
            candidates.append(edge_id)
            if len(candidates) >= per_hospital:
                break
        result.append(candidates)
    return result


def _reverse_reachable_edges(env, destination_ids):
    """병원 진입 Edge에서 역방향 그래프 탐색. 방향성을 고려한 빠른 사전 필터."""
    from collections import deque
    queue = deque(destination_ids)
    seen = set(destination_ids)
    while queue:
        edge_id = queue.popleft()
        edge = env.net.getEdge(edge_id)
        for previous in edge.getIncoming():
            prev_id = previous.getID()
            if prev_id in seen or prev_id.startswith(":"):
                continue
            if hasattr(previous, "allows") and not previous.allows("emergency"):
                continue
            seen.add(prev_id)
            queue.append(prev_id)
    return seen


def select_network_reachable_hospitals(env, sample_sources=3, candidate_targets=4):
    """Retain hospitals reachable from at least one emergency-drivable source.

    Two-stage check: directed network graph then TraCI findRoute. This does NOT
    guarantee every patient can reach every hospital; dispatch still checks.
    """
    hospitals = env.get_hospitals()
    access_edges = _hospital_access_edges(env, hospitals)
    ambulance_edges = set(env.get_ambulance_candidate_edges())
    accepted = []
    reason_counts = Counter()
    rng = random.Random(20261030)

    for hospital, targets in zip(hospitals, access_edges):
        if not targets:
            reason_counts["no_nearby_emergency_edge"] += 1
            continue
        # The graph is directional. Restrict origins to edges that can reach
        # at least one candidate destination, rather than sampling globally.
        graph_reachable = _reverse_reachable_edges(env, targets)
        possible_origins = sorted(graph_reachable & ambulance_edges)
        if not possible_origins:
            reason_counts["no_graph_reachable_origin"] += 1
            continue

        # Spread origins across the reachable component, deterministically.
        origins = [possible_origins[0], possible_origins[len(possible_origins)//2],
                   possible_origins[-1]]
        if len(possible_origins) > 3:
            origins += rng.sample(possible_origins, min(sample_sources, len(possible_origins)))
        origins = list(dict.fromkeys(origins))
        verified = False
        verified_target = None
        for origin in origins:
            for target in targets[:candidate_targets]:
                try:
                    route = traci.simulation.findRoute(
                        origin, target, vType="ambulance_custom", depart=-1
                    )
                    if route and route.edges and math.isfinite(float(route.travelTime)):
                        verified = True
                        verified_target = target
                        break
                except traci.TraCIException:
                    continue
            if verified:
                break
        if not verified:
            reason_counts["traci_route_not_verified"] += 1
            continue
        hospital["verified_access_edge"] = verified_target
        hospital["graph_reachable_origins"] = len(possible_origins)
        accepted.append(hospital)

    env.hospitals = accepted
    print(f"[병원 네트워크 검증] 사전 후보={len(hospitals)}, "
          f"TraCI 도달 확인={len(accepted)}, 제외={dict(reason_counts)}")
    if not accepted:
        raise RuntimeError("실제 경로가 확인된 병원이 없습니다. POI 좌표와 emergency 도로망을 확인하세요.")
    return accepted


def add_virtual_hospitals(env, count, strategy="far", placement_seed=42,
                          min_spacing_m=1300.0, max_candidates=800):
    """Place research-only hospitals on drivable SUMO edges; verify real routes.

    `demand-proxy` maximizes nearby potential patient-spawn edge density,
    NOT observed KTAS 1/2 calls. All virtual centers have the same capability.
    """
    if count <= 0:
        return []
    rng = random.Random(placement_seed)
    real_hospitals = list(env.get_hospitals())
    origins = sorted(set(env.get_ambulance_candidate_edges()))
    if not origins:
        raise RuntimeError("구급차 주행 가능 Edge가 없습니다")
    # Stable sample of origins for reachability verification.
    sample_origins = [origins[i] for i in (0, len(origins)//2, len(origins)-1)]
    sample_origins += rng.sample(origins, min(12, len(origins)))
    sample_origins = list(dict.fromkeys(sample_origins))
    # Draw candidate positions only from actual SUMO network edges.
    candidate_ids = [e for e in origins if not e.startswith(":")]
    rng.shuffle(candidate_ids)
    candidate_ids = candidate_ids[:max_candidates]
    points = []
    for edge_id in candidate_ids:
        try:
            edge = env.net.getEdge(edge_id)
            if not edge.allows("emergency"):
                continue
            x, y = edge.getShape()[len(edge.getShape())//2]
            points.append((edge_id, float(x), float(y)))
        except Exception:
            continue
    if not points:
        raise RuntimeError("가상 병원 후보 Edge가 없습니다")
    anchors = [(float(h["sumo_x"]), float(h["sumo_y"])) for h in real_hospitals]
    def sqdist(x, y, xx, yy):
        return (x-xx)**2 + (y-yy)**2
    # Spatial proxy based on potential spawnable road-edge density.
    density_points = points[:]
    ranked = []
    for edge_id, x, y in points:
        nearest = min((sqdist(x, y, xx, yy) for xx, yy in anchors), default=float("inf"))
        if nearest < min_spacing_m**2:
            continue
        density = sum(sqdist(x, y, xx, yy) < 2000.0**2 for _, xx, yy in density_points)
        score = rng.random() if strategy == "random" else (nearest if strategy == "far" else density)
        ranked.append((score, edge_id, x, y))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    for _, edge_id, x, y in ranked:
        if any(sqdist(x, y, xx, yy) < min_spacing_m**2 for xx, yy in anchors):
            continue
        verified = False
        for origin in sample_origins:
            try:
                route = traci.simulation.findRoute(origin, edge_id,
                                                    vType="ambulance_custom", depart=-1)
                if route and route.edges and math.isfinite(float(route.travelTime)):
                    verified = True
                    break
            except traci.TraCIException:
                continue
        if not verified:
            continue
        lon, lat = env.net.convertXY2LonLat(x, y)
        hospital = {
            "id": f"VIRTUAL_{strategy.upper().replace('-', '_')}_{placement_seed}_{len(selected)+1:02d}",
            "name": f"가상 권역응급의료센터 {len(selected)+1}",
            "lat": float(lat), "lon": float(lon),
            "sumo_x": x, "sumo_y": y,
            "type": "권역응급의료센터", "occupancy": 0.30,
            "virtual": True, "verified_access_edge": edge_id,
        }
        selected.append(hospital)
        anchors.append((x, y))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"가상 병원 {count}개 요청 중 {len(selected)}개만 도달 검증됨. "
                           "배치 조건/후보 수를 조정하세요")
    env.hospitals.extend(selected)
    env.hospitals.sort(key=lambda h: h["id"])
    print(f"[가상 병원] 전략={strategy}, 시드={placement_seed}, 추가={len(selected)}개")
    for h in selected:
        print(f"  {h['id']} | edge={h['verified_access_edge']} | "
              f"lon={h['lon']:.6f}, lat={h['lat']:.6f}")
    return selected


def build_ambulance_edges(env, max_check=1200, hospitals=None):
    """병원 도달 가능성을 검증한 환자 발생 Edge만 반환한다.

    1) 병원 근처의 구급차 진입 가능 Edge를 찾는다.
    2) 역방향 그래프 탐색으로 병원 접근 가능 도로를 사전 필터링한다.
    3) TraCI findRoute로 실제 도달 가능한 출발 Edge를 최종 확인한다.

    병상 점유율/KTAS 적합성은 매 환자 배정 시점에 따로 검사한다.
    """
    hospitals = hospitals if hospitals is not None else env.get_hospitals()
    hospital_edges = _hospital_access_edges(env, hospitals)
    targets = list(dict.fromkeys(e for group in hospital_edges for e in group))
    if not targets:
        raise RuntimeError("병원 인근의 ambulance 진입 Edge가 없습니다. 병원 좌표와 네트워크를 확인하세요.")

    reachable = _reverse_reachable_edges(env, targets)
    candidates = [e for e in env.get_ambulance_candidate_edges() if e in reachable]
    rng = random.Random(20261007)
    rng.shuffle(candidates)
    valid = []
    checked = 0
    # 병원별 최인접 Edge를 우선 확인하고, 실패하면 추가 진입 Edge를 검사한다.
    preferred = [group[0] for group in hospital_edges if group]
    fallback = [e for e in targets if e not in preferred]
    for edge_id in candidates:
        checked += 1
        if not edge_allows_ambulance(env, edge_id):
            continue
        # 병원 접근성이 없는 도로는 여기서 제거한다.
        found = False
        for target in preferred + fallback:
            try:
                route = traci.simulation.findRoute(
                    edge_id, target, vType="ambulance_custom", depart=-1
                )
                if route is not None and len(route.edges) > 0:
                    found = True
                    break
            except traci.TraCIException:
                continue
        if found:
            valid.append(edge_id)
        if len(valid) >= max_check:
            break

    print(f"[병원 접근성] 그래프 후보={len(candidates)}, 실제 검증={checked}, 유효={len(valid)}")
    if not valid:
        raise RuntimeError("병원까지 도달 가능한 환자 발생 Edge를 찾지 못했습니다. 병원 진입 Edge 및 네트워크를 확인하세요.")
    return valid


def get_reachable_hospital_routes(env, patient_edge, hospitals, edge_candidates=None):
    """Choose the minimum estimated travel-time route among nearby hospital edges.

    Does not relax hospital capability or occupancy constraints.
    """
    result = copy.deepcopy(hospitals)
    failures = 0
    if edge_candidates is None:
        edge_candidates = _hospital_access_edges(env, hospitals)
    for idx, hospital in enumerate(result):
        hospital.update(reachable=False, hospital_edge=None, route_length=None,
                        route_edges=None, route_error=None,
                        estimated_travel_time=MAX_ESTIMATED_TRAVEL_TIME)
        candidates = edge_candidates[idx]
        if not candidates:
            hospital["route_error"] = "no_hospital_access_edge"
            failures += 1
            continue
        best = None
        for target in candidates:
            try:
                with suppress_sumo_stdout():
                    route = traci.simulation.findRoute(
                        patient_edge, target, vType="ambulance_custom",
                        depart=-1, routingMode=0)
                if not route or not route.edges:
                    continue
                length = float(route.length)
                travel = float(route.travelTime)
                if not math.isfinite(length) or length < 0 or not math.isfinite(travel):
                    continue
                if travel <= 0:
                    travel = max(length / 13.9, 1.0)
                if best is None or travel < best[0]:
                    best = (travel, length, target, list(route.edges))
            except traci.TraCIException:
                continue
        if best is None:
            hospital["route_error"] = "no_route_to_hospital_candidate"
            failures += 1
            continue
        travel, length, target, edges = best
        hospital.update(reachable=True, hospital_edge=target,
                        route_length=length, route_edges=edges,
                        estimated_travel_time=travel)
    return result, failures



def prepare_environment(config, virtual_count=0, virtual_strategy="far", placement_seed=42,
                        output_dir=None, quiet=False):
    from experiment_config import seed_everything
    seed_everything(config.seed)
    env = SumoMedicalEnvironment()
    try:
        start_sumo(env, SCENARIOS[0], config.seed, output_dir=output_dir, quiet=quiet)
        select_network_reachable_hospitals(env)
        add_virtual_hospitals(env, virtual_count, virtual_strategy, placement_seed)
        hospitals = copy.deepcopy(env.get_hospitals())
        edges = build_ambulance_edges(env, max_check=config.candidate_limit, hospitals=hospitals)
        access = _hospital_access_edges(env, hospitals)
    finally:
        if traci.isLoaded():
            traci.close()
    env.ambulance_edges = edges
    env.hospital_access_edges = access
    return env, hospitals, edges, access


def network_digest(env):
    digest = hashlib.sha256()
    with Path(env.net_file).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_signature(manifest):
    data = {k: v for k, v in manifest.items() if k != "signature"}
    return hashlib.sha256(json.dumps(data, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def make_manifest(env, hospitals, edges, access, config, ambulance_count, queue_policy,
                  aging_seconds, curriculum):
    from experiment_config import EXPERIMENT_VERSION, STATE_SCHEMA_VERSION
    from dataclasses import asdict
    manifest = {
        "version": EXPERIMENT_VERSION, "state_schema_version": STATE_SCHEMA_VERSION,
        "config": config.to_dict(), "ambulance_count": ambulance_count,
        "queue_policy": queue_policy, "aging_seconds": aging_seconds,
        "curriculum": list(curriculum), "hospitals": hospitals,
        "ambulance_edges": edges, "hospital_access_edges": access,
        "network_sha256": network_digest(env),
        "reward_weights": asdict(WEIGHTS), "golden_time_limits": {str(k): v for k, v in GOLDEN_TIME_LIMITS.items()},
    }
    manifest["signature"] = manifest_signature(manifest)
    return manifest


def load_experiment(model_path):
    """Reconstruct the exact hospital order and verified origins used in training."""
    import torch
    from experiment_config import EXPERIMENT_VERSION, SimulationConfig
    from baseline import state_dimension
    path = Path(model_path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    manifest_path = Path(metadata.get("manifest_path", ""))
    if not manifest_path.is_file():
        raise ValueError("This model has no v3 experiment manifest. Train a new model.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") != EXPERIMENT_VERSION
            or manifest_signature(manifest) != manifest.get("signature")
            or metadata.get("experiment_signature") != manifest.get("signature")):
        raise ValueError("Model and experiment manifest do not match.")
    env = SumoMedicalEnvironment()
    if network_digest(env) != manifest["network_sha256"]:
        raise ValueError("The SUMO network changed since training.")
    env.hospitals = copy.deepcopy(manifest["hospitals"])
    env._coordinates_ready = True
    env.ambulance_edges = list(manifest["ambulance_edges"])
    env.hospital_access_edges = manifest["hospital_access_edges"]
    agent = DQNAmbulanceAgent(state_dimension(len(env.hospitals)), len(env.hospitals))
    agent.load_model(str(path))
    agent.epsilon = 0.
    agent.model.eval()
    return env, copy.deepcopy(env.hospitals), agent, SimulationConfig(**manifest["config"]), manifest


def latest_model(output_root=None):
    root = Path(output_root) if output_root else Path(__file__).resolve().parents[1] / "results"
    choices = list((root / "models").glob("dqn_v3_*_last.pth"))
    if not choices:
        raise FileNotFoundError("No v3 model found; run train_agent.py first or provide --model.")
    return max(choices, key=lambda p: p.stat().st_mtime)
