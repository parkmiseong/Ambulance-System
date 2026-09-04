# -*- coding: utf-8 -*-
"""
ambulance_rl.py  —  SUMO + TraCI 기반 구급차 이송병원 선정 RL 시뮬레이션
"""

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

# ── SUMO 경로 설정 ──────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    sys.exit(
        "[오류] 환경변수 SUMO_HOME이 설정되지 않았습니다.\n"
        "  export SUMO_HOME=/path/to/sumo  (Linux/macOS)\n"
        "  set SUMO_HOME=C:\\...\\sumo      (Windows)"
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib
import traci


# ══════════════════════════════════════════════════════════════
# §1. 설정
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
SUMO_DIR         = PROJECT_ROOT / "SUMO" / "강남"
NET_FILE         = SUMO_DIR / "osm.net.xml"
CFG_FILE         = SUMO_DIR / "osm.sumocfg"
USE_GUI          = True
N_EPISODES       = 50
PATIENT_INTERVAL = 60    # 환자 발생 간격 (시뮬 초)
LOADING_TIME     = 5     # 환자 탑승 대기 스텝
REWARD_SCALE     = 1/100.0

HOSPITALS = [
    {"name":"삼성서울병원",     "lon":127.0851,"lat":37.4881,
     "capability":3,"capacity":100,
     "equipment":["CT","MRI","수술실","중환자실","혈관조영실"]},
    {"name":"강남세브란스병원", "lon":127.0634,"lat":37.4929,
     "capability":3,"capacity":80,
     "equipment":["CT","MRI","수술실","중환자실","혈관조영실"]},
    {"name":"강남성심병원",     "lon":127.0562,"lat":37.5193,
     "capability":2,"capacity":40,
     "equipment":["CT","혈액검사기","수술실"]},
    {"name":"강남병원",         "lon":127.0326,"lat":37.5040,
     "capability":1,"capacity":20,
     "equipment":["혈액검사기"]},
]

SEV_REQUIRED_EQ = {
    1: [],
    2: ["CT","혈액검사기"],
    3: ["CT","수술실"],
    4: ["CT","수술실","중환자실"],
    5: ["혈관조영실","중환자실","수술실"],
}

SEV_COLOR = {
    1: (200,200,0,255), 2: (255,165,0,255),
    3: (255,80,80,255), 4: (200,0,0,255), 5: (100,0,0,255),
}


# ══════════════════════════════════════════════════════════════
# §2. NetworkHelper  ← 원본 get_reachable_patient_edge 완전 대체
# ══════════════════════════════════════════════════════════════
class NetworkHelper:
    """
    sumolib 네트워크를 래핑해 좌표 기반 edge 탐색을 제공.

    원본 문제:
        net.getEdges() 전체 순회 → findRoute O(N) 반복 호출
        강남 네트워크 기준 수만 번 API 호출 → 수분 이상 소요

    개선:
        net.convertLonLat2XY() + net.getNeighboringEdges(x, y, r)
        → O(1) 수준, 수 ms 내 완료
    """
    def __init__(self, net_file: str):
        print(f"[NetworkHelper] 네트워크 로딩: {net_file}")
        self.net = sumolib.net.readNet(net_file)
        print(f"[NetworkHelper] edge {len(list(self.net.getEdges()))}개 로드")

    def lon_lat_to_edge(self, lon: float, lat: float,
                        radius: float = 300.0) -> Optional[str]:
        """
        위도·경도 → 가장 가까운 유효 edge ID.

        Parameters
        ----------
        lon, lat : WGS84 경위도
        radius   : 탐색 반경(m) — 결과 없으면 자동 2배 확장

        Notes
        -----
        내부 junction edge(ID가 ':' 로 시작)는 제외.
        """
        x, y = self.net.convertLonLat2XY(lon, lat)
        for r in [radius, radius * 2, radius * 4]:
            neighbors = self.net.getNeighboringEdges(x, y, r)
            valid = [(e, d) for e, d in neighbors
                     if not e.getID().startswith(":")]
            if valid:
                valid.sort(key=lambda ed: ed[1])   # 거리 오름차순
                return valid[0][0].getID()
        return None

    def edge_to_lon_lat(self, edge_id: str):
        """edge 시작 노드 좌표 → (lon, lat)"""
        edge = self.net.getEdge(edge_id)
        x, y = edge.getFromNode().getCoord()
        return self.net.convertXY2LonLat(x, y)


# ══════════════════════════════════════════════════════════════
# §3. 병원 edge 매핑
# ══════════════════════════════════════════════════════════════
def init_hospitals(net_helper: NetworkHelper):
    print("\n[병원 edge 매핑]")
    for h in HOSPITALS:
        eid = net_helper.lon_lat_to_edge(h["lon"], h["lat"])
        h["edge_id"]  = eid
        h["occupied"] = 0
        status = eid if eid else "❌ 매핑 실패"
        print(f"  {h['name']:<18} → {status}")
    return [h for h in HOSPITALS if h.get("edge_id")]


# ══════════════════════════════════════════════════════════════
# §4. MDP — State / Action / Reward
# ══════════════════════════════════════════════════════════════
def required_level(sev: int) -> int:
    return 3 if sev >= 5 else 2 if sev >= 3 else 1

def has_required_eq(sev: int, h: dict) -> bool:
    req = SEV_REQUIRED_EQ.get(sev, [])
    return not req or any(e in h["equipment"] for e in req)

def eq_score(sev: int, h: dict) -> float:
    req = SEV_REQUIRED_EQ.get(sev, [])
    return sum(1 for e in req if e in h["equipment"]) / len(req) if req else 1.0

def get_tt_min(from_e: str, to_e: str) -> float:
    """TraCI findRoute → 이동시간(분). 실패 시 999 반환."""
    if not from_e or not to_e:
        return 999.0
    try:
        r = traci.simulation.findRoute(from_e, to_e, vType="ambulance")
        if r and r.travelTime > 0:
            return r.travelTime / 60.0
    except Exception:
        pass
    return 999.0

def build_obs(sev: int, p_edge: str, hospitals: list) -> np.ndarray:
    """
    State 벡터: [sev/5, tt_j/30, occ_j, cap_j/3, eq_score_j] × H
    차원: 1 + 4 × len(hospitals)
    """
    obs = [sev / 5.0]
    for h in hospitals:
        tt  = get_tt_min(p_edge, h["edge_id"])
        occ = min(h["occupied"] / max(h["capacity"], 1), 1.5)
        obs.extend([min(tt / 30.0, 1.0), occ, h["capability"] / 3.0,
                    eq_score(sev, h)])
    return np.array(obs, dtype=np.float32)

def compute_reward(sev: int, h: dict, tt_min: float) -> tuple:
    """
    Reward = -(severity_w × delay + cap_gap_pen + eq_pen)

    Reward 구성요소
    ───────────────
    dispatch_time  : 이동시간 + 과밀 패널티
    hospital_state : 역량 불일치 (capability gap)
    patient_cond   : 기기 부재 (equipment penalty)
    """
    OVERFLOW_U = 8.0
    MISMATCH_U = 25.0
    EQ_U       = 20.0

    overflow = max(0, h["occupied"] + 1 - h["capacity"])
    cap_gap  = max(0, required_level(sev) - h["capability"])
    eq_ok    = has_required_eq(sev, h)
    eq_pen   = 0 if eq_ok else EQ_U * len(SEV_REQUIRED_EQ.get(sev, []))

    sev_w  = sev ** 1.5
    delay  = tt_min + overflow * OVERFLOW_U
    reward = -(sev_w * delay + cap_gap * MISMATCH_U + eq_pen)

    info = {"tt_min": round(tt_min, 2), "overflow": overflow,
            "cap_gap": cap_gap, "eq_ok": eq_ok, "reward": round(reward, 2)}
    return reward * REWARD_SCALE, info


# ══════════════════════════════════════════════════════════════
# §5. Baseline 정책 3종
# ══════════════════════════════════════════════════════════════
def policy_nearest(sev, p_edge, hospitals):
    tts = [get_tt_min(p_edge, h["edge_id"]) for h in hospitals]
    return int(np.argmin(tts))

def policy_rule(sev, p_edge, hospitals):
    req = required_level(sev)
    tts = [get_tt_min(p_edge, h["edge_id"]) for h in hospitals]
    el  = [i for i,h in enumerate(hospitals) if h["capability"] >= req]
    return min(el or range(len(hospitals)), key=lambda i: tts[i])

def policy_equipment(sev, p_edge, hospitals):
    req = required_level(sev)
    tts = [get_tt_min(p_edge, h["edge_id"]) for h in hospitals]
    el  = [i for i,h in enumerate(hospitals)
           if h["capability"] >= req and has_required_eq(sev, h)]
    if not el:
        el = [i for i,h in enumerate(hospitals) if h["capability"] >= req]
    return min(el or range(len(hospitals)), key=lambda i: tts[i])

POLICY_FN = {
    "nearest"  : policy_nearest,
    "rule"     : policy_rule,
    "equipment": policy_equipment,
}


# ══════════════════════════════════════════════════════════════
# §6. 구급차 개체
# ══════════════════════════════════════════════════════════════
@dataclass
class Ambulance:
    amb_id       : str
    home_edge    : str
    severity     : int  = 0
    patient_edge : str  = ""
    hosp_edge    : str  = ""
    hosp_idx     : int  = -1
    # WAITING → TO_PATIENT → LOADING → TO_HOSPITAL → DONE → WAITING
    state        : str  = "WAITING"
    loading_cnt  : int  = 0
    dispatch_step: int  = 0
    obs          : Optional[np.ndarray] = field(default=None, repr=False)
    action       : int  = -1


def spawn_amb(amb_id, home_edge):
    route_id = f"route_{amb_id}"
    traci.route.add(route_id, [home_edge])
    traci.vehicle.add(amb_id, route_id, typeID="ambulance",
                      depart="now", departPos="base", departSpeed="max")
    traci.vehicle.setSpeedMode(amb_id, 7)    # 신호·속도제한 무시
    traci.vehicle.setLaneChangeMode(amb_id, 0)
    traci.vehicle.setColor(amb_id, (0, 120, 255, 255))  # 대기: 파랑

def dispatch_to(amb_id, from_e, to_e):
    """구급차 목적지 동적 변경"""
    if from_e == to_e:
        return
    try:
        r = traci.simulation.findRoute(from_e, to_e, vType="ambulance")
        if r and len(r.edges) > 1:
            traci.vehicle.setRoute(amb_id, list(r.edges))
            return
    except Exception:
        pass
    traci.vehicle.changeTarget(amb_id, to_e)

def cur_edge(amb_id) -> str:
    e = traci.vehicle.getRoadID(amb_id)
    if e.startswith(":"):
        e = traci.vehicle.getLaneID(amb_id).rsplit("_", 1)[0]
    return e


# ══════════════════════════════════════════════════════════════
# §7. 1 Episode 실행
# ══════════════════════════════════════════════════════════════
def run_episode(net_helper, hospitals, policy="equipment", episode=0):
    # 소방서 거점 (강남소방서 좌표)
    home_edge = net_helper.lon_lat_to_edge(127.0556, 37.5123) or ""
    if not home_edge:
        print("[경고] 소방서 edge 매핑 실패"); return {}

    binary = sumolib.checkBinary("sumo-gui" if USE_GUI else "sumo")
    traci.start([binary, "-c", str(CFG_FILE),
                 "--no-step-log", "true",
                 "--collision.action", "none",
                 "--time-to-teleport", "-1"])

    try:
        # 구급차 초기 삽입 (5스텝 후)
        for _ in range(5):
            traci.simulationStep()
        spawn_amb("amb_1", home_edge)

        amb = Ambulance(amb_id="amb_1", home_edge=home_edge)
        total_r, patient_cnt, completed_cnt = 0.0, 0, 0
        next_patient = 5 + PATIENT_INTERVAL
        policy_fn = POLICY_FN.get(policy, policy_equipment)
        log = []

        for step in range(6, 8000):
            traci.simulationStep()
            active = traci.vehicle.getIDList()

            # ── 환자 발생 ──────────────────────────────────────
            if step >= next_patient and amb.state == "WAITING" \
                    and amb.amb_id in active:
                sev = random.choices([1,2,3,4,5], weights=[35,30,20,10,5])[0]
                ce  = cur_edge(amb.amb_id)

                # 환자 위치: 현재 위치 인근 랜덤 edge
                lon, lat = net_helper.edge_to_lon_lat(ce)
                p_edge = net_helper.lon_lat_to_edge(
                    lon + random.uniform(-0.007, 0.007),
                    lat + random.uniform(-0.005, 0.005)
                )
                if not p_edge:
                    next_patient = step + 10
                    continue

                # MDP: obs → action → reward
                obs   = build_obs(sev, p_edge, hospitals)
                hIdx  = policy_fn(sev, p_edge, hospitals)
                # ★ DQN 연동 시: hIdx = agent.act(obs)

                h     = hospitals[hIdx]
                tt    = get_tt_min(p_edge, h["edge_id"])
                r, info = compute_reward(sev, h, tt)
                total_r += r
                h["occupied"] += 1
                patient_cnt   += 1

                # 구급차 출동
                dispatch_to(amb.amb_id, ce, p_edge)
                traci.vehicle.setColor(amb.amb_id, SEV_COLOR[sev])

                amb.severity      = sev
                amb.patient_edge  = p_edge
                amb.hosp_edge     = h["edge_id"]
                amb.hosp_idx      = hIdx
                amb.state         = "TO_PATIENT"
                amb.dispatch_step = step
                amb.obs           = obs
                amb.action        = hIdx
                next_patient      = step + PATIENT_INTERVAL

                print(f"[Ep{episode} Step{step:4d}] Sev={sev} "
                      f"→ {h['name']} | {tt:.1f}분 "
                      f"r={info['reward']:.1f} "
                      f"(gap={info['cap_gap']}, eq={'OK' if info['eq_ok'] else 'MISS'})")
                log.append({"step":step,"sev":sev,"hospital":h["name"],**info})

            # ── 상태 전이 ──────────────────────────────────────
            if amb.amb_id not in active:
                continue

            ce = cur_edge(amb.amb_id)

            if amb.state == "TO_PATIENT" and ce == amb.patient_edge:
                amb.state       = "LOADING"
                amb.loading_cnt = 0
                traci.vehicle.setSpeed(amb.amb_id, 0)     # 정차
                print(f"[Step{step:4d}] 환자 탑승 중...")

            elif amb.state == "LOADING":
                amb.loading_cnt += 1
                if amb.loading_cnt >= LOADING_TIME:
                    traci.vehicle.setSpeed(amb.amb_id, -1)
                    dispatch_to(amb.amb_id, amb.patient_edge, amb.hosp_edge)
                    amb.state = "TO_HOSPITAL"
                    print(f"[Step{step:4d}] 병원 출발 → "
                          f"{hospitals[amb.hosp_idx]['name']}")

            elif amb.state == "TO_HOSPITAL" and ce == amb.hosp_edge:
                hospitals[amb.hosp_idx]["occupied"] = max(
                    0, hospitals[amb.hosp_idx]["occupied"] - 1)
                completed_cnt += 1
                elapsed = step - amb.dispatch_step
                print(f"[Step{step:4d}] 이송 완료 ({elapsed}s) → 귀환")
                # ★ DQN 연동 시: agent.push(amb.obs, amb.action, r, next_obs, False)
                dispatch_to(amb.amb_id, amb.hosp_edge, amb.home_edge)
                amb.state = "WAITING"
                traci.vehicle.setColor(amb.amb_id, (0,120,255,255))

            if completed_cnt >= 10:
                print(f"[Episode {episode}] 10건 이송 완료")
                break

        return {"episode":episode, "policy":policy,
                "patient_count":completed_cnt,
                "dispatched_count":patient_cnt,
                "total_reward":round(total_r,3),
                "avg_reward":round(total_r/max(completed_cnt,1),3),
                "log":log}
    finally:
        traci.close()


# ══════════════════════════════════════════════════════════════
# §8. 메인
# ══════════════════════════════════════════════════════════════
def main():
    net_file = NET_FILE.with_suffix(".xml.gz") if NET_FILE.with_suffix(".xml.gz").exists() else NET_FILE
    if not CFG_FILE.exists():
        sys.exit(f"[오류] SUMO 설정 파일을 찾을 수 없습니다: {CFG_FILE}")
    if not net_file.exists():
        sys.exit(f"[오류] SUMO 네트워크 파일을 찾을 수 없습니다: {net_file}")

    net_helper = NetworkHelper(str(net_file))
    hospitals  = init_hospitals(net_helper)
    if not hospitals:
        sys.exit("[오류] 유효한 병원이 없습니다.")

    print(f"\n병원 {len(hospitals)}개 | "
          f"OBS dim = 1+4×{len(hospitals)} = {1+4*len(hospitals)} | "
          f"ACTION = {len(hospitals)}")

    results = []
    for ep, pol in enumerate(["nearest","rule","equipment"]):
        print(f"\n{'='*60}\nEpisode {ep+1} — 정책: {pol}\n{'='*60}")
        results.append(run_episode(net_helper, hospitals,
                                   policy=pol, episode=ep+1))
        time.sleep(1)

    print(f"\n{'='*60}\n정책 비교 결과\n{'='*60}")
    print(f"{'정책':<18} {'환자수':>6} {'총보상':>10} {'평균보상':>10}")
    print("-"*46)
    for r in results:
        print(f"{r['policy']:<18} {r['patient_count']:>6} "
              f"{r['total_reward']:>10.3f} {r['avg_reward']:>10.3f}")
        
    obs   = build_obs(sev, p_edge, hospitals)
    hIdx  = agent.act(obs)          # ← 이 줄로 policy_fn 교체
    agent.push(obs, hIdx, reward, next_obs, done=False)
    agent.update()                  # replay buffer 학습

    print("""
[3단계: DQN 연동 포인트]
  ① obs   = build_obs(sev, p_edge, hospitals)
  ② hIdx  = agent.act(obs)          # ← 이 줄로 policy_fn 교체
  ③ agent.push(obs, hIdx, reward, next_obs, done=False)
  ④ agent.update()                  # replay buffer 학습
""")


if __name__ == "__main__":
    main()
