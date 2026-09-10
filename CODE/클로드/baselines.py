# baselines.py  ── 개선 버전
#!/usr/bin/env python3
"""
[수정 사항]
  1. nearest: 유클리드 거리 → travel_times 파라미터 수신
  2. rule_based: type 문자열 의존 → capability 레벨 기반으로 교체
  3. heuristic: 스케일 불일치 → 정규화된 스코어로 개선
"""
import numpy as np


SEV_EQ = {
    1: [],
    2: ["CT","혈액검사기"],
    3: ["CT","수술실"],
    4: ["CT","수술실","중환자실"],
}

def req_level(severity: int) -> int:
    return 3 if severity >= 4 else 2 if severity >= 2 else 1

def eq_score(severity: int, h: dict) -> float:
    req = SEV_EQ.get(severity, [])
    if not req: return 1.0
    return sum(1 for e in req if e in h.get('equipment', [])) / len(req)


class HospitalRouters:
    def __init__(self, hospitals):
        self.hospitals = hospitals

    # ── [수정 ①] nearest: travel_times 파라미터 수신 ──
    def nearest_strategy(self, patient_pos, travel_times=None) -> int:
        """
        travel_times: 각 병원까지 이동시간(초) 리스트.
        없으면 유클리드 거리 fallback.
        """
        if travel_times is not None:
            return int(np.argmin(travel_times))
        dists = [np.hypot(patient_pos[0]-h['sumo_x'],
                          patient_pos[1]-h['sumo_y'])
                 for h in self.hospitals]
        return int(np.argmin(dists))

    # ── [수정 ②] rule_based: capability 레벨 기반 ──
    def rule_based_strategy(self, severity: int, travel_times=None) -> int:
        """
        필요 역량(req_level) 이상이고 점유율 0.9 미만인 병원 중
        이동시간이 가장 짧은 곳 선택.
        """
        req = req_level(severity)
        tts = travel_times if travel_times is not None else [1.0]*len(self.hospitals)

        # 역량 충족 + 여유 있는 병원
        eligible = [i for i, h in enumerate(self.hospitals)
                    if h['capability'] >= req and h['occupancy'] < 0.9]

        # 없으면 역량만 충족
        if not eligible:
            eligible = [i for i, h in enumerate(self.hospitals)
                        if h['capability'] >= req]

        # 그래도 없으면 전체
        if not eligible:
            eligible = list(range(len(self.hospitals)))

        return min(eligible, key=lambda i: tts[i])

    # ── [수정 ③] heuristic: 정규화 스코어 ──
    def heuristic_strategy(self, patient_pos, severity: int,
                           travel_times=None) -> int:
        """
        score = α×norm_travel + β×occupancy + γ×(1-eq_score) + δ×cap_gap
        모든 항을 [0,1] 범위로 정규화 후 가중합 → 스케일 불일치 해결.
        """
        tts  = travel_times if travel_times is not None else [
            np.hypot(patient_pos[0]-h['sumo_x'], patient_pos[1]-h['sumo_y'])
            for h in self.hospitals
        ]
        max_tt = max(tts) + 1e-8
        req    = req_level(severity)

        scores = []
        for i, h in enumerate(self.hospitals):
            norm_tt  = tts[i] / max_tt                      # [0,1]
            occ      = h['occupancy']                        # [0,1]
            eq       = 1.0 - eq_score(severity, h)          # 높을수록 부적합
            cap_gap  = max(0, req - h['capability']) / 3.0  # [0,1]

            # 가중치: 이동시간 30%, 점유율 30%, 기기 20%, 역량 20%
            score = 0.30*norm_tt + 0.30*occ + 0.20*eq + 0.20*cap_gap
            scores.append(score)

        return int(np.argmin(scores))
