# -*- coding: utf-8 -*-
"""병원 선택 baseline + DQN 상태 벡터."""

import math

import numpy as np

from reward import get_hospital_suitability


MAX_ESTIMATED_TRAVEL_TIME = 600.0


def _travel(h):
    return float(h.get("estimated_travel_time", MAX_ESTIMATED_TRAVEL_TIME))


def _distance(patient_pos, h):
    if h.get("sumo_x") is None or h.get("sumo_y") is None:
        return float("inf")
    return math.hypot(
        float(patient_pos[0]) - float(h["sumo_x"]),
        float(patient_pos[1]) - float(h["sumo_y"]),
    )


def build_state_vector(patient_pos, ktas, hospitals):
    state = [
        float(patient_pos[0]) / 10000.0,
        float(patient_pos[1]) / 10000.0,
        float(ktas) / 5.0,
    ]

    for h in hospitals:
        state.extend([
            min(_travel(h) / MAX_ESTIMATED_TRAVEL_TIME, 1.0),
            float(h.get("occupancy", 0.0)),
            float(h.get("capability", _capability(h))),
            float(get_hospital_suitability(ktas, h)),
        ])

    return np.asarray(state, dtype=np.float32)


def _capability(h):
    t = str(h.get("type", ""))
    if "권역응급의료센터" in t:
        return 1.0
    if "지역응급의료센터" in t:
        return 0.8
    if "지역응급의료기관" in t:
        return 0.6
    return 0.3


class HospitalRouters:
    def __init__(self, hospitals=None):
        self.hospitals = hospitals or []

    def _get(self, actions):
        return [int(i) for i in actions if 0 <= int(i) < len(self.hospitals)]

    def shortest_distance_strategy(self, actions):
        a = self._get(actions)
        return min(a, key=lambda i: _distance((self._patient_x, self._patient_y), self.hospitals[i]))

    def shortest_time_strategy(self, actions):
        a = self._get(actions)
        return min(a, key=lambda i: _travel(self.hospitals[i]))

    def bed_strategy(self, ktas, actions):
        a = self._get(actions)
        return min(a, key=lambda i: self.hospitals[i].get("occupancy", 1.0))

    def rule_based_strategy(self, ktas, actions):
        a = self._get(actions)
        if not a:
            raise ValueError("유효 action이 없습니다.")
        return max(
            a,
            key=lambda i: (
                get_hospital_suitability(ktas, self.hospitals[i])
                - 0.5 * float(self.hospitals[i].get("occupancy", 1.0))
            ),
        )

    def heuristic_strategy(self, ktas, actions):
        a = self._get(actions)
        if not a:
            raise ValueError("유효 action이 없습니다.")

        def score(i):
            h = self.hospitals[i]
            time_score = 1.0 - min(_travel(h) / MAX_ESTIMATED_TRAVEL_TIME, 1.0)
            bed_score = 1.0 - float(h.get("occupancy", 1.0))
            capability = float(h.get("capability", _capability(h)))
            suitability = get_hospital_suitability(ktas, h)
            return 0.40 * time_score + 0.20 * bed_score + 0.15 * capability + 0.25 * suitability

        return max(a, key=score)

    # 이전 코드와의 호환
    def nearest_strategy(self, patient_pos):
        self._patient_x, self._patient_y = patient_pos[:2]
        return self.shortest_distance_strategy(range(len(self.hospitals)))

    def rul_based_strategy(self, ktas, actions):
        return self.rule_based_strategy(ktas, actions)
