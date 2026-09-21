#!/usr/bin/env python3
"""
병원 선택 Baseline 및 공통 State 구성 함수

새로운 의사결정 구조:

환자 상태
    +
병원별 예상 이송시간
병원별 점유율
병원별 치료역량
병원별 환자 적합도

-> 병원 선택

DQN과 Baseline 모두 동일한 hospital_info를 사용합니다.
"""

import numpy as np


# ============================================================
# 공통 상수
# ============================================================

MAX_ESTIMATED_TRAVEL_TIME = 600.0


# ============================================================
# 병원 유형 인코딩
# ============================================================

def encode_hospital_type(hospital_type):
    """
    병원 유형을 치료역량 값으로 변환합니다.

    권역응급의료센터 : 1.0
    지역응급의료센터 : 0.8
    일반              : 0.3
    """

    hospital_type = str(hospital_type or "")

    if "권역" in hospital_type:
        return 1.0

    if "지역" in hospital_type:
        return 0.8

    if "일반" in hospital_type:
        return 0.3

    return 0.5


# ============================================================
# 환자-병원 적합도
# ============================================================

def get_hospital_suitability(severity, hospital):
    """
    환자 중증도와 병원 유형의 적합도를 계산합니다.

    중증 환자:
        권역 = 1.0
        지역 = 0.8
        일반 = 0.0

    비중증 환자:
        권역 = 0.8
        지역 = 1.0
        일반 = 0.7
    """

    severity = int(severity)
    hospital_type = str(hospital.get("type", "일반"))

    if severity >= 3:

        if "권역" in hospital_type:
            return 1.0

        if "지역" in hospital_type:
            return 0.8

        if "일반" in hospital_type:
            return 0.0

        return 0.5

    # 경증 / 중등도
    if "지역" in hospital_type:
        return 1.0

    if "권역" in hospital_type:
        return 0.8

    if "일반" in hospital_type:
        return 0.7

    return 0.5


# ============================================================
# 병원 정보 정규화
# ============================================================

def normalize_travel_time(travel_time):
    """
    예상 이송시간을 0~1 범위로 정규화합니다.
    """

    if travel_time is None:
        return 1.0

    try:
        travel_time = float(travel_time)
    except (TypeError, ValueError):
        return 1.0

    travel_time = max(0.0, travel_time)

    return min(
        travel_time / MAX_ESTIMATED_TRAVEL_TIME,
        1.0
    )


def get_hospital_capability(hospital):
    """
    병원 치료역량을 반환합니다.
    """

    if "capability" in hospital:
        try:
            return float(
                np.clip(
                    hospital["capability"],
                    0.0,
                    1.0
                )
            )
        except (TypeError, ValueError):
            pass

    return encode_hospital_type(
        hospital.get("type", "일반")
    )


# ============================================================
# 새로운 State Vector
# ============================================================

def build_state_vector(
    patient_pos,
    severity,
    hospitals
):
    """
    train_agent.py와 동일한 State 구조입니다.

    State:

    [환자 x,
     환자 y,
     중증도,

     병원별 예상 이송시간 N개,

     병원별 점유율 N개,

     병원별 치료역량 N개,

     병원별 환자 적합도 N개]
    """

    patient_x = float(patient_pos[0]) / 10000.0
    patient_y = float(patient_pos[1]) / 10000.0
    severity_value = float(severity) / 4.0

    travel_times = []
    occupancies = []
    capabilities = []
    suitabilities = []

    for hospital in hospitals:

        travel_time = hospital.get(
            "estimated_travel_time",
            hospital.get("travel_time", MAX_ESTIMATED_TRAVEL_TIME)
        )

        travel_times.append(
            normalize_travel_time(travel_time)
        )

        occupancy = float(
            np.clip(
                hospital.get("occupancy", 0.0),
                0.0,
                1.0
            )
        )

        occupancies.append(occupancy)

        capabilities.append(
            get_hospital_capability(hospital)
        )

        suitabilities.append(
            get_hospital_suitability(
                severity,
                hospital
            )
        )

    state = np.asarray(
        [
            patient_x,
            patient_y,
            severity_value,
        ]
        + travel_times
        + occupancies
        + capabilities
        + suitabilities,
        dtype=np.float32
    )

    expected_dim = 3 + len(hospitals) * 4

    if state.size != expected_dim:
        raise ValueError(
            f"State 차원 오류: "
            f"{state.size} != {expected_dim}"
        )

    return state


# ============================================================
# 병원 선택 Baseline
# ============================================================

class HospitalRouters:

    def __init__(self, hospitals):

        self.hospitals = hospitals


    # --------------------------------------------------------
    # 유효 병원
    # --------------------------------------------------------

    def get_valid_actions(
        self,
        severity,
        occupancy_threshold=0.85
    ):
        """
        현재 환자에게 선택 가능한 병원을 반환합니다.

        점유율이 85% 이상이면 우선 제외합니다.
        """

        valid = []

        for i, hospital in enumerate(self.hospitals):

            occupancy = float(
                hospital.get("occupancy", 0.0)
            )

            if occupancy >= occupancy_threshold:
                continue

            valid.append(i)

        return valid


    # --------------------------------------------------------
    # 최단 거리
    # --------------------------------------------------------

    def shortest_distance_strategy(
        self,
        valid_actions=None
    ):
        """
        환자 -> 병원 실제 SUMO route의
        route_length가 가장 짧은 병원을 선택합니다.
        """

        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))

        candidates = [
            i for i in valid_actions
            if self.hospitals[i].get("route_length") is not None
        ]

        if not candidates:
            return self.lowest_occupancy_strategy(
                valid_actions
            )

        return int(
            min(
                candidates,
                key=lambda i:
                float(
                    self.hospitals[i]["route_length"]
                )
            )
        )


    # --------------------------------------------------------
    # 최단 시간
    # --------------------------------------------------------

    def shortest_time_strategy(
        self,
        valid_actions=None
    ):
        """
        예상 이송시간이 가장 짧은 병원을 선택합니다.
        """

        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))

        candidates = [
            i for i in valid_actions
            if self.hospitals[i].get(
                "estimated_travel_time"
            ) is not None
        ]

        if not candidates:
            return self.lowest_occupancy_strategy(
                valid_actions
            )

        return int(
            min(
                candidates,
                key=lambda i:
                float(
                    self.hospitals[i][
                        "estimated_travel_time"
                    ]
                )
            )
        )


    # --------------------------------------------------------
    # 병상 우선
    # --------------------------------------------------------

    def bed_strategy(
        self,
        severity,
        valid_actions=None
    ):
        """
        점유율이 낮으면서 환자에게 적합한 병원을 선택합니다.

        단순히 가장 가까운 병원이 아니라
        병상 여유 + 환자 적합도를 함께 사용합니다.
        """

        if valid_actions is None:
            valid_actions = list(
                range(len(self.hospitals))
            )

        if not valid_actions:
            return self.lowest_occupancy_strategy()

        scores = {}

        for i in valid_actions:

            hospital = self.hospitals[i]

            occupancy = float(
                hospital.get("occupancy", 1.0)
            )

            suitability = get_hospital_suitability(
                severity,
                hospital
            )

            score = (
                occupancy * 0.7
                + (1.0 - suitability) * 0.3
            )

            scores[i] = score

        return int(
            min(
                scores,
                key=scores.get
            )
        )


    # --------------------------------------------------------
    # 치료역량 기반
    # --------------------------------------------------------

    def rule_based_strategy(
        self,
        severity,
        valid_actions=None
    ):
        """
        중증도에 따른 치료역량을 우선 고려합니다.
        """

        if valid_actions is None:
            valid_actions = list(
                range(len(self.hospitals))
            )

        if not valid_actions:
            return self.lowest_occupancy_strategy()

        return int(
            max(
                valid_actions,
                key=lambda i: (
                    get_hospital_suitability(
                        severity,
                        self.hospitals[i]
                    ),
                    -float(
                        self.hospitals[i].get(
                            "occupancy",
                            1.0
                        )
                    )
                )
            )
        )


    # --------------------------------------------------------
    # 종합 휴리스틱
    # --------------------------------------------------------

    def heuristic_strategy(
        self,
        severity,
        valid_actions=None
    ):
        """
        ETA + 병상 + 치료역량/적합도를
        함께 고려하는 휴리스틱입니다.

        낮을수록 좋은 score:

            0.50 * ETA
          + 0.25 * occupancy
          + 0.25 * mismatch
        """

        if valid_actions is None:
            valid_actions = list(
                range(len(self.hospitals))
            )

        if not valid_actions:
            return self.lowest_occupancy_strategy()

        best_idx = valid_actions[0]
        best_score = float("inf")

        for i in valid_actions:

            hospital = self.hospitals[i]

            travel_time = float(
                hospital.get(
                    "estimated_travel_time",
                    MAX_ESTIMATED_TRAVEL_TIME
                )
            )

            eta_score = min(
                travel_time /
                MAX_ESTIMATED_TRAVEL_TIME,
                1.0
            )

            occupancy = float(
                np.clip(
                    hospital.get(
                        "occupancy",
                        1.0
                    ),
                    0.0,
                    1.0
                )
            )

            suitability = get_hospital_suitability(
                severity,
                hospital
            )

            mismatch_score = 1.0 - suitability

            score = (
                0.50 * eta_score
                + 0.25 * occupancy
                + 0.25 * mismatch_score
            )

            if score < best_score:

                best_score = score
                best_idx = i

        return int(best_idx)


    # --------------------------------------------------------
    # 기존 함수명 호환
    # --------------------------------------------------------

    def nearest_strategy(
        self,
        pat_pos=None,
        valid_actions=None
    ):
        """
        기존 코드와의 호환성을 위한 함수.

        실제 평가는 shortest_distance_strategy를
        사용하는 것을 권장합니다.
        """

        return self.shortest_distance_strategy(
            valid_actions
        )


    # --------------------------------------------------------
    # 최저 점유율
    # --------------------------------------------------------

    def lowest_occupancy_strategy(
        self,
        valid_actions=None
    ):
        if valid_actions is None:
            valid_actions = list(
                range(len(self.hospitals))
            )

        if not valid_actions:
            return 0

        return int(
            min(
                valid_actions,
                key=lambda i:
                float(
                    self.hospitals[i].get(
                        "occupancy",
                        1.0
                    )
                )
            )
        )