"""
DQN 학습/평가에서 공통으로 사용하는 보상 함수.

보상 구성
---------
1. 골든타임 보상
2. 이동시간 패널티
3. 골든타임 초과 패널티
4. 중증 환자 + 일반 병원 불일치 패널티
5. 병원 거부 패널티
6. 병원 점유율 패널티
7. 경로 생성 실패 패널티
"""


# ============================================================
# 보상 가중치
# ============================================================

# 골든타임 이내에 도착했을 때 기본 보상
GOLDEN_REWARD = 10.0

# 골든타임 이내 이동시간에 대한 가중치
TIME_WEIGHT = 3.0

# 골든타임 초과에 대한 가중치
OVERTIME_WEIGHT = 5.0

# 중증 환자가 일반 병원으로 배정되었을 때의 패널티
MISMATCH_PENALTY = 8.0

# 병원 거부 1회당 패널티
REJECTION_PENALTY = 4.0

# 병원 점유율에 대한 패널티 가중치
OCCUPANCY_WEIGHT = 2.0

# 경로 생성 실패 패널티
ROUTE_FAILURE_PENALTY = 15.0


# ============================================================
# 골든타임 기준
# ============================================================
#
# severity
# 4 : 가장 중증
# 3 : 중증
# 2 : 중간
# 1 : 경증
#
# 단위: 초
# ============================================================

GOLDEN_TIME_LIMITS = {
    4: 120.0,
    3: 180.0,
    2: 240.0,
    1: 300.0,
}


# ============================================================
# 보상 계산 함수
# ============================================================

def calculate_reward(
    travel_time,
    severity,
    hospital,
    golden_limit,
    rejection_count=0,
    occupancy_at_selection=0.0,
    route_failed=False,
):
    """
    병원 선택 결과에 대한 최종 보상을 계산합니다.

    Parameters
    ----------
    travel_time : float
        환자 발생부터 병원 도착 또는 종료까지 걸린 실제 시간(초)

    severity : int
        환자 중증도
        1 ~ 4

    hospital : dict
        선택된 병원 정보

        예:
        {
            "id": "...",
            "name": "...",
            "type": "...",
            "occupancy": ...
        }

    golden_limit : float
        해당 환자 중증도의 골든타임(초)

    rejection_count : int
        병원 선택 과정에서 발생한 거부 횟수

    occupancy_at_selection : float
        DQN이 병원을 선택했을 당시의 병원 점유율

    route_failed : bool
        병원까지의 경로 생성 또는 미션 수행에 실패했는지 여부


    Returns
    -------
    float
        최종 보상값


    보상 구성
    ---------
    골든타임 이내:

        +10
        - 시간 비율에 따른 패널티

    골든타임 초과:

        - 초과 시간에 따른 패널티
        - 추가 시간 패널티

    추가 패널티:

        중증 환자 + 일반 병원
        병원 거부
        병원 점유율
        경로 생성 실패
    """

    # ========================================================
    # 1. 입력값 정규화
    # ========================================================

    travel_time = float(
        max(0.0, travel_time)
    )

    golden_limit = float(
        max(1.0, golden_limit)
    )

    rejection_count = int(
        max(0, rejection_count)
    )

    occupancy_at_selection = np_clip(
        occupancy_at_selection,
        0.0,
        1.0
    )

    # ========================================================
    # 2. 골든타임 대비 이동시간 비율
    # ========================================================

    time_ratio = (
        travel_time / golden_limit
    )

    # ========================================================
    # 3. 골든타임 보상 / 초과 패널티
    # ========================================================

    if time_ratio <= 1.0:

        # --------------------------------------------
        # 골든타임 이내 도착
        # --------------------------------------------

        golden_component = GOLDEN_REWARD

        # 골든타임에 가까울수록 시간 패널티 증가
        #
        # 예:
        # travel_time = 60
        # golden_limit = 120
        #
        # time_ratio = 0.5
        # time_component = -1.5
        #
        # 따라서 빠르게 도착할수록 보상이 높습니다.
        # --------------------------------------------

        time_component = (
            -TIME_WEIGHT * time_ratio
        )

    else:

        # --------------------------------------------
        # 골든타임 초과
        # --------------------------------------------

        overtime_ratio = (
            time_ratio - 1.0
        )

        # 초과 비율이 지나치게 커지는 것을 방지
        capped_overtime = min(
            overtime_ratio,
            2.0
        )

        golden_component = (
            -OVERTIME_WEIGHT
            * capped_overtime
        )

        # 골든타임 초과 시 추가 시간 패널티
        time_component = -TIME_WEIGHT

    # ========================================================
    # 4. 병원 유형 불일치 패널티
    # ========================================================

    mismatch_component = 0.0

    hospital_type = str(
        hospital.get(
            "type",
            "일반"
        )
    )

    # 중증 환자(severity 3 이상)가
    # 일반 병원으로 선택된 경우
    if severity >= 3 and "일반" in hospital_type:

        mismatch_component = (
            -MISMATCH_PENALTY
        )

    # ========================================================
    # 5. 병원 거부 패널티
    # ========================================================

    rejection_component = (
        -REJECTION_PENALTY
        * rejection_count
    )

    # ========================================================
    # 6. 병원 점유율 패널티
    # ========================================================
    #
    # 반드시 "선택 당시"의 점유율을 사용합니다.
    #
    # 예:
    # occupancy = 0.3
    # → -0.6
    #
    # occupancy = 0.8
    # → -1.6
    # ========================================================

    occupancy_component = (
        -OCCUPANCY_WEIGHT
        * occupancy_at_selection
    )

    # ========================================================
    # 7. 경로 실패 패널티
    # ========================================================

    if route_failed:

        route_component = (
            -ROUTE_FAILURE_PENALTY
        )

    else:

        route_component = 0.0

    # ========================================================
    # 8. 최종 보상
    # ========================================================

    total_reward = (
        golden_component
        + time_component
        + mismatch_component
        + rejection_component
        + occupancy_component
        + route_component
    )

    return float(total_reward)


# ============================================================
# 값 제한 함수
# ============================================================

def np_clip(value, low, high):
    """
    외부 라이브러리 없이 값을 [low, high] 범위로 제한합니다.

    예:
        np_clip(1.2, 0.0, 1.0)
        -> 1.0

        np_clip(-0.2, 0.0, 1.0)
        -> 0.0

        np_clip(0.5, 0.0, 1.0)
        -> 0.5
    """

    return max(
        low,
        min(high, value)
    )