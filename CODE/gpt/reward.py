"""
DQN 학습/평가에서 공통으로 사용하는 보상 함수.

보상 구성
---------
1. 골든타임 달성 보상
2. 이동시간 패널티
3. 골든타임 초과 패널티
4. 환자-병원 치료역량 불일치 패널티
5. 병원 거부 패널티
6. 병원 점유율 패널티
7. 경로 생성/이송 실패 패널티
"""


# ============================================================
# 보상 가중치
# ============================================================

# 골든타임 이내 도착 시 기본 보상
GOLDEN_REWARD = 10.0

# 골든타임 이내 이동시간 패널티
TIME_WEIGHT = 3.0

# 골든타임 초과 패널티
OVERTIME_WEIGHT = 5.0

# 중증 환자와 병원 치료역량 불일치 패널티
MISMATCH_PENALTY = 8.0

# 병원 거부 1회당 패널티
REJECTION_PENALTY = 4.0

# 병원 점유율 패널티
OCCUPANCY_WEIGHT = 2.0

# 경로 생성 또는 미션 실패 패널티
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
# 입력값 제한
# ============================================================

def np_clip(value, low, high):
    """
    값을 [low, high] 범위로 제한합니다.
    """

    return max(
        low,
        min(high, value)
    )


# ============================================================
# 치료역량 불일치 판단
# ============================================================

def calculate_mismatch_penalty(
    severity,
    hospital,
):
    """
    환자 중증도와 병원 치료역량의 불일치 정도를 계산합니다.

    현재 프로젝트의 병원 type을 기준으로 판단합니다.

    현재 기준
    ----------
    severity >= 3
        일반 병원 → 불일치

    그 외
        별도 패널티 없음

    Returns
    -------
    float
        패널티 값
    """

    hospital_type = str(
        hospital.get(
            "type",
            "일반"
        )
    )

    # 중증 환자가 일반 병원으로 배정된 경우
    if severity >= 3 and "일반" in hospital_type:

        return MISMATCH_PENALTY

    return 0.0


# ============================================================
# 골든타임 보상
# ============================================================

def calculate_time_component(
    travel_time,
    golden_limit,
):
    """
    이동시간과 골든타임을 기반으로
    시간 관련 보상을 계산합니다.

    골든타임 이내
        +GOLDEN_REWARD
        - 시간 비율 패널티

    골든타임 초과
        - 초과 비율 패널티
        - 기본 시간 패널티

    Returns
    -------
    tuple
        (golden_component, time_component)
    """

    travel_time = float(
        max(
            0.0,
            travel_time
        )
    )

    golden_limit = float(
        max(
            1.0,
            golden_limit
        )
    )

    time_ratio = (
        travel_time
        / golden_limit
    )

    # ========================================================
    # 골든타임 이내
    # ========================================================

    if time_ratio <= 1.0:

        golden_component = (
            GOLDEN_REWARD
        )

        time_component = (
            -TIME_WEIGHT
            * time_ratio
        )

        return (
            golden_component,
            time_component
        )

    # ========================================================
    # 골든타임 초과
    # ========================================================

    overtime_ratio = (
        time_ratio - 1.0
    )

    # 지나치게 큰 패널티가 발생하지 않도록 제한
    capped_overtime = min(
        overtime_ratio,
        2.0
    )

    golden_component = (
        -OVERTIME_WEIGHT
        * capped_overtime
    )

    # 골든타임을 초과한 경우
    # 추가 시간 패널티
    time_component = (
        -TIME_WEIGHT
    )

    return (
        golden_component,
        time_component
    )


# ============================================================
# 보상 계산
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
        환자 발생부터 병원 도착 또는 종료까지의 실제 시간

    severity : int
        환자 중증도
        1 ~ 4

    hospital : dict
        선택된 병원 정보

    golden_limit : float
        해당 환자의 골든타임

    rejection_count : int
        병원 선택 과정에서 발생한 거부 횟수

    occupancy_at_selection : float
        병원 선택 당시의 점유율

    route_failed : bool
        경로 생성 또는 실제 이송 실패 여부

    Returns
    -------
    float
        최종 보상값
    """

    # ========================================================
    # 1. 입력값 정규화
    # ========================================================

    travel_time = float(
        max(
            0.0,
            travel_time
        )
    )

    golden_limit = float(
        max(
            1.0,
            golden_limit
        )
    )

    severity = int(
        max(
            1,
            min(
                4,
                severity
            )
        )
    )

    rejection_count = int(
        max(
            0,
            rejection_count
        )
    )

    occupancy_at_selection = np_clip(
        float(
            occupancy_at_selection
        ),
        0.0,
        1.0
    )

    # ========================================================
    # 2. 골든타임 + 이동시간
    # ========================================================

    (
        golden_component,
        time_component
    ) = calculate_time_component(
        travel_time,
        golden_limit
    )

    # ========================================================
    # 3. 치료역량 불일치
    # ========================================================

    mismatch_value = (
        calculate_mismatch_penalty(
            severity,
            hospital
        )
    )

    mismatch_component = (
        -mismatch_value
    )

    # ========================================================
    # 4. 병원 거부
    # ========================================================

    rejection_component = (
        -REJECTION_PENALTY
        * rejection_count
    )

    # ========================================================
    # 5. 병원 점유율
    # ========================================================

    occupancy_component = (
        -OCCUPANCY_WEIGHT
        * occupancy_at_selection
    )

    # ========================================================
    # 6. 경로 실패
    # ========================================================

    if route_failed:

        route_component = (
            -ROUTE_FAILURE_PENALTY
        )

    else:

        route_component = 0.0

    # ========================================================
    # 7. 최종 보상
    # ========================================================

    total_reward = (
        golden_component
        + time_component
        + mismatch_component
        + rejection_component
        + occupancy_component
        + route_component
    )

    return float(
        total_reward
    )


# ============================================================
# 보상 상세 정보
# ============================================================

def calculate_reward_components(
    travel_time,
    severity,
    hospital,
    golden_limit,
    rejection_count=0,
    occupancy_at_selection=0.0,
    route_failed=False,
):
    """
    보상값뿐 아니라 각 보상 구성요소를 반환합니다.

    연구 결과 분석 및 디버깅에 사용합니다.

    Returns
    -------
    dict
        {
            "total": ...,
            "golden": ...,
            "travel": ...,
            "mismatch": ...,
            "rejection": ...,
            "occupancy": ...,
            "route_failure": ...
        }
    """

    travel_time = float(
        max(
            0.0,
            travel_time
        )
    )

    golden_limit = float(
        max(
            1.0,
            golden_limit
        )
    )

    severity = int(
        max(
            1,
            min(
                4,
                severity
            )
        )
    )

    rejection_count = int(
        max(
            0,
            rejection_count
        )
    )

    occupancy_at_selection = np_clip(
        float(
            occupancy_at_selection
        ),
        0.0,
        1.0
    )

    (
        golden_component,
        time_component
    ) = calculate_time_component(
        travel_time,
        golden_limit
    )

    mismatch_component = (
        -calculate_mismatch_penalty(
            severity,
            hospital
        )
    )

    rejection_component = (
        -REJECTION_PENALTY
        * rejection_count
    )

    occupancy_component = (
        -OCCUPANCY_WEIGHT
        * occupancy_at_selection
    )

    route_component = (
        -ROUTE_FAILURE_PENALTY
        if route_failed
        else 0.0
    )

    total_reward = (
        golden_component
        + time_component
        + mismatch_component
        + rejection_component
        + occupancy_component
        + route_component
    )

    return {
        "total": float(total_reward),

        "golden": float(
            golden_component
        ),

        "travel": float(
            time_component
        ),

        "mismatch": float(
            mismatch_component
        ),

        "rejection": float(
            rejection_component
        ),

        "occupancy": float(
            occupancy_component
        ),

        "route_failure": float(
            route_component
        ),
    }