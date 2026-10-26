"""DQN 학습/평가에서 공통으로 사용하는 보상 함수."""

GOLDEN_REWARD = 10.0
TIME_WEIGHT = 3.0
OVERTIME_WEIGHT = 5.0
MISMATCH_PENALTY = 8.0
REJECTION_PENALTY = 4.0
OCCUPANCY_WEIGHT = 2.0
ROUTE_FAILURE_PENALTY = 15.0


def calculate_reward(
    travel_time,
    severity,
    hospital,
    golden_limit,
    rejection_count=0,
    occupancy_at_selection=0.0,
    route_failed=False,
):
    """병원 선택 결과에 대한 최종 보상 계산.

    - 골든타임 이내: 기본 +10
    - 골든타임에 가까울수록 시간 보상 추가
    - 골든타임 초과: 초과 비율에 따라 음의 보상
    - 중증 환자의 일반 병원 선택: 패널티
    - 병원 거부 발생: 거부 횟수만큼 패널티
    - 선택 당시 병원 점유율: 패널티
    - 경로 생성 실패: 큰 패널티
    """
    travel_time = float(max(0.0, travel_time))
    golden_limit = float(max(1.0, golden_limit))
    rejection_count = int(max(0, rejection_count))
    occupancy_at_selection = float(np_clip(occupancy_at_selection, 0.0, 1.0))

    time_ratio = travel_time / golden_limit

    if time_ratio <= 1.0:
        golden_component = GOLDEN_REWARD
        time_component = -TIME_WEIGHT * time_ratio
    else:
        overtime_ratio = time_ratio - 1.0
        golden_component = -OVERTIME_WEIGHT * min(overtime_ratio, 2.0)
        time_component = -TIME_WEIGHT

    mismatch_component = 0.0
    hospital_type = str(hospital.get("type", "일반"))
    if severity >= 3 and "일반" in hospital_type:
        mismatch_component = -MISMATCH_PENALTY

    rejection_component = -REJECTION_PENALTY * rejection_count
    occupancy_component = -OCCUPANCY_WEIGHT * occupancy_at_selection
    route_component = -ROUTE_FAILURE_PENALTY if route_failed else 0.0

    return float(
        golden_component
        + time_component
        + mismatch_component
        + rejection_component
        + occupancy_component
        + route_component
    )


def np_clip(value, low, high):
    return max(low, min(high, value))
