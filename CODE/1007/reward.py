'''
DQN 학습/평가에 사용할 보상 함수

구성
- 골든타임 달성 보상 
- 골든타임 초과 패널티
- 이동시간 패널티
- 환자-병원 치료역량 불일치 패널티
- 병원 거부 패널티
- 병원 점유율 패널티
- 경로 생성/이송 실패 패널티
'''

## 상수 ##
GOLDEN_REWARD = 10.0            # 골든타임 달성
OVERTIME_WEIGHT = 5.0           # 골든타임 초과
TIME_WEIGHT = 3.0               # 이동시간
REJECTION_PENALTY = 4.0         # 병원 거부
OCCUPANCY_WEIGHT = 2.0          # 병원 점유율
ROUTE_FAILURE_PENALTY = 15.0    # 경로 생성/이송 실패

# 골든 타임 기준 : KTAS 기준(1 소생 -> 5 비응급) #
GOLDEN_TIME_LIMITS = {
    1 : 120.0,
    2 : 180.0,
    3 : 240.0,
    4 : 300.0,
    5 : 360.0
}

# 병원 분류 #
HOSPITAL_TYPES = ("권역응급의료센터", "지역응급의료센터", "지역응급의료기관", "응급실운영신고기관")

# KTAS - 병원 적합도 #
KTAS_HOSPITAL_SUITABILITY = {
    # KTAS 1 : 소생
    1: {
        "권역응급의료센터": 1.0,
        "지역응급의료센터": 0.7,
        "지역응급의료기관": 0.0,
        "응급실운영신고기관": 0.0,
    },

    # KTAS 2 : 긴급
    2: {
        "권역응급의료센터": 1.0,
        "지역응급의료센터": 0.9,
        "지역응급의료기관": 0.2,
        "응급실운영신고기관": 0.0,
    },

    # KTAS 3 : 응급
    3: {
        "권역응급의료센터": 1.0,
        "지역응급의료센터": 1.0,
        "지역응급의료기관": 0.6,
        "응급실운영신고기관": 0.2,
    },

    # KTAS 4 : 준응급
    4: {
        "권역응급의료센터": 0.8,
        "지역응급의료센터": 0.9,
        "지역응급의료기관": 1.0,
        "응급실운영신고기관": 0.8,
    },

    # KTAS 5 : 비응급
    5: {
        "권역응급의료센터": 0.7,
        "지역응급의료센터": 0.8,
        "지역응급의료기관": 0.9,
        "응급실운영신고기관": 1.0,
    },
}

# 역량 불일치 패널티 #
KTAS_MISMATCH_PENALTIES = {
    # KTAS 1
    1: {
        "권역응급의료센터": 0.0,
        "지역응급의료센터": 2.0,
        "지역응급의료기관": 8.0,
        "응급실운영신고기관": 12.0,
    },

    # KTAS 2
    2: {
        "권역응급의료센터": 0.0,
        "지역응급의료센터": 1.0,
        "지역응급의료기관": 6.0,
        "응급실운영신고기관": 10.0,
    },

    # KTAS 3
    3: {
        "권역응급의료센터": 0.0,
        "지역응급의료센터": 0.0,
        "지역응급의료기관": 3.0,
        "응급실운영신고기관": 6.0,
    },

    # KTAS 4
    4: {
        "권역응급의료센터": 0.0,
        "지역응급의료센터": 0.0,
        "지역응급의료기관": 0.0,
        "응급실운영신고기관": 1.0,
    },

    # KTAS 5
    5: {
        "권역응급의료센터": 0.0,
        "지역응급의료센터": 0.0,
        "지역응급의료기관": 0.0,
        "응급실운영신고기관": 0.0,
    },
}

## 함수 ##
# 기관 유형 판별 #
def normalize_hospital_type(hospital_type):
    hospital_type = str(hospital_type or "").strip()

    if "권역응급의료센터" in hospital_type:
        return "권역응급의료센터"
    elif "지역응급의료센터" in hospital_type:
        return "지역응급의료센터"
    elif "지역응급의료기관" in hospital_type:
        return "지역응급의료기관"
    else:
        return "응급실운영신고기관"

# 병원 적합도 : 1.0 매우 적합 ~ 0.0 부적합 #
def get_hospital_suitability(ktas, hospital):
    try:
        ktas = int(ktas)
    except(TypeError, ValueError):
        ktas = 3

    ktas = int(max(1, min(5, ktas)))

    hospital_type = normalize_hospital_type(hospital.get("type", ""))

    return float(KTAS_HOSPITAL_SUITABILITY[ktas][hospital_type])

# 보상 계산 #
def calculate_reward(travel_time, ktas, hospital, golden_limit, rejection_count=0, occupancy = 0.0, route_failed=False):
    # 입력값 정규화
    travel_time = float(max(0.0, travel_time))
    golden_limit = float(max(1.0, golden_limit))
    rejection_count = int(max(0, rejection_count))
    occupancy = float(max(0.0, min(1.0, occupancy)))
    try:
        ktas = int(ktas)
    except(TypeError, ValueError):
        ktas = 3

    ktas = int(max(1, min(5, ktas)))

    # 골든 타임 보상 및 시간 패널티
    time_ratio = travel_time / golden_limit

    if time_ratio <= 1.0:
        golden_component = GOLDEN_REWARD
        time_component = -TIME_WEIGHT * time_ratio
    else:
        overtime_ratio = time_ratio - 1.0
        capped_overtime = min(overtime_ratio, 2.0)
        golden_component = -OVERTIME_WEIGHT * capped_overtime
        time_component = - TIME_WEIGHT

    # 역량 불일치 패널티
    hospital_type = normalize_hospital_type(hospital.get("type", ""))
    mismatch_value = KTAS_MISMATCH_PENALTIES[ktas][hospital_type]
    mismatch_component = -mismatch_value

    rejection_component = -REJECTION_PENALTY * rejection_count      # 병원 거부 패널티
    occupancy_component = -OCCUPANCY_WEIGHT * occupancy             # 병원 점유율 패널티

    # 경로 실패 패널티
    if route_failed:
        route_component = -ROUTE_FAILURE_PENALTY
    else:
        route_component = 0.0

    # 최종 보상
    total_reward = golden_component + time_component + mismatch_component + rejection_component + occupancy_component + route_component

    return float(total_reward)