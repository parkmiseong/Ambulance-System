'''
비교 대상
- 최단 거리
- 최단 시간
- 병상 기반
- 치료역량 기반
- 종합 휴리스틱
- DQN
'''

from reward import get_hospital_suitability as get_reard_suitability
import numpy as np

## 상수 ##
MAX_ESTIMATED_TRAVEL_TIME = 600.0
OCCUPANCY_THRESHOLD = 0.85
HOSPITAL_TYPES = ("권역응급의료센터", "지역응급의료센터", "지역응급의료기관", "응급실운영신고기관")
HOSPITAL_CAPABILITY = {
    "권역응급의료센터" : 1.0,
    "지역응급의료센터" : 0.8,
    "지역응급의료기관" : 0.6,
    "응급실운영신고기관" : 0.3
}
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

## 함수 ##
# 병원 유형 표준화 #
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

# 병원 유형을 값으로 변환 #
def encode_hospital_type(hospital):
    if isinstance(hospital, str):
        hospital_type = normalize_hospital_type(hospital)
    else:
        hospital_type = normalize_hospital_type(hospital.get("type", hospital.get("hospital_type", "")))

    return float(HOSPITAL_CAPABILITY.get(hospital_type, 0.3))

# 적합도 계산 #
def get_hospital_suitability(ktas, hospital):
    try:
        ktas = int(ktas)
    except(TypeError, ValueError):
        ktas = 3

    ktas = int(max(1, min(5, ktas)))

    hospital_type = normalize_hospital_type(hospital.get("type", hospital.get("hospital_type", "")))

    return float(KTAS_HOSPITAL_SUITABILITY[ktas][hospital_type])

# 이동시간 정규화 #
def normalize_travel_time(travel_time):
    try:
        travel_time = float(travel_time)
    except(TypeError, ValueError):
        return 1.0

    travel_time = max(0.0, travel_time)
    return min(travel_time/MAX_ESTIMATED_TRAVEL_TIME, 1.0)

# 병원 치료 역량 #
def get_hospital_capability(hospital):
    if "capability" in hospital:
        try:
            return float(max(0.0, min(1.0, float(hospital["capability"]))))
        except(TypeError, ValueError):
            pass

    return encode_hospital_type(hospital)

# DQN state 벡터 #
# 구조 : 환자 정보(x, y, ktas), 병원 정보(예상 이동시간, 병상 점유율, 치료역량, 적합도)
def build_state_vector(patient_pos, ktas, hospitals):
    # 환자 위치
    patient_x = float(patient_pos[0]) / 10000.0
    patient_y = float(patient_pos[1]) / 10000.0

    # KTAS
    try:
        ktas = int(ktas)
    except(TypeError, ValueError):
        ktas = 3

    ktas = max(1, min(5, ktas))
    urgency = (6.0 - ktas) / 5.0

    # State 배열
    state = [patient_x, patient_y, urgency]

    # 병원 정보 배열
    travel_times = []
    occupancies = []
    capabilities = []
    suitabilities = []

    # 병원 정보
    for hospital in hospitals:
        travel_time = hospital.get("estimated_travel_time", hospital.get("travel_time", MAX_ESTIMATED_TRAVEL_TIME))
        travel_times.append(normalize_travel_time(travel_time))

    # 점유율
    try:
        occupancy = float(hospital.get("occupancy", 0.0))
    except(TypeError, ValueError):
        occupancy = 0.0

    occupancy = float(max(0.0, min(1.0, occupancy)))
    occupancies.append(occupancy)

    capabilities.append(get_hospital_capability(hospital))          # 치료 역량
    suitabilities.append(get_hospital_suitability(ktas, hospital))  # 적합도

    # State 결합
    state.extend(travel_times)
    state.extend(occupancies)
    state.extend(capabilities)
    state.extend(suitabilities)
    state = np.asarray(state, dtype=np.float32)

    # State 차원 검증
    expected_dim = 3 + len(hospital) * 4
    if state.size != expected_dim:
        raise ValueError("State 차원 오류" + state.size + "!=" + expected_dim)

    return state

# 병원 경로 #
class HospitalRouters:
    def __init__(self, hospitals):
        self.hospitals = hospitals

    # 현재 선택 가능한 병원 반환
    def get_valid_actions(self, ktas=None):
        valid_actions = []
        for i, hospital in enumerate(self.hospitals):
            try:
                occupancy = float(hospital.get("occupancy",))
            except(TypeError, ValueError):
                occupancy = 0.0

            if occupancy >= OCCUPANCY_THRESHOLD:
                continue

            valid_actions.append(i)

        return valid_actions

    # 최단 거리 #
    def shortest_distance_strategy(self, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))

        if not valid_actions:
            return 0

        def get_distance(index):
            hospital = self.hospitals[index]
            value = hospital.get("route_length", hospital.get("distance", float("inf")))

            try:
                return float(value)
            except(TypeError, ValueError):
                return float("inf")

        return int(min(valid_actions, key=get_distance))

    # 최단 시간 #
    def shortest_time_strategy(self, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))

        if not valid_actions:
            return 0

        def get_time(index):
            hospital = self.hospitals[index]
            value = hospital.get("estimated_travel_time", hospital.get("travel_time", MAX_ESTIMATED_TRAVEL_TIME))

            try:
                return float(value)
            except(TypeError, ValueError):
                return MAX_ESTIMATED_TRAVEL_TIME

        return int(min(valid_actions, key=get_time))

    # 혼잡도 기반 #
    def bed_strategy(self, ktas=None, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))
        
        if not valid_actions:
            return 0

        def bed_score(index):
            hospital = self.hospitals[index]

            try:
                occupancy = float(hospital.get("occupancy", 1.0))
            except(TypeError, ValueError):
                occupancy = 1.0

            suitability = 0.0

            if ktas is not None:
                suitability = get_hospital_suitability(ktas, hospital)

            return (occupancy - suitability)

        return int(min(valid_actions, key=bed_score))

    # 치료역량 기반 : 우선순위(KTAS -> 치료역량 -> 병상 점유율) #
    def rul_based_strategy(self, ktas, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))
                
        if not valid_actions:
            return 0

        def capability_score(index):
            hospital = self.hospitals[index]

            suitability = get_hospital_suitability(ktas, hospital)
            capability = get_hospital_capability(hospital)

            try:
                occupancy = float(hospital.get("occupancy", 1.0))
            except(TypeError, ValueError):
                occupancy = 1.0

            return (suitability, capability - occupancy)

        return int(max(valid_actions, key=capability_score))

    # 종합 휴리스틱 : 이동시간 + 병상 + 적합도 -> Score가 낮을수록 좋은 병원 #
    def heuristic_strategy(self, ktas, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))
                        
        if not valid_actions:
            return 0

        best_index = valid_actions[0]
        best_score = float("inf")

        for index in valid_actions:
            hospital = self.hospitals[index]

            # 이동 시간
            travel_time = hospital.get("estimated_travel_time", hospital.get("travel_time", MAX_ESTIMATED_TRAVEL_TIME))
            travel_score = normalize_travel_time(travel_time)

            # 병상
            try:
                occupancy = float(hospital.get("occupancy", 1.0))
            except(TypeError, ValueError):
                occupancy = 1.0

            occupancy = float(max(0.0, min(1.0, occupancy)))

            # 적합도
            suitability = get_hospital_suitability(ktas, hospital)
            mismatch_score = 1.0 - suitability

            # 최종 점수
            score = 0.50 * travel_score + 0.25 * occupancy + 0.25 * mismatch_score

            if score < best_score:
                best_score = score
                best_index = index

        return int(best_index)

    # 기존 nearest_strategy 호환 #
    def nearest_strategy(self, pat_pos=None, valid_actions=None):
        return self.shortest_distance_strategy(valid_actions)

    # 최저 점유율 #
    def lowest_occupancy_strategy(self, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(len(self.hospitals)))

        if not valid_actions:
            return 0

        def get_occupancy(index):
            try:
                return float(self.hospitals[index].get("occupancy", 1.0))
            except(TypeError, ValueError):
                return 1.0

        return int(min(valid_actions, key=get_occupancy))