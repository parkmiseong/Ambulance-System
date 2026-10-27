# -*- coding: utf-8 -*-
"""환자/병원 상태 확장 설계: 현재 DQN 43차원과 별개로 독립 테스트 가능.

미관측 의료정보를 KTAS만으로 채워 넣지 않는다. 누락은 missing으로 표시한다.
향후 자료가 확보되면 이 스키마를 사용하여 상태 벡터 v2를 별도 학습한다.
"""
PATIENT_FIELDS = {
    "pre_ktas": "현장 추정 중증도(1~5), 현재 존재",
    "age_group": "연령군, 데이터 확보 필요",
    "consciousness": "의식 상태, 데이터 확보 필요",
    "respiratory_risk": "호흡 관련 위험 신호, 데이터 확보 필요",
    "trauma_flag": "외상 여부, 데이터 확보 필요",
    "required_specialty": "필요 진료과/처치, 데이터 확보 필요",
}
HOSPITAL_FIELDS = {
    "type": "응급의료기관 분류, 현재 존재",
    "occupancy": "현재는 모의 점유율, 실제 가용병상 아님",
    "estimated_travel_time": "SUMO 경로 추정 시간, 현재 존재",
    "capability": "기관 유형 기반 추정치, 실제 처치 가능 여부 아님",
    "available_beds": "실시간 가용병상, 데이터 확보 필요",
    "specialty_availability": "전문 진료/수술 가능 여부, 데이터 확보 필요",
    "acceptance_status": "현재 수용 가능 여부, 데이터 확보 필요",
}

def analyze_patient(patient):
    return {k: {"value": patient.get(k), "observed": patient.get(k) is not None,
                "description": desc} for k, desc in PATIENT_FIELDS.items()}

def analyze_hospital(hospital):
    return {k: {"value": hospital.get(k), "observed": hospital.get(k) is not None,
                "description": desc} for k, desc in HOSPITAL_FIELDS.items()}
