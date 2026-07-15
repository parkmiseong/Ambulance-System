import os
import sys
import asyncio
from fastapi import FastAPI, BackgroundTasks

# ==========================================
# 1. SUMO 환경 변수 확인 및 도구 경로 등록
# ==========================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("환경 변수 'SUMO_HOME'이 정의되지 않았습니다.")

import traci
import sumolib

# ==========================================
# 2. FastAPI 백엔드 설정
# ==========================================
app = FastAPI(title="Emergency Ambulance RL Simulation API")

# 기존에 보유 중인 설정 파일 경로 매핑
SUMO_CFG = r"C:\Users\emaet\Sumo\2026-07-14-09-46-12\osm.sumocfg"
SUMO_BINARY = sumolib.checkBinary('sumo-gui')  # 시각화 없이 학습만 진행하려면 'sumo'로 변경

# 병원 ID와 SUMO 도로망(Edge) ID 매핑 데이터베이스
HOSPITALS_DATABASE = {
    "h1": "edge_hospital_gangnam",
    "h2": "edge_hospital_seocho",
    "h3": "edge_hospital_songpa"
}

# ==========================================
# 3. 강화학습 에이전트 인터페이스 (RL Agent Mock)
# ==========================================
def get_best_hospital_from_rl(ambulance_state: dict, patient_condition: int) -> str:
    """
    [rl_agent.py]의 강화학습 모델을 호출하여 최적의 병원을 선택하는 함수입니다.
    """
    # 현재 구급차 위치 상태(ambulance_state)와 환자 중증도 등을 입력받아
    # 모델 추론(predict) 후 결정된 최적 병원 ID를 반환합니다.
    print(f"[RL Agent] State 입력받음 -> 위치: {ambulance_state['edge']}, 환자상태: {patient_condition}")
    
    # 예시 행동 변환 (실제 코드는 모델 결과 적용)
    best_hospital_id = "h1"
    return best_hospital_id

# ==========================================
# 4. SUMO 실시간 제어 루프 (Socket.IO 제거 버전)
# ==========================================
async def run_sumo_simulation():
    # SUMO 시뮬레이션 엔진 연결 시작
    traci.start([SUMO_BINARY, "-c", SUMO_CFG])
    print("SUMO 시뮬레이션 엔진 구동 시작")

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()  # 시뮬레이션 1스텝 전진 (0.5초 단위)
            
            # 1. 시뮬레이션 내 모든 차량 ID 추출
            active_vehicles = traci.vehicle.getIDList()
            ambulances = [v for v in active_vehicles if traci.vehicle.getTypeID(v) == "ambulance"]
            
            for amb_id in ambulances:
                # 2. 구급차의 현재 물리 상태 데이터 추출 (RL의 State 목적)
                pos = traci.vehicle.getPosition(amb_id)
                current_edge = traci.vehicle.getRoadID(amb_id)
                speed = traci.vehicle.getSpeed(amb_id)
                
                ambulance_state = {
                    "id": amb_id,
                    "position": pos,
                    "edge": current_edge,
                    "speed": speed
                }

                # 3. 특정 이벤트 시점에 강화학습 의사결정 수행 (예: 100번째 스텝일 때)
                if step == 100 and amb_id == "amb_0":
                    # RL 에이전트를 호출하여 목적지 병원 결정
                    recommended_hospital = get_best_hospital_from_rl(ambulance_state, patient_condition=3)
                    target_edge = HOSPITALS_DATABASE[recommended_hospital]
                    
                    # 4. TraCI를 이용한 차량 실시간 목적지 변경 및 경로 재배정
                    try:
                        traci.vehicle.changeTarget(amb_id, target_edge)
                        print(f"[{amb_id}] RL 추천 반영 완료 -> 목적지: {recommended_hospital} ({target_edge})")
                    except traci.TraCIException as ex:
                        print(f"경로 재설정 오류 발생: {ex}")

            step += 1
            await asyncio.sleep(0.01)  # CPU 과점 방지를 위한 비동기 틱 대기
            
    finally:
        traci.close()
        print("SUMO 시뮬레이션 엔진이 안전하게 종료되었습니다.")

# ==========================================
# 5. FastAPI 백엔드 API 엔드포인트
# ==========================================
@app.post("/start-simulation")
async def start_simulation(background_tasks: BackgroundTasks):
    """
    백그라운드 스레드에서 시뮬레이션 엔진을 구동하는 API입니다.
    """
    background_tasks.add_task(run_sumo_simulation)
    return {"status": "시뮬레이션이 백그라운드에서 정상 시작되었습니다."}

@app.get("/get_hospitals")
async def get_hospitals():
    """
    FastAPI를 통해 병원의 실시간 가용 병상 자원 정보를 수신하는 프로토콜입니다.
    """
    return [
        {"id": "h1", "status": "available", "beds": 5},
        {"id": "h2", "status": "full", "beds": 0},
        {"id": "h3", "status": "available", "beds": 2}
    ]