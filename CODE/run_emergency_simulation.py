## !/usr/bin/env python3
## 라이브러리 임포트 ##
'''
    os : 운영체제 관련 기능 제공
    sys : 파이썬 인터프리터 관련 기능 제공
    pathlib : 파일 경로 관련 기능 제공
    random : 난수 생성 및 랜덤 선택 기능 제공
    numpy : 수치 계산 및 배열 처리 기능 제공
    sumolib : SUMO 네트워크 관련 기능 제공
    traci : SUMO 시뮬레이션과 상호작용하는 TraCI 인터페이스 제공
'''
import os
import sys
from pathlib import Path
import random
import numpy as np
import sumolib
import traci

## 환경 변수 여부 확인 ##
if 'SUMO_HOME' in os.environ:
    # SUMO_HOME 환경 변수에서 tools 경로를 가져와서 sys.path에 추가
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    # sys.path에 tools 경로 추가
    sys.path.append(tools)
else:
    # 환경 변수 X -> 설정 안내 메시지 출력 후 종료
    sys.exit("please declare environment variable 'SUMO_HOME'")     


## 프로젝트 루트 경로 및 SUMO 관련 파일 경로 설정 ##
'''
    PROJECT_ROOT: 프로젝트의 루트 디렉토리 경로
    SUMO_DIR: SUMO 관련 파일들이 위치한 디렉토리 경로
    NET_FILE: SUMO 네트워크 파일 경로
    CFG_FILE: SUMO 설정 파일 경로
    AMBULANCE_ID: 시뮬레이션에서 사용할 구급차 차량 ID
    DISPATCH_STEP: 환자 발생 및 구급차 출동 이벤트가 발생하는 시뮬레이션 스텝
    MAX_STEPS: 시뮬레이션의 최대 스텝 수
'''
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = PROJECT_ROOT / "SUMO" / "강남"
NET_FILE = SUMO_DIR / "osm.net.xml"
CFG_FILE = SUMO_DIR / "osm.sumocfg"
AMBULANCE_ID = "amb_1110108"
DISPATCH_STEP = 50
MAX_STEPS = 1000


## 구급차의 현재 위치에서 실제로 경로가 연결되어 도달 가능한 유효한 도로 Edge ID 찾기 ##
def get_reachable_patient_edge(ambulance_id, net_file):
    # SUMO 네트워크 파일 읽기
    net = sumolib.net.readNet(net_file)

    # 구급차의 현재 도로 Edge ID 가져오기    
    curr_edge = traci.vehicle.getRoadID(ambulance_id)

    if curr_edge.startswith(':'):
        # 현재 Edge가 내부 Edge인 경우, 실제 도로 Edge로 변환
        curr_edge = traci.vehicle.getLaneID(ambulance_id).rsplit('_', 1)[0]     

    # 네트워크 내 다른 엣지들을 순회하며 도달 가능한 엣지 탐색
    for edge in net.getEdges():
        eid = edge.getID()  # Edge ID 가져오기

        # 현재 Edge와 동일하지 않고, 내부 Edge가 아닌 경우에만 경로 탐색 시도
        if not eid.startswith(':') and eid != curr_edge:
            try:
                # TraCI를 통해 실제로 경로 탐색이 가능한지 확인
                route = traci.simulation.findRoute(curr_edge, eid, vType="ambulance")
                if route and len(route.edges) > 0:
                    return eid
            except Exception:
                continue
    return None


## 시뮬레이션 실행 함수 ##
def run_simulation():
    # 압축된 osm.net.xml.gz 파일이 존재하면 해당 파일을 사용하고, 그렇지 않으면 osm.net.xml 파일을 사용
    net_file = NET_FILE.with_suffix(".xml.gz") if NET_FILE.with_suffix(".xml.gz").exists() else NET_FILE

    # 설정 파일과 네트워크 파일이 존재하지 않으면 종료
    if not CFG_FILE.exists():
        sys.exit(f"SUMO 설정 파일을 찾을 수 없습니다: {CFG_FILE}")
    if not net_file.exists():
        sys.exit(f"SUMO 네트워크 파일을 찾을 수 없습니다: {net_file}")

    # SUMO GUI 실행을 위한 바이너리 경로 확인
    '''
        - SUMO GUI 실행
        sumo_binary: SUMO GUI 실행 파일 경로
        sumo_config: SUMO GUI 실행 시 사용할 설정 옵션 리스트
        
        - 임무 상태 초기화
        mission_state: 시뮬레이션 내 구급차 출동 상태를 나타내는 변수 (WAITING, TO_PATIENT, ARRIVED)
        patient_edge: 환자가 발생한 도로 Edge ID를 저장하는 변수
                    - WAITING: 환자 발생 전 상태
                    - TO_PATIENT: 구급차가 환자에게 이동 중인 상태
                    - ARRIVED: 구급차가 환자에게 도착한 상태
        dispatched_ambulance: 출동이 결정된 구급차 ID를 저장하는 변수
    '''
    sumo_binary = sumolib.checkBinary("sumo-gui")
    sumo_config = [sumo_binary, "-c", str(CFG_FILE), "--scale", "0.5", "--no-warnings"]
    mission_state = "WAITING"
    patient_edge = None
    dispatched_ambulance = None

    traci.start(sumo_config)
    print("SUMO GUI가 실행되었습니다.")

    # SUMO 네트워크 파일 읽기 및 유효한 도로 Edge ID 목록 생성
    net = sumolib.net.readNet(str(net_file))
    valid_edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(":")]

    try:
        for step in range(1, MAX_STEPS + 1):
            traci.simulationStep()

            # 구급차가 화면에 존재하면 SUMO GUI 카메라가 해당 차량을 실시간 추적
            if AMBULANCE_ID in traci.vehicle.getIDList():
                # 'View #0'은 SUMO GUI의 기본 뷰 이름입니다.
                traci.gui.trackVehicle("View #0", AMBULANCE_ID)

            
            # 모든 차량 가져오기
            all_vehicles = traci.vehicle.getIDList()
            # 구급차만 필터링
            ambulance_list = [v for v in all_vehicles if 'amb' in v.lower() or 'ambulance' in v.lower()]

            # 10번째 스텝에서까지 구급차가 없으면 테스트용 구급차 생성
            if not ambulance_list and step == 10:
                try:
                    print(f"[Step {step}] 테스트용 구급차를 생성합니다.")
                    # 출발지와 목적지 랜덤 선택
                    s_edge, e_edge = random.choice(valid_edges), random.choice(valid_edges)
                    # 경로 선택
                    route = traci.simulation.findRoute(s_edge, e_edge)

                    # 구급차 생성 및 경로 설정
                    if route and len(route.edges) > 0:
                        test_amb_id = "amb_auto_1"
                        traci.route.add(f"route_{test_amb_id}", route.edges)
                        traci.vehicle.add(test_amb_id, f"route_{test_amb_id}", typeID="DEFAULT_VEHTYPE")
                        traci.vehicle.setColor(test_amb_id, (255, 0, 0, 255))
                        ambulance_list = [test_amb_id]
                except:
                    pass

            # [이벤트 1] 지정된 스텝(DISPATCH_STEP)에 환자 발생 및 가장 가까운 구급차 배정
            if step == DISPATCH_STEP and mission_state == "WAITING":
                if ambulance_list:
                    # 임의의 환자 발생 위치(Edge) 설정
                    patient_edge = random.choice(valid_edges)
                    patient_pos = net.getEdge(patient_edge).getFromNode().getCoord() # 환자 좌표 추정용

                    # 가장 가까운 구급차 찾기
                    best_ambulance = None
                    min_dist = float('inf')

                    for amb_id in ambulance_list:
                        try:
                            amb_pos = traci.vehicle.getPosition(amb_id)
                            # 2차원 유클리드 거리 계산 (구급차 좌표 vs 환자 도로 시작점 좌표)
                            dist = np.sqrt((amb_pos[0] - patient_pos[0])**2 + (amb_pos[1] - patient_pos[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_ambulance = amb_id
                        except:
                            continue

                    if best_ambulance and patient_edge:
                        dispatched_ambulance = best_ambulance
                        traci.vehicle.changeTarget(dispatched_ambulance, patient_edge)
                        try:
                            traci.vehicle.resume(dispatched_ambulance)
                        except:
                            pass
                        mission_state = "TO_PATIENT"
                        print(f"[Step {step}] 환자 발생! 가장 가까운 구급차 [{dispatched_ambulance}] 출동! (거리: {min_dist:.1f}m, 목적지: {patient_edge})")
                    else:
                        print(f"[Step {step}] 출동 가능한 구급차나 환자 위치를 확정하지 못했습니다.")
                else:
                    print(f"[Step {step}] 맵에 대기 중인 구급차가 없습니다.")

            # [추가] 출동이 시작된 구급차가 있다면 카메라가 실시간으로 그 구급차를 추적
            if dispatched_ambulance and dispatched_ambulance in traci.vehicle.getIDList():
                traci.gui.trackVehicle("View #0", dispatched_ambulance)

            # [이벤트 2] 출동한 구급차의 현장 도착 여부 체크
            if mission_state == "TO_PATIENT" and dispatched_ambulance and dispatched_ambulance in traci.vehicle.getIDList():
                current_edge = traci.vehicle.getRoadID(dispatched_ambulance)
                if current_edge.startswith(":"):
                    current_edge = traci.vehicle.getLaneID(dispatched_ambulance).rsplit("_", 1)[0]

                if current_edge == patient_edge:
                    mission_state = "ARRIVED"
                    print(f"[Step {step}] 구급차 [{dispatched_ambulance}]가 환자 현장에 최종 도달했습니다!")
                    break
        else:
            print(f"[Step {MAX_STEPS}] 제한 시간 안에 도착하지 못했습니다.")
    finally:
        traci.close()
        print("시뮬레이션 완료.")


## 메인 함수 실행 ##
if __name__ == "__main__":
    run_simulation()