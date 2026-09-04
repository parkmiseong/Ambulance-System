import os
import sys
from pathlib import Path
import random
import numpy as np

## 환경 변수 여부 확인 ##
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')          # SUMO_HOME 환경 변수에서 tools 경로를 가져와서 sys.path에 추가
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")     # 환경 변수 X -> 설정 안내 메시지 출력 후 종료

import sumolib
import traci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = PROJECT_ROOT / "SUMO" / "강남"
NET_FILE = SUMO_DIR / "osm.net.xml"
CFG_FILE = SUMO_DIR / "osm.sumocfg"
AMBULANCE_ID = "amb_1110108"
DISPATCH_STEP = 50
MAX_STEPS = 1000


## 구급차의 현재 위치에서 실제로 경로가 연결되어 도달 가능한 유효한 도로 Edge ID 찾기 ##
def get_reachable_patient_edge(ambulance_id, net_file):
    net = sumolib.net.readNet(net_file)     # SUMO 네트워크 파일 읽기
    curr_edge = traci.vehicle.getRoadID(ambulance_id)       # 구급차의 현재 도로 Edge ID 가져오기
    if curr_edge.startswith(':'):
        curr_edge = traci.vehicle.getLaneID(ambulance_id).rsplit('_', 1)[0]     # 현재 Edge가 내부 Edge인 경우, 실제 도로 Edge로 변환

    # 네트워크 내 다른 엣지들을 순회하며 도달 가능한 엣지 탐색
    for edge in net.getEdges():
        eid = edge.getID()
        if not eid.startswith(':') and eid != curr_edge:
            try:
                # TraCI를 통해 실제로 경로 탐색이 가능한지 확인
                route = traci.simulation.findRoute(curr_edge, eid, vType="ambulance")
                if route and len(route.edges) > 0:
                    return eid
            except Exception:
                continue
    return None

def run_simulation():
    net_file = NET_FILE.with_suffix(".xml.gz") if NET_FILE.with_suffix(".xml.gz").exists() else NET_FILE
    if not CFG_FILE.exists():
        sys.exit(f"SUMO 설정 파일을 찾을 수 없습니다: {CFG_FILE}")
    if not net_file.exists():
        sys.exit(f"SUMO 네트워크 파일을 찾을 수 없습니다: {net_file}")

    sumo_binary = sumolib.checkBinary("sumo-gui")
    sumo_config = [sumo_binary, "-c", str(CFG_FILE), "--scale", "0.5", "--no-warnings"]
    mission_state = "WAITING"
    patient_edge = None
    dispatched_ambulance = None  # 출동이 결정된 구급차 ID를 저장할 변수

    traci.start(sumo_config)
    print("SUMO GUI가 실행되었습니다.")

    net = sumolib.net.readNet(str(net_file))
    valid_edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(":")]

    try:
        for step in range(1, MAX_STEPS + 1):
            traci.simulationStep()

            # [추가] 구급차가 화면에 존재하면 SUMO GUI 카메라가 해당 차량을 실시간으로 추적하도록 설정
            if AMBULANCE_ID in traci.vehicle.getIDList():
                # 'View #0'은 SUMO GUI의 기본 뷰 이름입니다.
                traci.gui.trackVehicle("View #0", AMBULANCE_ID)
                # 첫 추적 시 카메라 줌인(확대)을 적절하게 맞추고 싶다면 아래 주석을 해제하세요 (값은 줌 레벨)
                # traci.gui.setZoom("View #0", 1500)

            # [선택사항] 구급차가 부족하다면 스텝마다 몇 대씩 강제로 생성해서 테스트할 수도 있습니다.
            # 여기서는 편의상 구급차 ID에 'amb'가 포함된 차량들이 이미 맵에 있다고 가정하거나 자동 생성합니다.
            
            # 현재 맵 위에 있는 모든 구급차 찾기 (ID에 'amb' 또는 'ambulance'가 포함된 차량)
            all_vehicles = traci.vehicle.getIDList()
            ambulance_list = [v for v in all_vehicles if 'amb' in v.lower() or 'ambulance' in v.lower()]

            # 만약 테스트용 구급차가 아예 없다면 1대 강제 생성 (테스트용)
            if not ambulance_list and step == 10:
                try:
                    print(f"[Step {step}] 테스트용 구급차를 생성합니다.")
                    s_edge, e_edge = random.choice(valid_edges), random.choice(valid_edges)
                    route = traci.simulation.findRoute(s_edge, e_edge)
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

if __name__ == "__main__":
    run_simulation()
