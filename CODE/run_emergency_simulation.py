import os
import sys
import sumolib
import traci

## 환경 변수 여부 확인 ##
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')          # SUMO_HOME 환경 변수에서 tools 경로를 가져와서 sys.path에 추가
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")     # 환경 변수 X -> 설정 안내 메시지 출력 후 종료


## 구급차의 현재 위치에서 실제로 경로가 연결되어 도달 가능한 유효한 도로 Edge ID 찾기 ##
def get_reachable_patient_edge(ambulance_id, net_file):
    net = sumolib.net.readNet(net_file)     # SUMO 네트워크 파일 읽기
    curr_edge = traci.vehicle.getRoadID(ambulance_id)       # 구급차의 현재 도로 Edge ID 가져오기
    if curr_edge.startswith(':'):
        curr_edge = traci.vehicle.getLaneID(ambulance_id).split('_')[0]     # 현재 Edge가 내부 Edge인 경우, 실제 도로 Edge로 변환

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
    net_file = 'SUMO/강남/osm.net.xml.gz' if os.path.exists('SUMO/강남/osm.net.xml.gz') else 'SUMO/강남/osm.net.xml'
    
    sumoConfig = ["sumo-gui", "-c", "SUMO/강남/osm.sumocfg"]
    traci.start(sumoConfig)
    print("SUMO GUI가 실행되었습니다.")

    ambulance_id = "amb_1110108"
    mission_state = "WAITING"
    patient_edge = None

    for step in range(1, 1000):
        traci.simulationStep()

        # [이벤트 1] 구급차가 로드된 이후 50스텝째에 도달 가능한 환자 위치 동적 설정 및 출동
        if step == 50:
            try:
                if ambulance_id in traci.vehicle.getIDList():
                    # 도달 가능한 Edge를 동적으로 탐색
                    patient_edge = get_reachable_patient_edge(ambulance_id, net_file)
                    if patient_edge:
                        traci.vehicle.changeTarget(ambulance_id, patient_edge)
                        mission_state = "TO_PATIENT"
                        print(f"[Step {step}] 환자 발생! 구급차 출동 (목적지 Edge: {patient_edge})")
                    else:
                        print(f"[Step {step}] 도달 가능한 환자 위치를 찾지 못했습니다.")
                else:
                    print(f"[Step {step}] 구급차({ambulance_id})가 아직 로드되지 않았습니다.")
            except Exception as e:
                print(f"[Step {step}] 출동 제어 에러: {e}")

        # [이벤트 2] 구급차 상태 체크
        if ambulance_id in traci.vehicle.getIDList() and patient_edge:
            current_edge = traci.vehicle.getRoadID(ambulance_id)
            if mission_state == "TO_PATIENT" and current_edge == patient_edge:
                mission_state = "ARRIVED"
                print(f"[Step {step}] 구급차가 환자 현장 도로에 도달했습니다.")

    traci.close()
    print("시뮬레이션 완료.")

if __name__ == "__main__":
    run_simulation()