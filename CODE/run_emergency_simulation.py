import os
import sys
import json
import traci
import sumolib

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def get_hospital_edges(net):
    """
    응급실 JSON 파일에서 강남/서초구 병원들을 읽어와 
    SUMO 네트워크 상에 실제로 존재하는 Edge ID만 필터링하여 반환합니다.
    """
    with open('DATASET/서울시 응급실 위치 정보.json', 'r', encoding='utf-8') as f:
        er_data = json.load(f)
        
    hospital_edges = {}
    for h in er_data.get('DATA', []):
        name = h.get('dutyname')
        addr = h.get('dutyaddr', '')
        
        if '강남구' in addr or '서초구' in addr:
            lon = float(h.get('wgs84lon', 0))
            lat = float(h.get('wgs84lat', 0))
            
            if lon and lat:
                x, y = net.convertLonLat2XY(lon, lat)
                edges = net.getNeighboringEdges(x, y, 300)
                if edges:
                    closest_edge = sorted(edges, key=lambda e: e[1])[0][0].getID()
                    # 네트워크에 실제 존재하는 Edge인지 한 번 더 검증
                    if net.hasEdge(closest_edge):
                        hospital_edges[name] = closest_edge
                        
    return hospital_edges

def get_fastest_hospital(ambulance_id, hospital_edges):
    curr_edge = traci.vehicle.getRoadID(ambulance_id)
    if curr_edge.startswith(':'):
        curr_edge = traci.vehicle.getLaneID(ambulance_id).split('_')[0]

    best_hospital = None
    min_travel_time = float('inf')

    for hosp_name, hosp_edge in hospital_edges.items():
        try:
            route_info = traci.simulation.findRoute(curr_edge, hosp_edge, vType="ambulance")
            travel_time = route_info.travelTime

            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_hospital = (hosp_name, hosp_edge, travel_time)
        except Exception:
            continue

    return best_hospital

def run_simulation():
    net_file = 'SUMO/강남/osm.net.xml.gz' if os.path.exists('SUMO/강남/osm.net.xml.gz') else 'osm.net.xml'
    net = sumolib.net.readNet(net_file)
    
    print("병원 위치 데이터 자동 매칭 중...")
    hospital_edges = get_hospital_edges(net)
    print(f"총 {len(hospital_edges)}개의 유효한 병원 도로 매칭 완료!\n")

    sumoConfig = ["sumo-gui", "-c", "SUMO/강남/osm.sumocfg"]
    traci.start(sumoConfig)
    print("SUMO 시뮬레이션 및 TraCI 연동 시작!")

    ambulance_id = "amb_1110108"
    step = 0
    mission_state = "WAITING" 

    patient_edge = list(hospital_edges.values())[0] if hospital_edges else "" 

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        if step == 50:
            try:
                # 차선 에러를 방지하기 위해 정차 위치 설정 시 Edge 기준으로 처리
                traci.vehicle.setStop(ambulance_id, edgeID=patient_edge, pos=10, duration=0)
                traci.vehicle.changeTarget(ambulance_id, patient_edge)
                mission_state = "TO_PATIENT"
                print(f"[Step {step}] 환자 발생! 구급차({ambulance_id}) 현장 출동.")
            except Exception as e:
                print(f"[Step {step}] 출동 명령 에러: {e}")

        if ambulance_id in traci.vehicle.getIDList():
            current_edge = traci.vehicle.getRoadID(ambulance_id)

            if mission_state == "TO_PATIENT" and current_edge == patient_edge:
                # 엣지 단위 정차 명령 적용
                traci.vehicle.setStop(ambulance_id, edgeID=patient_edge, pos=10, duration=5)
                mission_state = "LOADING"
                print(f"[Step {step}] 현장 도착. 환자 탑승 중...")

            elif mission_state == "LOADING" and traci.vehicle.getSpeed(ambulance_id) == 0:
                best_hosp = get_fastest_hospital(ambulance_id, hospital_edges)
                if best_hosp:
                    hosp_name, hosp_edge, travel_time = best_hosp
                    traci.vehicle.changeTarget(ambulance_id, hosp_edge)
                    mission_state = "TO_HOSPITAL"
                    print(f"[Step {step}] 최적 병원 선정: '{hosp_name}' (예상 소요 시간: {travel_time:.1f}초)")

        if step > 3600:
            break

    traci.close()
    print("시뮬레이션 종료.")

if __name__ == "__main__":
    run_simulation()