import os
import sys
import random
import traci
import sumolib


# ============================================================
# 1. SUMO_HOME 환경변수 확인
# ============================================================

if "SUMO_HOME" not in os.environ:
    sys.exit(
        "SUMO_HOME 환경 변수가 설정되어 있지 않습니다.\n"
        "Windows 환경변수에 SUMO_HOME을 설정해주세요."
    )

tools = os.path.join(os.environ["SUMO_HOME"], "tools")

if tools not in sys.path:
    sys.path.append(tools)


# ============================================================
# 2. 기본 설정
# ============================================================

SUMO_CONFIG = "SUMO/강남/osm.sumocfg"

AMBULANCE_ID = "amb_1110108"

# 환자 발생을 시작할 최소 시뮬레이션 스텝
DISPATCH_STEP = 50

# 최대 시뮬레이션 스텝
MAX_SIMULATION_STEPS = 1000

# 환자 후보 Edge를 최대 몇 개까지 검사할지
# 너무 많은 Edge에 findRoute()를 호출하면 느려질 수 있으므로 제한
MAX_ROUTE_CANDIDATES = 300

# 랜덤 시드
# 같은 결과를 재현하고 싶으면 숫자를 고정합니다.
RANDOM_SEED = 42


# ============================================================
# 3. 네트워크 파일 찾기
# ============================================================

def get_network_file():
    """
    SUMO 네트워크 파일 경로를 반환합니다.
    .net.xml.gz 우선 사용하고 없으면 .net.xml 사용
    """

    gz_file = "SUMO/강남/osm.net.xml.gz"
    xml_file = "SUMO/강남/osm.net.xml"

    if os.path.exists(gz_file):
        return gz_file

    if os.path.exists(xml_file):
        return xml_file

    raise FileNotFoundError(
        "SUMO 네트워크 파일을 찾을 수 없습니다.\n"
        f"확인한 경로:\n"
        f"- {gz_file}\n"
        f"- {xml_file}"
    )


# ============================================================
# 4. 차량 존재 여부 확인
# ============================================================

def is_vehicle_available(vehicle_id):
    """
    현재 시뮬레이션에 해당 차량이 존재하는지 확인
    """

    return vehicle_id in traci.vehicle.getIDList()


# ============================================================
# 5. 차량의 현재 일반 도로 Edge 가져오기
# ============================================================

def get_current_edge(vehicle_id):
    """
    차량이 현재 위치한 일반 도로 Edge ID를 반환합니다.

    SUMO에서 교차로 내부 Edge는 ':'로 시작하므로
    일반 Edge와 구분합니다.
    """

    road_id = traci.vehicle.getRoadID(vehicle_id)

    # 차량의 현재 Road ID가 정상적인 경우
    if road_id and not road_id.startswith(":"):
        return road_id

    # 내부 Edge에 위치하고 있을 경우
    lane_id = traci.vehicle.getLaneID(vehicle_id)

    if lane_id and not lane_id.startswith(":"):
        # 일반적으로 lane ID는 edgeID_0 형태
        return lane_id.rsplit("_", 1)[0]

    # 현재 Edge를 정상적으로 얻지 못한 경우
    return None


# ============================================================
# 6. 차량 정보 확인
# ============================================================

def get_vehicle_info(vehicle_id):
    """
    차량의 현재 위치와 차량 타입 정보를 반환합니다.
    """

    try:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
    except traci.TraCIException:
        vehicle_type = ""

    try:
        vehicle_class = traci.vehicle.getVehicleClass(vehicle_id)
    except traci.TraCIException:
        vehicle_class = ""

    return vehicle_type, vehicle_class


from collections import deque
import random
import traci


# ============================================================
# 현재 Edge에서 실제로 연결되어 있는 도로들을 탐색
# ============================================================

def get_reachable_edges(net, current_edge_id, max_edges=1000):
    """
    현재 Edge에서 도로 연결 관계를 따라 이동 가능한 Edge들을 탐색합니다.

    BFS(너비 우선 탐색)를 사용하기 때문에
    현재 위치에서 가까운 연결 도로부터 순서대로 탐색합니다.

    return:
        [(edge_id, distance), ...]
    """

    try:
        start_edge = net.getEdge(current_edge_id)
    except Exception:
        print(f"[오류] 현재 Edge를 네트워크에서 찾을 수 없습니다: {current_edge_id}")
        return []

    visited = set()
    queue = deque()

    visited.add(current_edge_id)
    queue.append((start_edge, 0))

    reachable = []

    while queue and len(reachable) < max_edges:

        current, distance = queue.popleft()

        # 시작 Edge는 환자 위치로 사용하지 않음
        if current.getID() != current_edge_id:
            reachable.append(
                (current.getID(), distance)
            )

        # 현재 Edge에서 다음 Edge로 이동
        for next_edge in current.getOutgoing():

            next_id = next_edge.getID()

            # 내부 Edge는 제외
            if next_id.startswith(":"):
                continue

            # 이미 방문한 Edge는 제외
            if next_id in visited:
                continue

            visited.add(next_id)

            queue.append(
                (next_edge, distance + 1)
            )

    return reachable


# ============================================================
# 도달 가능한 환자 Edge 찾기
# ============================================================

def find_reachable_patient_edge(
    ambulance_id,
    net,
    max_search_edges=1000
):
    """
    구급차 현재 위치에서 실제 도로 연결을 따라 이동 가능한
    환자 위치 Edge를 찾습니다.

    기존 방식:
        네트워크 전체에서 무작위 Edge 300개
        -> findRoute()
        -> 운 좋게 연결된 Edge가 있어야 함

    변경 방식:
        현재 Edge
        -> 연결된 도로 탐색
        -> 실제로 갈 수 있는 Edge 목록 생성
        -> 후보 중 하나 선택
        -> findRoute()로 최종 검증

    return:
        patient_edge, route
    """

    # --------------------------------------------------------
    # 1. 구급차 현재 Edge 확인
    # --------------------------------------------------------

    current_edge = get_current_edge(ambulance_id)

    if current_edge is None:
        print("[오류] 구급차의 현재 Edge를 확인할 수 없습니다.")
        return None, None

    print(f"[정보] 현재 Edge : {current_edge}")

    # --------------------------------------------------------
    # 2. 차량 정보
    # --------------------------------------------------------

    try:
        vehicle_type = traci.vehicle.getTypeID(ambulance_id)
    except Exception:
        vehicle_type = None

    print(f"[정보] Vehicle Type : {vehicle_type}")

    # --------------------------------------------------------
    # 3. 실제 연결된 도로 탐색
    # --------------------------------------------------------

    reachable_edges = get_reachable_edges(
        net,
        current_edge,
        max_edges=max_search_edges
    )

    if not reachable_edges:

        print(
            "[경고] 현재 Edge에서 연결된 다른 도로를 "
            "찾지 못했습니다."
        )

        return None, None

    print(
        f"[정보] 실제 연결된 도로 수 : "
        f"{len(reachable_edges)}"
    )

    # --------------------------------------------------------
    # 4. 너무 가까운 Edge는 제외
    #
    # distance = 1
    # 바로 다음 도로
    #
    # 환자 위치가 너무 가까우면 테스트 의미가 떨어질 수
    # 있으므로 일정 거리 이상인 Edge를 우선 사용
    # --------------------------------------------------------

    preferred_edges = [
        item
        for item in reachable_edges
        if item[1] >= 3
    ]

    # 멀리 있는 후보가 하나도 없다면
    # 전체 reachable edge 사용
    if not preferred_edges:
        preferred_edges = reachable_edges

    # --------------------------------------------------------
    # 5. 후보를 무작위 순서로 섞음
    #
    # 이렇게 하면 항상 동일한 Edge만 환자 위치가 되지 않음
    # --------------------------------------------------------

    random.shuffle(preferred_edges)

    # --------------------------------------------------------
    # 6. 실제 SUMO route가 생성되는지 최종 검증
    # --------------------------------------------------------

    checked = 0

    for edge_id, distance in preferred_edges:

        try:

            checked += 1

            route = traci.simulation.findRoute(
                current_edge,
                edge_id,
                vType=vehicle_type
            )

            # route가 정상적으로 존재하는지 확인
            if route is None:
                continue

            # Edge가 하나도 없는 경우
            if not route.edges:
                continue

            # 현재 위치에서 목적지까지 이동하는 경로인지 확인
            if len(route.edges) < 2:
                continue

            print()
            print("[성공] 도달 가능한 환자 위치를 찾았습니다.")
            print(f"       환자 Edge       : {edge_id}")
            print(f"       연결 거리       : {distance} Edge")
            print(f"       예상 이동시간   : {route.travelTime:.2f}초")
            print(f"       경로 Edge 수    : {len(route.edges)}")
            print()

            return edge_id, route

        except traci.TraCIException:
            continue

        except Exception:
            continue

    # --------------------------------------------------------
    # 7. 최종 실패
    # --------------------------------------------------------

    print(
        f"[경고] {checked}개의 실제 연결 Edge를 검사했지만 "
        "SUMO 경로를 생성하지 못했습니다."
    )

    return None, None

# ============================================================
# 9. 구급차 출동
# ============================================================

def dispatch_ambulance(
    ambulance_id,
    patient_edge,
    route
):
    """
    구급차를 환자 위치로 출동시킵니다.
    """

    try:

        # 최종적으로 다시 경로가 존재하는지 확인
        if route is None or not route.edges:
            print("[오류] 사용할 수 있는 경로가 없습니다.")
            return False

        # changeTarget을 사용하면 목적지 Edge를 설정하고
        # 차량 경로가 다시 계산됩니다.
        traci.vehicle.changeTarget(
            ambulance_id,
            patient_edge
        )

        print(
            f"[출동] 구급차 {ambulance_id} 출동\n"
            f"       목적지 Edge : {patient_edge}\n"
            f"       예상 시간   : {route.travelTime:.2f}초"
        )

        return True

    except traci.TraCIException as e:

        print(f"[오류] 구급차 출동 실패: {e}")

        return False


# ============================================================
# 10. 구급차 도착 여부 확인
# ============================================================

def check_arrival(
    ambulance_id,
    patient_edge
):
    """
    구급차가 환자 Edge에 실제 진입했는지 확인합니다.
    """

    if not is_vehicle_available(ambulance_id):
        return False

    current_edge = get_current_edge(ambulance_id)

    return current_edge == patient_edge


# ============================================================
# 11. 시뮬레이션 실행
# ============================================================

def run_simulation():

    # 네트워크 파일
    net_file = get_network_file()

    print("=" * 60)
    print("SUMO 구급차 시뮬레이션 시작")
    print("=" * 60)

    print(f"[정보] SUMO 설정 파일 : {SUMO_CONFIG}")
    print(f"[정보] 네트워크 파일   : {net_file}")
    print(f"[정보] 구급차 ID       : {AMBULANCE_ID}")

    # 랜덤 결과 재현
    random.seed(RANDOM_SEED)

    # 네트워크는 매번 읽지 않고 프로그램 시작 시 한 번만 읽습니다.
    print("[정보] SUMO 네트워크를 읽는 중...")
    net = sumolib.net.readNet(net_file)

    print(
        f"[정보] 일반 도로 Edge 수 : "
        f"{len(net.getEdges(withInternal=False))}"
    )

    # --------------------------------------------------------
    # SUMO 실행
    # --------------------------------------------------------

    sumo_command = [
        "sumo-gui",
        "-c",
        SUMO_CONFIG,
    ]

    traci.start(sumo_command)

    print("[정보] SUMO GUI가 실행되었습니다.")

    # --------------------------------------------------------
    # 상태 변수
    # --------------------------------------------------------

    mission_state = "WAITING"

    patient_edge = None
    dispatch_time = None
    arrival_time = None

    # 출동을 한 번만 실행하기 위한 변수
    dispatch_requested = False

    try:

        # ----------------------------------------------------
        # 메인 시뮬레이션
        # ----------------------------------------------------

        for step in range(1, MAX_SIMULATION_STEPS + 1):

            # SUMO를 한 스텝 진행
            traci.simulationStep()

            # ------------------------------------------------
            # 현재 차량 존재 확인
            # ------------------------------------------------

            ambulance_exists = is_vehicle_available(
                AMBULANCE_ID
            )

            # ------------------------------------------------
            # 이벤트 1
            # 환자 발생 및 구급차 출동
            # ------------------------------------------------

            if (
                step >= DISPATCH_STEP
                and not dispatch_requested
                and mission_state == "WAITING"
            ):

                dispatch_requested = True

                if ambulance_exists:

                    print()
                    print("=" * 60)
                    print(f"[Step {step}] 환자 발생")
                    print("=" * 60)

                    try:

                        patient_edge, route = find_reachable_patient_edge(
                            AMBULANCE_ID,
                            net
                        )

                        if patient_edge is not None:

                            success = dispatch_ambulance(
                                AMBULANCE_ID,
                                patient_edge,
                                route
                            )

                            if success:

                                mission_state = "TO_PATIENT"
                                dispatch_time = step

                            else:

                                mission_state = "FAILED"

                        else:

                            mission_state = "FAILED"

                    except Exception as e:

                        mission_state = "FAILED"

                        print(
                            f"[오류] 환자 위치 탐색 중 오류 발생: {e}"
                        )

                else:

                    # 차량이 아직 출발하지 않은 상황
                    # 기존 코드와 달리 여기서 바로 실패 처리하지 않습니다.
                    #
                    # dispatch_requested를 다시 False로 돌려놓아서
                    # 다음 Step에서 차량이 나타났을 때 다시 시도합니다.

                    dispatch_requested = False

                    if step % 10 == 0:

                        print(
                            f"[Step {step}] "
                            f"구급차 {AMBULANCE_ID}가 아직 로드되지 않았습니다."
                        )

            # ------------------------------------------------
            # 이벤트 2
            # 구급차 이동 상태 확인
            # ------------------------------------------------

            if (
                mission_state == "TO_PATIENT"
                and patient_edge is not None
                and ambulance_exists
            ):

                current_edge = get_current_edge(
                    AMBULANCE_ID
                )

                # 현재 위치 출력
                if step % 20 == 0:

                    print(
                        f"[Step {step}] "
                        f"구급차 현재 위치 = {current_edge}, "
                        f"목적지 = {patient_edge}"
                    )

                # --------------------------------------------
                # 도착 판정
                # --------------------------------------------

                if check_arrival(
                    AMBULANCE_ID,
                    patient_edge
                ):

                    mission_state = "ARRIVED"
                    arrival_time = step

                    travel_steps = arrival_time - dispatch_time

                    print()
                    print("=" * 60)
                    print("환자 현장 도착")
                    print("=" * 60)
                    print(f"출동 Step : {dispatch_time}")
                    print(f"도착 Step : {arrival_time}")
                    print(f"이동 시간 : {travel_steps} Step")
                    print(f"환자 Edge : {patient_edge}")
                    print("=" * 60)

                    # 목적지 도착했으므로 시뮬레이션 종료
                    break

            # ------------------------------------------------
            # 이벤트 3
            # 시뮬레이션 중 차량이 teleport되는 상황 확인
            # ------------------------------------------------

            try:

                teleporting_vehicles = (
                    traci.simulation.getEndingTeleportIDList()
                )

                if AMBULANCE_ID in teleporting_vehicles:

                    print(
                        f"[경고] Step {step}: "
                        f"구급차가 Teleport 처리되었습니다."
                    )

            except Exception:
                pass

            # ------------------------------------------------
            # 이벤트 4
            # SUMO 내부 차량이 모두 없어졌는지 확인
            # ------------------------------------------------

            if traci.simulation.getMinExpectedNumber() == 0:

                print(
                    f"[정보] Step {step}: "
                    "더 이상 진행할 차량이 없어 시뮬레이션을 종료합니다."
                )

                break

        # ----------------------------------------------------
        # 시뮬레이션 종료 결과
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("시뮬레이션 결과")
        print("=" * 60)

        print(f"최종 상태 : {mission_state}")

        if patient_edge is not None:
            print(f"환자 Edge : {patient_edge}")

        if dispatch_time is not None:
            print(f"출동 Step : {dispatch_time}")

        if arrival_time is not None:

            print(f"도착 Step : {arrival_time}")
            print(
                f"응답 시간 : "
                f"{arrival_time - dispatch_time} Step"
            )

        elif mission_state == "TO_PATIENT":

            print(
                "결과      : "
                "구급차가 제한된 시뮬레이션 시간 내에 도착하지 못했습니다."
            )

        elif mission_state == "FAILED":

            print(
                "결과      : "
                "환자 위치 설정 또는 구급차 출동에 실패했습니다."
            )

        print("=" * 60)

    except traci.TraCIException as e:

        print()
        print("[TraCI 오류]")
        print(e)

    except KeyboardInterrupt:

        print()
        print("[정보] 사용자가 시뮬레이션을 중단했습니다.")

    finally:

        # ----------------------------------------------------
        # 반드시 SUMO 연결 종료
        # ----------------------------------------------------

        try:
            traci.close()
        except Exception:
            pass

        print("[정보] SUMO 연결이 종료되었습니다.")
        print("[정보] 시뮬레이션 완료.")


# ============================================================
# 12. 프로그램 시작점
# ============================================================

if __name__ == "__main__":
    run_simulation()