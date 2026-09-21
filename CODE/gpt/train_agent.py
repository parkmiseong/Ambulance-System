#!/usr/bin/env python3

import contextlib
import copy
import os
import random
import sys

import numpy as np
import sumolib
import traci

from baselines import HospitalRouters
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import (
    GOLDEN_TIME_LIMITS,
    REJECTION_PENALTY,
    calculate_reward,
)


# ============================================================
# 학습 설정
# ============================================================

MODEL_PATH = "dqn_ambulance_model_v3.pth"

MAX_STEPS = 500
MAX_AMBULANCES = 30

# 하나의 이송이 지나치게 오래 지속되는 것을 방지
MISSION_TIMEOUT_SECONDS = 350.0

# 병원 도착 판정 거리
ARRIVAL_DISTANCE = 120.0

# 병원 혼잡도 기준
HOSPITAL_FULL_THRESHOLD = 0.85

# 상태에 사용할 예상 이동시간의 최대값
# 이 값을 넘는 이동시간은 이 값으로 clipping
MAX_ESTIMATED_TRAVEL_TIME = 600.0


# ============================================================
# 시나리오
# ============================================================

SCENARIOS = [
    {
        "id": 1,
        "patient_freq": "원활",
        "traffic": "원활",
        "complexity": 2,
        "prob": 0.20,
        "scale": 0.3,
    },
    {
        "id": 2,
        "patient_freq": "원활",
        "traffic": "보통",
        "complexity": 3,
        "prob": 0.20,
        "scale": 0.5,
    },
    {
        "id": 3,
        "patient_freq": "원활",
        "traffic": "혼잡",
        "complexity": 4,
        "prob": 0.20,
        "scale": 0.8,
    },
    {
        "id": 4,
        "patient_freq": "보통",
        "traffic": "원활",
        "complexity": 3,
        "prob": 0.30,
        "scale": 0.3,
    },
    {
        "id": 5,
        "patient_freq": "보통",
        "traffic": "보통",
        "complexity": 4,
        "prob": 0.30,
        "scale": 0.5,
    },
    {
        "id": 6,
        "patient_freq": "보통",
        "traffic": "혼잡",
        "complexity": 5,
        "prob": 0.30,
        "scale": 0.8,
    },
    {
        "id": 7,
        "patient_freq": "혼잡",
        "traffic": "원활",
        "complexity": 4,
        "prob": 0.40,
        "scale": 0.3,
    },
    {
        "id": 8,
        "patient_freq": "혼잡",
        "traffic": "보통",
        "complexity": 5,
        "prob": 0.40,
        "scale": 0.5,
    },
    {
        "id": 9,
        "patient_freq": "혼잡",
        "traffic": "혼잡",
        "complexity": 6,
        "prob": 0.40,
        "scale": 0.8,
    },
]


# ============================================================
# SUMO 출력 억제
# ============================================================

@contextlib.contextmanager
def suppress_sumo_stdout():
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# ============================================================
# 병원 edge 찾기
# ============================================================

def get_reachable_edge_id(
    net,
    x,
    y,
    from_edge=None,
    vtype="ambulance_custom",
):
    """
    좌표 주변에서 해당 차량 유형이 실제로 출발/도착할 수 있고,
    현재 출발 Edge에서 접근 가능한 도로 Edge를 찾는다.

    SUMO 내부 오류 메시지는 출력하지 않는다.
    """

    radius = 100.0
    visited_edges = set()

    # 출발 Edge 자체가 유효한지 먼저 확인
    if from_edge is not None:

        try:
            from_edge_obj = net.getEdge(from_edge)

            # 차량 유형이 해당 Edge를 이용할 수 있는지 확인
            lanes = from_edge_obj.getLanes()

            allowed = False

            for lane in lanes:
                try:
                    permissions = lane.getPermissions()

                    # permissions가 없거나 비어 있으면
                    # 일반적으로 접근 가능하다고 판단
                    if not permissions:
                        allowed = True
                        break

                    # emergency 차량이 허용되는지 확인
                    if (
                        "emergency"
                        in permissions
                        or "all"
                        in permissions
                    ):
                        allowed = True
                        break

                except Exception:
                    allowed = True
                    break

            if not allowed:
                return None

        except Exception:
            return None

    while radius <= 5000.0:

        try:
            nearby = net.getNeighboringEdges(
                x,
                y,
                radius,
            )
        except Exception:
            nearby = []

        for edge, _distance in sorted(
            nearby,
            key=lambda item: item[1],
        ):

            edge_id = edge.getID()

            # SUMO 내부 edge 제외
            if edge_id.startswith(":"):
                continue

            # 이미 검사한 edge 제외
            if edge_id in visited_edges:
                continue

            visited_edges.add(edge_id)

            # 출발 Edge와 동일하면 바로 사용
            if from_edge is not None:
                if edge_id == from_edge:
                    return edge_id

            # 출발지가 없는 경우
            if from_edge is None:
                return edge_id

            # ------------------------------------------------
            # 해당 Edge에 실제로 접근 가능한지 확인
            # ------------------------------------------------

            try:

                # SUMO 오류 출력을 숨김
                with suppress_sumo_stdout():

                    route = traci.simulation.findRoute(
                        from_edge,
                        edge_id,
                        vType=vtype,
                        depart=-1,
                        routingMode=0,
                    )

                if (
                    route is not None
                    and len(route.edges) > 0
                ):
                    return edge_id

            except traci.TraCIException:
                # 접근 불가능한 Edge는 조용히 건너뜀
                continue

            except Exception:
                continue

        radius += 200.0

    return None


# ============================================================
# 환자 발생 스케줄
# ============================================================

def generate_fixed_patient_schedule(
    scenario,
    valid_edges,
    seed_val,
):
    """
    구급차가 사용할 수 있는 일반 도로 Edge 중에서
    환자 발생 위치를 생성한다.
    """

    random.seed(seed_val)
    np.random.seed(seed_val)

    # 내부 Edge 및 비정상 Edge 제거
    usable_edges = [
        edge_id
        for edge_id in valid_edges
        if edge_id
        and not str(edge_id).startswith(":")
    ]

    if not usable_edges:
        raise RuntimeError(
            "사용 가능한 환자 발생 Edge가 없습니다."
        )

    schedule = {}

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        if random.random() < scenario["prob"]:

            patient_edge = random.choice(
                usable_edges
            )

            schedule[step] = {
                "edge_id": patient_edge,
                "severity": random.randint(
                    1,
                    4,
                ),
            }

    return schedule


# ============================================================
# 병원 치료역량 인코딩
# ============================================================

def encode_hospital_type(hospital):
    """
    병원 유형을 수치화한다.

    권역응급의료센터 → 1.0
    지역응급의료센터 → 0.8
    일반 응급실     → 0.3
    """

    hospital_type = str(
        hospital.get("type", "")
    )

    if "권역" in hospital_type:
        return 1.0

    if "지역" in hospital_type:
        return 0.8

    return 0.3


# ============================================================
# 환자-병원 치료 적합도
# ============================================================

def get_hospital_suitability(
    severity,
    hospital,
):
    """
    환자 중증도와 병원 치료역량의 적합도를 계산한다.

    반환값:
        1.0 = 매우 적합
        0.5 = 일반적으로 가능
        0.0 = 부적합 가능성이 높음
    """

    hospital_type = str(
        hospital.get("type", "")
    )

    # 중증 환자
    if severity >= 3:

        if "권역" in hospital_type:
            return 1.0

        if "지역" in hospital_type:
            return 0.8

        return 0.0

    # 경증/중등도
    if "권역" in hospital_type:
        return 0.8

    if "지역" in hospital_type:
        return 1.0

    return 0.7


# ============================================================
# 병원별 route 계산
# ============================================================

def calculate_hospital_routes(
    net,
    patient_edge,
    hospitals,
):
    """
    현재 환자 위치에서 모든 병원까지의
    최소 예상 이동시간 경로를 계산한다.

    반환:
        hospital_routes = [
            {
                "edge_id": ...,
                "route": ...,
                "travel_time": ...
            },
            ...
        ]
    """

    hospital_routes = []

    for hospital in hospitals:

        x = hospital.get("sumo_x")
        y = hospital.get("sumo_y")

        result = {
            "edge_id": None,
            "route": None,
            "travel_time": MAX_ESTIMATED_TRAVEL_TIME,
        }

        if x is None or y is None:
            hospital_routes.append(result)
            continue

        # 병원 좌표를 실제 SUMO road edge로 변환
        hospital_edge = get_reachable_edge_id(
            net,
            x,
            y,
            from_edge=patient_edge,
            vtype="ambulance_custom",
        )

        if hospital_edge is None:
            hospital_routes.append(result)
            continue

        result["edge_id"] = hospital_edge

        try:

            route = traci.simulation.findRoute(
                patient_edge,
                hospital_edge,
                vType="ambulance_custom",
                depart=-1,
                routingMode=0,
            )

            if route and len(route.edges) > 0:

                travel_time = float(
                    max(
                        0.0,
                        route.travelTime,
                    )
                )

                result["route"] = route
                result["travel_time"] = min(
                    travel_time,
                    MAX_ESTIMATED_TRAVEL_TIME,
                )

        except Exception:
            pass

        hospital_routes.append(result)

    return hospital_routes


# ============================================================
# 상태 벡터
# ============================================================

def build_state_vector(
    patient_pos,
    severity,
    hospitals,
    hospital_routes,
):
    """
    DQN 상태 벡터.

    구성:

    [환자 정보]
        환자 X
        환자 Y
        중증도

    [병원별 정보]
        예상 이동시간
        병상 점유율
        치료역량
        환자-병원 적합도

    따라서 DQN은 단순 거리만 보는 것이 아니라
    이동시간 + 병상 + 치료역량을 함께 고려한다.
    """

    hospital_count = len(hospitals)

    # --------------------------------------------------------
    # 환자 위치
    # --------------------------------------------------------

    patient_x = float(patient_pos[0]) / 10000.0
    patient_y = float(patient_pos[1]) / 10000.0

    normalized_severity = float(
        severity
    ) / 4.0

    state = [
        patient_x,
        patient_y,
        normalized_severity,
    ]

    # --------------------------------------------------------
    # 병원별 예상 이동시간
    # --------------------------------------------------------

    for route_info in hospital_routes:

        travel_time = float(
            route_info.get(
                "travel_time",
                MAX_ESTIMATED_TRAVEL_TIME,
            )
        )

        normalized_time = min(
            travel_time / MAX_ESTIMATED_TRAVEL_TIME,
            1.0,
        )

        state.append(normalized_time)

    # --------------------------------------------------------
    # 병원별 병상 점유율
    # --------------------------------------------------------

    for hospital in hospitals:

        occupancy = float(
            hospital.get(
                "occupancy",
                0.0,
            )
        )

        occupancy = max(
            0.0,
            min(1.0, occupancy),
        )

        state.append(occupancy)

    # --------------------------------------------------------
    # 병원별 치료역량
    # --------------------------------------------------------

    for hospital in hospitals:

        state.append(
            encode_hospital_type(hospital)
        )

    # --------------------------------------------------------
    # 병원별 환자 적합도
    # --------------------------------------------------------

    for hospital in hospitals:

        state.append(
            get_hospital_suitability(
                severity,
                hospital,
            )
        )

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    expected_dim = (
        3
        + hospital_count
        + hospital_count
        + hospital_count
        + hospital_count
    )

    if state.size != expected_dim:

        raise ValueError(
            f"상태 차원 오류: "
            f"현재={state.size}, "
            f"예상={expected_dim}"
        )

    return state


# ============================================================
# SUMO 시작
# ============================================================

def _start_sumo(
    env,
    scenario,
    seed_val,
):
    sumo_binary = sumolib.checkBinary(
        "sumo"
    )

    config = [
        sumo_binary,
        "-c",
        str(env.cfg_file),

        "--scale",
        str(scenario["scale"]),

        "--no-warnings",
        "true",

        "--no-step-log",
        "true",

        "--duration-log.disable",
        "true",

        "--seed",
        str(seed_val),

        "--start",
    ]

    with suppress_sumo_stdout():

        traci.start(config)

        # 기본 차량 유형을 복사하여
        # 구급차 전용 차량 유형 생성
        traci.vehicletype.copy(
            "DEFAULT_VEHTYPE",
            "ambulance_custom",
        )

        traci.vehicletype.setVehicleClass(
            "ambulance_custom",
            "emergency",
        )

        traci.vehicletype.setShapeClass(
            "ambulance_custom",
            "emergency",
        )

    env.update_hospital_coordinates()

    step_length = float(
        traci.simulation.getDeltaT()
    )

    if step_length <= 0:
        step_length = 1.0

    return step_length


# ============================================================
# 실제 이동용 최단 시간 경로
# ============================================================


def _find_route(
    from_edge,
    to_edge,
):
    """
    출발 Edge → 목적지 Edge의 이동시간 최소 경로를 계산한다.

    잘못된 Edge나 차량 유형이 접근할 수 없는 Edge가 들어오면
    오류를 출력하지 않고 None을 반환한다.
    """

    if not from_edge or not to_edge:
        return None

    # SUMO 내부 Edge는 목적지로 사용하지 않음
    if str(from_edge).startswith(":"):
        return None

    if str(to_edge).startswith(":"):
        return None

    try:

        with suppress_sumo_stdout():

            route = traci.simulation.findRoute(
                from_edge,
                to_edge,
                vType="ambulance_custom",
                depart=-1,
                routingMode=0,
            )

        if (
            route is not None
            and len(route.edges) > 0
        ):
            return route

    except traci.TraCIException:
        return None

    except Exception:
        return None

    return None


# ============================================================
# DQN Action
# ============================================================

def _get_valid_actions(
    hospitals,
    severity,
):
    """
    현재 환자가 갈 수 있는 병원 Action.

    병상이 사실상 가득 찬 병원은 제외한다.

    단, 치료역량은 가능한 한 reward가 학습하도록
    상태에 포함하고 지나치게 강한 masking은 하지 않는다.
    """

    valid_actions = []

    for index, hospital in enumerate(hospitals):

        occupancy = float(
            hospital.get(
                "occupancy",
                0.0,
            )
        )

        if occupancy >= HOSPITAL_FULL_THRESHOLD:
            continue

        valid_actions.append(index)

    return valid_actions


# ============================================================
# fallback action
# ============================================================

def _get_fallback_actions(
    hospitals,
):
    """
    모든 병원이 가득 찬 경우
    가장 여유 있는 병원을 선택 후보로 반환한다.
    """

    if not hospitals:
        return []

    min_occupancy = min(
        float(
            hospital.get(
                "occupancy",
                1.0,
            )
        )
        for hospital in hospitals
    )

    return [
        index
        for index, hospital in enumerate(hospitals)
        if float(
            hospital.get(
                "occupancy",
                1.0,
            )
        )
        <= min_occupancy + 1e-6
    ]


# ============================================================
# DQN 학습
# ============================================================

def train_dqn_fast(
    num_episodes=50,
):

    env = SumoMedicalEnvironment()

    valid_edges = env.get_valid_edges()

    temp_hospitals = env.get_hospitals()

    action_dim = len(
        temp_hospitals
    )

    # --------------------------------------------------------
    # 새로운 상태 차원
    #
    # 3
    # + 병원별 예상 이동시간
    # + 병원별 occupancy
    # + 병원별 치료역량
    # + 병원별 적합도
    # --------------------------------------------------------

    state_dim = (
        3
        + action_dim * 4
    )

    agent = DQNAmbulanceAgent(
        state_dim,
        action_dim,
    )

    # 새 모델로 학습
    agent.epsilon = 1.0

    print()
    print("=" * 70)
    print("DQN 구급차 병원 선택 학습 시작")
    print("=" * 70)
    print(f"State Dimension : {state_dim}")
    print(f"Action Dimension: {action_dim}")
    print(f"Episodes        : {num_episodes}")
    print("=" * 70)

    for episode in range(
        1,
        num_episodes + 1,
    ):

        scenario = random.choice(
            SCENARIOS
        )

        seed_val = (
            10000
            + episode
        )

        patient_schedule = (
            generate_fixed_patient_schedule(
                scenario,
                valid_edges,
                seed_val,
            )
        )

        hospitals = copy.deepcopy(
            temp_hospitals
        )

        # 병원 점유율을 매 episode마다
        # 새롭게 초기화
        for hospital in hospitals:

            hospital["occupancy"] = random.uniform(
                0.2,
                0.6,
            )

        active_missions = {}

        episode_reward = 0.0
        episode_losses = []

        completed_missions = 0
        failed_missions = 0

        rejection_total = 0

        step_length = 1.0

        try:

            step_length = _start_sumo(
                env,
                scenario,
                seed_val,
            )

            # ------------------------------------------------
            # Simulation
            # ------------------------------------------------

            for step in range(
                1,
                MAX_STEPS + 1,
            ):

                traci.simulationStep()

                # ------------------------------------------------
                # 병원 점유율 자연 감소
                # ------------------------------------------------

                for hospital in hospitals:

                    hospital["occupancy"] = max(
                        0.0,
                        float(
                            hospital.get(
                                "occupancy",
                                0.0,
                            )
                        ) - 0.0005,
                    )

                # ------------------------------------------------
                # 현재 구급차 확인
                # ------------------------------------------------

                ambulance_ids = [
                    vid
                    for vid in traci.vehicle.getIDList()
                    if "ambulance" in vid.lower()
                ]

                # ------------------------------------------------
                # 기존 임무 처리
                # ------------------------------------------------

                finished_ids = []

                for ambulance_id, mission in list(
                    active_missions.items()
                ):

                    if ambulance_id not in ambulance_ids:

                        reward = calculate_reward(
                            travel_time=mission[
                                "elapsed_time"
                            ],
                            severity=mission[
                                "severity"
                            ],
                            hospital=mission[
                                "hospital"
                            ],
                            golden_limit=mission[
                                "golden_limit"
                            ],
                            rejection_count=mission[
                                "rejection_count"
                            ],
                            occupancy_at_selection=mission[
                                "occupancy_at_selection"
                            ],
                            route_failed=True,
                        )

                        zero_state = np.zeros(
                            state_dim,
                            dtype=np.float32,
                        )

                        agent.store_transition(
                            mission["state"],
                            mission["action"],
                            reward,
                            zero_state,
                            True,
                        )

                        loss = agent.train_step()

                        if loss is not None:
                            episode_losses.append(
                                loss
                            )

                        episode_reward += reward

                        failed_missions += 1

                        finished_ids.append(
                            ambulance_id
                        )

                        continue

                    # ----------------------------------------
                    # 경과시간
                    # ----------------------------------------

                    mission["elapsed_time"] += (
                        step_length
                    )

                    try:

                        vehicle_pos = (
                            traci.vehicle.getPosition(
                                ambulance_id
                            )
                        )

                        target_pos = mission[
                            "target_pos"
                        ]

                        distance = (
                            (
                                vehicle_pos[0]
                                - target_pos[0]
                            ) ** 2
                            +
                            (
                                vehicle_pos[1]
                                - target_pos[1]
                            ) ** 2
                        ) ** 0.5

                    except Exception:

                        distance = float("inf")

                    arrived = (
                        distance
                        <= ARRIVAL_DISTANCE
                    )

                    timeout = (
                        mission["elapsed_time"]
                        >= MISSION_TIMEOUT_SECONDS
                    )

                    if arrived or timeout:

                        reward = calculate_reward(
                            travel_time=mission[
                                "elapsed_time"
                            ],
                            severity=mission[
                                "severity"
                            ],
                            hospital=mission[
                                "hospital"
                            ],
                            golden_limit=mission[
                                "golden_limit"
                            ],
                            rejection_count=mission[
                                "rejection_count"
                            ],
                            occupancy_at_selection=mission[
                                "occupancy_at_selection"
                            ],
                            route_failed=not arrived,
                        )

                        zero_state = np.zeros(
                            state_dim,
                            dtype=np.float32,
                        )

                        agent.store_transition(
                            mission["state"],
                            mission["action"],
                            reward,
                            zero_state,
                            True,
                        )

                        loss = agent.train_step()

                        if loss is not None:
                            episode_losses.append(
                                loss
                            )

                        episode_reward += reward

                        if arrived:
                            completed_missions += 1
                        else:
                            failed_missions += 1

                        finished_ids.append(
                            ambulance_id
                        )

                for ambulance_id in finished_ids:

                    active_missions.pop(
                        ambulance_id,
                        None,
                    )

                # ------------------------------------------------
                # 새로운 환자 발생
                # ------------------------------------------------

                patient = patient_schedule.get(
                    step
                )

                if patient is None:
                    continue

                if len(ambulance_ids) >= MAX_AMBULANCES:
                    continue

                patient_edge = patient[
                    "edge_id"
                ]

                severity = int(
                    patient[
                        "severity"
                    ]
                )

                # ------------------------------------------------
                # 환자 위치
                # ------------------------------------------------

                try:

                    edge_obj = env.net.getEdge(
                        patient_edge
                    )

                    patient_pos = (
                        edge_obj.getFromNode().getCoord()
                    )

                except Exception:

                    continue

                # ------------------------------------------------
                # 병원별 예상 경로 계산
                # ------------------------------------------------

                hospital_routes = (
                    calculate_hospital_routes(
                        env.net,
                        patient_edge,
                        hospitals,
                    )
                )

                # ------------------------------------------------
                # DQN State 생성
                # ------------------------------------------------

                state = build_state_vector(
                    patient_pos,
                    severity,
                    hospitals,
                    hospital_routes,
                )

                # ------------------------------------------------
                # 선택 가능한 병원
                # ------------------------------------------------

                valid_actions = _get_valid_actions(
                    hospitals,
                    severity,
                )

                if not valid_actions:

                    valid_actions = (
                        _get_fallback_actions(
                            hospitals
                        )
                    )

                if not valid_actions:
                    continue

                # ------------------------------------------------
                # 초기 학습에서는 일부 rule-based 행동
                # ------------------------------------------------

                if (
                    episode <= 10
                    and random.random() < 0.25
                ):

                    # 가장 빠른 병원 선택
                    selected_action = min(
                        valid_actions,
                        key=lambda index:
                            hospital_routes[index][
                                "travel_time"
                            ],
                    )

                else:

                    selected_action = (
                        agent.select_action(
                            state,
                            valid_actions,
                        )
                    )

                selected_action = int(
                    selected_action
                )

                selected_hospital = hospitals[
                    selected_action
                ]

                # ------------------------------------------------
                # 병상 거부 처리
                #
                # 여기서는 별도의 terminal transition을
                # 만들지 않는다.
                #
                # 최종 reward에서 rejection_count를 반영한다.
                # ------------------------------------------------

                rejection_count = 0

                while (
                    float(
                        selected_hospital.get(
                            "occupancy",
                            0.0,
                        )
                    )
                    >= HOSPITAL_FULL_THRESHOLD
                    and rejection_count < 4
                ):

                    rejection_count += 1

                    rejection_total += 1

                    alternative_actions = [
                        action
                        for action in valid_actions
                        if action != selected_action
                    ]

                    if not alternative_actions:
                        break

                    selected_action = (
                        agent.select_action(
                            state,
                            alternative_actions,
                        )
                    )

                    selected_hospital = hospitals[
                        selected_action
                    ]

                # ------------------------------------------------
                # 선택 당시 병원 상태 기록
                # ------------------------------------------------

                occupancy_at_selection = float(
                    selected_hospital.get(
                        "occupancy",
                        0.0,
                    )
                )

                # ------------------------------------------------
                # 병원 수용 처리
                # ------------------------------------------------

                selected_hospital[
                    "occupancy"
                ] = min(
                    1.0,
                    occupancy_at_selection
                    + 0.08,
                )

                # ------------------------------------------------
                # 선택 병원의 route
                # ------------------------------------------------

                route_info = (
                    hospital_routes[
                        selected_action
                    ]
                )

                hospital_edge = (
                    route_info.get(
                        "edge_id"
                    )
                )

                route = (
                    route_info.get(
                        "route"
                    )
                )

                # route가 사전에 계산되지 않았으면
                # 다시 계산
                if (
                    route is None
                    and hospital_edge is not None
                ):

                    route = _find_route(
                        patient_edge,
                        hospital_edge,
                    )

                # ------------------------------------------------
                # route 실패
                # ------------------------------------------------

                if (
                    hospital_edge is None
                    or route is None
                    or len(route.edges) == 0
                ):

                    reward = calculate_reward(
                        travel_time=0.0,
                        severity=severity,
                        hospital=selected_hospital,
                        golden_limit=GOLDEN_TIME_LIMITS.get(
                            severity,
                            300.0,
                        ),
                        rejection_count=rejection_count,
                        occupancy_at_selection=occupancy_at_selection,
                        route_failed=True,
                    )

                    zero_state = np.zeros(
                        state_dim,
                        dtype=np.float32,
                    )

                    agent.store_transition(
                        state,
                        selected_action,
                        reward,
                        zero_state,
                        True,
                    )

                    loss = agent.train_step()

                    if loss is not None:
                        episode_losses.append(
                            loss
                        )

                    episode_reward += reward

                    failed_missions += 1

                    continue

                # ------------------------------------------------
                # 구급차 생성
                # ------------------------------------------------

                ambulance_id = (
                    f"ambulance_{episode}_{step}"
                )

                route_id = (
                    f"route_{episode}_{step}"
                )

                try:

                    if route_id in traci.route.getIDList():
                        traci.route.remove(
                            route_id
                        )

                except Exception:
                    pass

                try:

                    traci.route.add(
                        route_id,
                        list(route.edges),
                    )

                    traci.vehicle.add(
                        ambulance_id,
                        route_id,
                        typeID="ambulance_custom",
                    )

                except Exception:

                    reward = calculate_reward(
                        travel_time=0.0,
                        severity=severity,
                        hospital=selected_hospital,
                        golden_limit=GOLDEN_TIME_LIMITS.get(
                            severity,
                            300.0,
                        ),
                        rejection_count=rejection_count,
                        occupancy_at_selection=occupancy_at_selection,
                        route_failed=True,
                    )

                    zero_state = np.zeros(
                        state_dim,
                        dtype=np.float32,
                    )

                    agent.store_transition(
                        state,
                        selected_action,
                        reward,
                        zero_state,
                        True,
                    )

                    loss = agent.train_step()

                    if loss is not None:
                        episode_losses.append(
                            loss
                        )

                    episode_reward += reward

                    failed_missions += 1

                    continue

                # ------------------------------------------------
                # 실제 예상 이동시간
                # ------------------------------------------------

                estimated_travel_time = float(
                    max(
                        0.0,
                        route.travelTime,
                    )
                )

                # ------------------------------------------------
                # 목표 위치
                # ------------------------------------------------

                try:

                    target_edge = env.net.getEdge(
                        hospital_edge
                    )

                    target_pos = (
                        target_edge.getToNode().getCoord()
                    )

                except Exception:

                    target_pos = (
                        selected_hospital.get(
                            "sumo_x",
                            0.0,
                        ),
                        selected_hospital.get(
                            "sumo_y",
                            0.0,
                        ),
                    )

                # ------------------------------------------------
                # Mission 등록
                # ------------------------------------------------

                active_missions[
                    ambulance_id
                ] = {
                    "hospital": selected_hospital,
                    "severity": severity,

                    "elapsed_time": 0.0,

                    "estimated_travel_time":
                        estimated_travel_time,

                    "rejection_count":
                        rejection_count,

                    "occupancy_at_selection":
                        occupancy_at_selection,

                    "state":
                        state,

                    "action":
                        selected_action,

                    "target_pos":
                        target_pos,

                    "golden_limit":
                        GOLDEN_TIME_LIMITS.get(
                            severity,
                            300.0,
                        ),
                }

        except traci.TraCIException as exc:

            print(
                f"[Episode {episode}] "
                f"SUMO 오류: {exc}"
            )

        finally:

            # ----------------------------------------------------
            # 아직 도착하지 못한 임무 처리
            # ----------------------------------------------------

            for ambulance_id, mission in list(
                active_missions.items()
            ):

                reward = calculate_reward(
                    travel_time=mission[
                        "elapsed_time"
                    ],
                    severity=mission[
                        "severity"
                    ],
                    hospital=mission[
                        "hospital"
                    ],
                    golden_limit=mission[
                        "golden_limit"
                    ],
                    rejection_count=mission[
                        "rejection_count"
                    ],
                    occupancy_at_selection=mission[
                        "occupancy_at_selection"
                    ],
                    route_failed=True,
                )

                zero_state = np.zeros(
                    state_dim,
                    dtype=np.float32,
                )

                agent.store_transition(
                    mission["state"],
                    mission["action"],
                    reward,
                    zero_state,
                    True,
                )

                loss = agent.train_step()

                if loss is not None:
                    episode_losses.append(
                        loss
                    )

                episode_reward += reward

                failed_missions += 1

            active_missions.clear()

            if traci.isLoaded():

                try:
                    traci.close()
                except Exception:
                    pass

        # --------------------------------------------------------
        # Target Network
        # --------------------------------------------------------

        if episode % 5 == 0:
            agent.update_target_network()

        # --------------------------------------------------------
        # 모델 저장
        # --------------------------------------------------------

        if episode % 5 == 0:

            agent.save_model(
                MODEL_PATH
            )

        # --------------------------------------------------------
        # 통계
        # --------------------------------------------------------

        avg_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else 0.0
        )

        print(
            f"[Episode {episode:03d}] "
            f"Scenario={scenario['id']} | "
            f"Reward={episode_reward:8.2f} | "
            f"Completed={completed_missions:3d} | "
            f"Failed={failed_missions:3d} | "
            f"Reject={rejection_total:3d} | "
            f"Loss={avg_loss:.5f} | "
            f"Epsilon={agent.epsilon:.4f}"
        )

    # ------------------------------------------------------------
    # 최종 저장
    # ------------------------------------------------------------

    agent.save_model(
        MODEL_PATH
    )

    print()
    print("=" * 70)
    print("DQN 학습 완료")
    print(f"모델: {MODEL_PATH}")
    print("=" * 70)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    train_dqn_fast(
        num_episodes=50
    )