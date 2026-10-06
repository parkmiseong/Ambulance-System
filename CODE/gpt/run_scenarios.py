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
    calculate_reward,
)


# ============================================================
# 설정
# ============================================================

MAX_STEPS = 600

MAX_AMBULANCES = 30

MISSION_TIMEOUT_SECONDS = 350.0

ARRIVAL_DISTANCE = 120.0

# train_agent.py와 동일한 모델 사용
MODEL_PATH = "dqn_ambulance_model_v4.pth"

MAX_ESTIMATED_TRAVEL_TIME = 600.0

HOSPITAL_FULL_THRESHOLD = 0.85


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
    """
    SUMO/TraCI의 불필요한 출력 억제.
    """

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:

        with open(
            os.devnull,
            "w",
            encoding="utf-8",
        ) as devnull:

            sys.stdout = devnull
            sys.stderr = devnull

            yield

    finally:

        sys.stdout = original_stdout
        sys.stderr = original_stderr


# ============================================================
# SUMO 시작
# ============================================================

def start_sumo(
    env,
    scenario,
    seed_val,
):
    """
    SUMO를 시작하고 구급차 차량 유형을 등록한다.
    """

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

        traci.start(
            config
        )

        # ----------------------------------------------------
        # 구급차 차량 유형
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # 병원 좌표 갱신
    # --------------------------------------------------------

    env.update_hospital_coordinates()

    delta_t = float(
        traci.simulation.getDeltaT()
    )

    if delta_t <= 0:
        delta_t = 1.0

    return delta_t


# ============================================================
# 환자 발생 스케줄
# ============================================================

def generate_fixed_patient_schedule(
    scenario,
    valid_edges,
    seed_val,
):
    """
    동일한 seed를 사용하면 동일한 환자 발생 위치와
    중증도 스케줄을 생성한다.
    """

    random.seed(
        seed_val
    )

    np.random.seed(
        seed_val
    )

    usable_edges = [
        edge_id
        for edge_id in valid_edges
        if edge_id
        and not str(edge_id).startswith(":")
    ]

    schedule = {}

    if not usable_edges:
        return schedule

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        if random.random() < scenario["prob"]:

            schedule[step] = {
                "edge_id": random.choice(
                    usable_edges
                ),
                "severity": random.randint(
                    1,
                    4,
                ),
            }

    return schedule


# ============================================================
# 접근 가능한 Edge 찾기
# ============================================================

def get_reachable_edge_id(
    net,
    x,
    y,
    from_edge=None,
    vtype="ambulance_custom",
):
    """
    병원 좌표 주변에서 구급차가 접근할 수 있는
    SUMO Edge를 찾는다.
    """

    if x is None or y is None:
        return None

    radius = 100.0

    visited = set()

    while radius <= 5000.0:

        try:

            nearby_edges = (
                net.getNeighboringEdges(
                    x,
                    y,
                    radius,
                )
            )

        except Exception:

            nearby_edges = []

        nearby_edges = sorted(
            nearby_edges,
            key=lambda item: item[1],
        )

        for edge, _distance in nearby_edges:

            edge_id = edge.getID()

            # 내부 Edge 제외
            if str(edge_id).startswith(":"):
                continue

            if edge_id in visited:
                continue

            visited.add(
                edge_id
            )

            # 출발 Edge가 없는 경우
            if from_edge is None:
                return edge_id

            # 동일 Edge
            if edge_id == from_edge:
                return edge_id

            # 실제 route 가능 여부 확인
            try:

                with suppress_sumo_stdout():

                    route = (
                        traci.simulation.findRoute(
                            from_edge,
                            edge_id,
                            vType=vtype,
                            depart=-1,
                            routingMode=0,
                        )
                    )

                if (
                    route is not None
                    and len(route.edges) > 0
                ):

                    return edge_id

            except traci.TraCIException:

                continue

            except Exception:

                continue

        radius += 200.0

    return None


# ============================================================
# Route 계산
# ============================================================

def find_route(
    from_edge,
    to_edge,
):
    """
    환자 Edge → 병원 Edge의 실제 route를 계산한다.
    """

    if not from_edge or not to_edge:
        return None

    if str(from_edge).startswith(":"):
        return None

    if str(to_edge).startswith(":"):
        return None

    try:

        with suppress_sumo_stdout():

            route = (
                traci.simulation.findRoute(
                    from_edge,
                    to_edge,
                    vType="ambulance_custom",
                    depart=-1,
                    routingMode=0,
                )
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
# 병원 Route 계산
# ============================================================

def calculate_hospital_routes(
    env,
    patient_edge,
    hospitals,
):
    """
    모든 병원에 대해

    - hospital_edge
    - route_length
    - estimated_travel_time
    - reachable

    을 계산한다.
    """

    for hospital in hospitals:

        hospital[
            "reachable"
        ] = False

        hospital[
            "hospital_edge"
        ] = None

        hospital[
            "route_length"
        ] = None

        hospital[
            "estimated_travel_time"
        ] = None

        x = hospital.get(
            "sumo_x"
        )

        y = hospital.get(
            "sumo_y"
        )

        if x is None or y is None:
            continue

        try:

            hospital_edge = (
                get_reachable_edge_id(
                    env.net,
                    x,
                    y,
                    from_edge=patient_edge,
                    vtype="ambulance_custom",
                )
            )

            if hospital_edge is None:
                continue

            route = find_route(
                patient_edge,
                hospital_edge,
            )

            if route is None:
                continue

            route_length = float(
                getattr(
                    route,
                    "length",
                    0.0,
                )
            )

            travel_time = float(
                getattr(
                    route,
                    "travelTime",
                    0.0,
                )
            )

            # travelTime이 비정상적인 경우
            if travel_time <= 0:

                travel_time = max(
                    route_length / 13.9,
                    1.0,
                )

            # 최대 이동시간 제한
            travel_time = min(
                travel_time,
                MAX_ESTIMATED_TRAVEL_TIME,
            )

            hospital[
                "reachable"
            ] = True

            hospital[
                "hospital_edge"
            ] = hospital_edge

            hospital[
                "route_length"
            ] = route_length

            hospital[
                "estimated_travel_time"
            ] = travel_time

        except Exception:

            continue

    return hospitals


# ============================================================
# 병원 치료역량
# ============================================================

def encode_hospital_type(
    hospital,
):
    """
    병원 유형을 수치화한다.

    권역 = 1.0
    지역 = 0.8
    일반 = 0.3
    """

    hospital_type = str(
        hospital.get(
            "type",
            "",
        )
    )

    if "권역" in hospital_type:
        return 1.0

    if "지역" in hospital_type:
        return 0.8

    return 0.3


# ============================================================
# 환자-병원 적합도
# ============================================================

def get_hospital_suitability(
    severity,
    hospital,
):
    """
    환자 중증도와 병원 유형에 따른 적합도.
    """

    hospital_type = str(
        hospital.get(
            "type",
            "",
        )
    )

    # --------------------------------------------------------
    # 중증 환자
    # --------------------------------------------------------

    if severity >= 3:

        if "권역" in hospital_type:
            return 1.0

        if "지역" in hospital_type:
            return 0.8

        return 0.0

    # --------------------------------------------------------
    # 경증 / 중등도
    # --------------------------------------------------------

    if "권역" in hospital_type:
        return 0.8

    if "지역" in hospital_type:
        return 1.0

    return 0.7


# ============================================================
# State Vector
# ============================================================

def build_state_vector(
    patient_pos,
    severity,
    hospitals,
):
    """
    train_agent.py와 동일한 State 구조.

    환자 정보:
        3개

    병원별 정보:
        예상 이동시간
        occupancy
        치료역량
        환자-병원 적합도

    총 차원:
        3 + 병원 수 × 4
    """

    state = []

    # --------------------------------------------------------
    # 환자 정보
    # --------------------------------------------------------

    patient_x = (
        float(patient_pos[0])
        / 10000.0
    )

    patient_y = (
        float(patient_pos[1])
        / 10000.0
    )

    normalized_severity = (
        float(severity)
        / 4.0
    )

    state.extend([
        patient_x,
        patient_y,
        normalized_severity,
    ])

    # --------------------------------------------------------
    # 병원별 예상 이동시간
    # --------------------------------------------------------

    for hospital in hospitals:

        travel_time = hospital.get(
            "estimated_travel_time"
        )

        if travel_time is None:

            travel_time = (
                MAX_ESTIMATED_TRAVEL_TIME
            )

        travel_time = float(
            travel_time
        )

        normalized_time = min(
            max(
                travel_time,
                0.0,
            )
            / MAX_ESTIMATED_TRAVEL_TIME,
            1.0,
        )

        state.append(
            normalized_time
        )

    # --------------------------------------------------------
    # 병원별 occupancy
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
            min(
                1.0,
                occupancy,
            ),
        )

        state.append(
            occupancy
        )

    # --------------------------------------------------------
    # 병원별 치료역량
    # --------------------------------------------------------

    for hospital in hospitals:

        state.append(
            encode_hospital_type(
                hospital
            )
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
        + len(hospitals) * 4
    )

    if state.shape[0] != expected_dim:

        raise ValueError(
            "State dimension 오류: "
            f"현재={state.shape[0]}, "
            f"예상={expected_dim}"
        )

    return state


# ============================================================
# 병원 준비
# ============================================================

def prepare_hospitals(
    initial_hospitals,
    env,
    seed_val,
):
    """
    평가용 병원 데이터를 준비한다.

    학습과 동일하게 Episode마다
    초기 occupancy를 seed 기반으로 설정한다.
    """

    env.update_hospital_coordinates()

    env_hospitals = (
        env.get_hospitals()
    )

    hospitals = copy.deepcopy(
        initial_hospitals
    )

    coordinate_map = {}

    for hospital in env_hospitals:

        hospital_id = hospital.get(
            "id"
        )

        coordinate_map[
            hospital_id
        ] = (
            hospital.get("sumo_x"),
            hospital.get("sumo_y"),
        )

    # --------------------------------------------------------
    # 좌표 + occupancy 설정
    # --------------------------------------------------------

    for hospital_index, hospital in enumerate(
        hospitals
    ):

        hospital_id = hospital.get(
            "id"
        )

        if hospital_id in coordinate_map:

            x, y = coordinate_map[
                hospital_id
            ]

            hospital["sumo_x"] = x
            hospital["sumo_y"] = y

        # train_agent.py와 동일한 방식
        hospital_rng = random.Random(
            seed_val
            + hospital_index
        )

        hospital[
            "occupancy"
        ] = hospital_rng.uniform(
            0.2,
            0.6,
        )

    return hospitals


# ============================================================
# 선택 가능한 Action
# ============================================================

def get_valid_actions(
    hospitals,
):
    """
    실제 route가 존재하면서
    병원 점유율이 85% 미만인 병원만 선택 가능.
    """

    valid_actions = []

    for index, hospital in enumerate(
        hospitals
    ):

        reachable = hospital.get(
            "reachable",
            False,
        )

        travel_time = hospital.get(
            "estimated_travel_time"
        )

        occupancy = float(
            hospital.get(
                "occupancy",
                1.0,
            )
        )

        if not reachable:
            continue

        if travel_time is None:
            continue

        if (
            occupancy
            >= HOSPITAL_FULL_THRESHOLD
        ):
            continue

        valid_actions.append(
            index
        )

    return valid_actions


# ============================================================
# Fallback Action
# ============================================================

def get_fallback_actions(
    hospitals,
):
    """
    모든 병원이 85% 이상이면
    reachable 병원 중 가장 낮은 occupancy 병원을
    fallback 후보로 사용한다.
    """

    reachable = [
        index
        for index, hospital in enumerate(
            hospitals
        )
        if hospital.get(
            "reachable",
            False,
        )
    ]

    if not reachable:
        return []

    min_occupancy = min(
        float(
            hospitals[index].get(
                "occupancy",
                1.0,
            )
        )
        for index in reachable
    )

    return [
        index
        for index in reachable
        if float(
            hospitals[index].get(
                "occupancy",
                1.0,
            )
        )
        <= min_occupancy + 1e-6
    ]


# ============================================================
# Action 선택
# ============================================================

def select_action(
    mode,
    router,
    severity,
    hospitals,
    state,
    agent,
    valid_actions,
):
    """
    DQN과 Baseline의 Action 선택을 통일한다.
    """

    if not valid_actions:
        return None

    # --------------------------------------------------------
    # DQN
    # --------------------------------------------------------

    if mode == "dqn":

        action = agent.select_action(
            state,
            valid_actions,
        )

        return int(action)

    # --------------------------------------------------------
    # 최단시간
    # --------------------------------------------------------

    if mode == "shortest_time":

        return int(
            router.shortest_time_strategy(
                valid_actions
            )
        )

    # --------------------------------------------------------
    # 최단거리
    # --------------------------------------------------------

    if mode == "shortest_distance":

        return int(
            router.shortest_distance_strategy(
                valid_actions
            )
        )

    # --------------------------------------------------------
    # 병상 기반
    # --------------------------------------------------------

    if mode == "bed":

        return int(
            router.bed_strategy(
                severity,
                valid_actions,
            )
        )

    # --------------------------------------------------------
    # 치료역량 기반
    # --------------------------------------------------------

    if mode == "rule":

        return int(
            router.rule_based_strategy(
                severity,
                valid_actions,
            )
        )

    # --------------------------------------------------------
    # 종합 휴리스틱
    # --------------------------------------------------------

    if mode == "heuristic":

        return int(
            router.heuristic_strategy(
                severity,
                valid_actions,
            )
        )

    raise ValueError(
        f"알 수 없는 평가 전략: {mode}"
    )


# ============================================================
# 단일 시나리오 평가
# ============================================================

def run_single_scenario(
    scenario,
    initial_hospitals,
    seed_val,
    patient_schedule,
    mode="dqn",
    agent=None,
):
    """
    하나의 시나리오에 대해 하나의 전략을 평가한다.
    """

    env = SumoMedicalEnvironment()

    total_reward = 0.0

    completed_missions = 0

    golden_success_count = 0

    total_rejections = 0

    total_transfer_time = 0.0

    hospital_assignment_counts = {}

    active_missions = {}

    spawned_count = 0

    try:

        # ====================================================
        # SUMO 시작
        # ====================================================

        step_length = start_sumo(
            env,
            scenario,
            seed_val,
        )

        # ====================================================
        # 병원 준비
        # ====================================================

        hospitals = prepare_hospitals(
            initial_hospitals,
            env,
            seed_val,
        )

        # ====================================================
        # Baseline Router
        # ====================================================

        router = HospitalRouters(
            hospitals
        )

        # ====================================================
        # 병원 배정 횟수
        # ====================================================

        hospital_assignment_counts = {
            hospital.get("id", index): 0
            for index, hospital in enumerate(
                hospitals
            )
        }

        # ====================================================
        # SUMO Simulation
        # ====================================================

        for step in range(
            1,
            MAX_STEPS + 1,
        ):

            try:

                traci.simulationStep()

            except traci.TraCIException:

                break

            # =================================================
            # 병원 occupancy 감소
            # =================================================

            for hospital in hospitals:

                current_occupancy = float(
                    hospital.get(
                        "occupancy",
                        0.0,
                    )
                )

                hospital[
                    "occupancy"
                ] = max(
                    0.0,
                    current_occupancy
                    - 0.0005,
                )

            # =================================================
            # 현재 구급차
            # =================================================

            ambulance_ids = set(
                traci.vehicle.getIDList()
            )

            # =================================================
            # 기존 Mission 처리
            # =================================================

            finished_ids = []

            for (
                ambulance_id,
                mission,
            ) in list(
                active_missions.items()
            ):

                # --------------------------------------------
                # 차량이 사라진 경우
                # --------------------------------------------

                if (
                    ambulance_id
                    not in ambulance_ids
                ):

                    travel_time = (
                        mission[
                            "elapsed_time"
                        ]
                    )

                    reward = calculate_reward(
                        travel_time=travel_time,
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

                    total_reward += reward

                    completed_missions += 1

                    total_transfer_time += (
                        travel_time
                    )

                    finished_ids.append(
                        ambulance_id
                    )

                    continue

                # --------------------------------------------
                # 실제 경과시간
                # --------------------------------------------

                mission[
                    "elapsed_time"
                ] += step_length

                # --------------------------------------------
                # 구급차 위치
                # --------------------------------------------

                try:

                    vehicle_pos = (
                        traci.vehicle.getPosition(
                            ambulance_id
                        )
                    )

                    target_pos = mission[
                        "target_pos"
                    ]

                    distance = float(
                        np.hypot(
                            vehicle_pos[0]
                            - target_pos[0],
                            vehicle_pos[1]
                            - target_pos[1],
                        )
                    )

                except Exception:

                    distance = float(
                        "inf"
                    )

                arrived = (
                    distance
                    <= ARRIVAL_DISTANCE
                )

                timeout = (
                    mission[
                        "elapsed_time"
                    ]
                    >= MISSION_TIMEOUT_SECONDS
                )

                if not arrived and not timeout:
                    continue

                # --------------------------------------------
                # Mission 종료
                # --------------------------------------------

                travel_time = (
                    mission[
                        "elapsed_time"
                    ]
                )

                golden_limit = (
                    mission[
                        "golden_limit"
                    ]
                )

                # --------------------------------------------
                # 골든타임 성공
                # --------------------------------------------

                if (
                    arrived
                    and travel_time
                    <= golden_limit
                ):

                    golden_success_count += 1

                # --------------------------------------------
                # Reward
                # --------------------------------------------

                reward = calculate_reward(
                    travel_time=travel_time,
                    severity=mission[
                        "severity"
                    ],
                    hospital=mission[
                        "hospital"
                    ],
                    golden_limit=golden_limit,
                    rejection_count=mission[
                        "rejection_count"
                    ],
                    occupancy_at_selection=mission[
                        "occupancy_at_selection"
                    ],
                    route_failed=not arrived,
                )

                total_reward += reward

                completed_missions += 1

                total_transfer_time += (
                    travel_time
                )

                finished_ids.append(
                    ambulance_id
                )

            # =================================================
            # Mission 제거
            # =================================================

            for ambulance_id in finished_ids:

                active_missions.pop(
                    ambulance_id,
                    None,
                )

                try:

                    if (
                        ambulance_id
                        in traci.vehicle.getIDList()
                    ):

                        traci.vehicle.remove(
                            ambulance_id
                        )

                except Exception:
                    pass

            # =================================================
            # 신규 환자
            # =================================================

            patient = (
                patient_schedule.get(
                    step
                )
            )

            if patient is None:
                continue

            # =================================================
            # 최대 구급차 수
            # =================================================

            if (
                len(active_missions)
                >= MAX_AMBULANCES
            ):
                continue

            patient_edge = (
                patient["edge_id"]
            )

            severity = int(
                patient["severity"]
            )

            # =================================================
            # 환자 Edge 검증
            # =================================================

            try:

                patient_edge_obj = (
                    env.net.getEdge(
                        patient_edge
                    )
                )

            except Exception:

                continue

            if patient_edge_obj is None:
                continue

            if str(
                patient_edge
            ).startswith(":"):

                continue

            # =================================================
            # 환자 위치
            # =================================================

            try:

                patient_pos = (
                    patient_edge_obj
                    .getFromNode()
                    .getCoord()
                )

            except Exception:

                continue

            # =================================================
            # 병원 Route 계산
            # =================================================

            hospitals = (
                calculate_hospital_routes(
                    env,
                    patient_edge,
                    hospitals,
                )
            )

            # =================================================
            # Valid Actions
            # =================================================

            valid_actions = (
                get_valid_actions(
                    hospitals
                )
            )

            # =================================================
            # 모든 병원이 포화된 경우
            # =================================================

            if not valid_actions:

                valid_actions = (
                    get_fallback_actions(
                        hospitals
                    )
                )

            if not valid_actions:
                continue

            # =================================================
            # State 생성
            # =================================================

            state = (
                build_state_vector(
                    patient_pos,
                    severity,
                    hospitals,
                )
            )

            # =================================================
            # Action 선택
            # =================================================

            try:

                action = select_action(
                    mode=mode,
                    router=router,
                    severity=severity,
                    hospitals=hospitals,
                    state=state,
                    agent=agent,
                    valid_actions=valid_actions,
                )

            except Exception:

                continue

            if action is None:
                continue

            action = int(
                action
            )

            # =================================================
            # 잘못된 Action 처리
            # =================================================

            rejection_count = 0

            if action not in valid_actions:

                # ------------------------------------------------
                # 유효하지 않은 병원을 선택한 경우
                # 평가에서는 임의로 다른 병원으로 바꾸지 않고
                # rejection으로 기록한다.
                # ------------------------------------------------

                rejection_count = 1

                total_rejections += 1

                # 평가를 계속할 수 있도록
                # 가장 낮은 occupancy 병원을 실제 이송 대상으로 사용
                fallback_actions = (
                    get_fallback_actions(
                        hospitals
                    )
                )

                if not fallback_actions:
                    continue

                action = min(
                    fallback_actions,
                    key=lambda index:
                        hospitals[index].get(
                            "estimated_travel_time",
                            MAX_ESTIMATED_TRAVEL_TIME,
                        ),
                )

            # =================================================
            # 선택 병원
            # =================================================

            selected_hospital = (
                hospitals[action]
            )

            # =================================================
            # 선택 당시 occupancy
            # =================================================

            occupancy_at_selection = float(
                selected_hospital.get(
                    "occupancy",
                    0.0,
                )
            )

            # =================================================
            # 병원 occupancy 증가
            # =================================================

            selected_hospital[
                "occupancy"
            ] = min(
                1.0,
                occupancy_at_selection
                + 0.08,
            )

            # =================================================
            # 병원 배정 횟수
            # =================================================

            hospital_id = (
                selected_hospital.get(
                    "id",
                    action,
                )
            )

            if (
                hospital_id
                not in hospital_assignment_counts
            ):

                hospital_assignment_counts[
                    hospital_id
                ] = 0

            hospital_assignment_counts[
                hospital_id
            ] += 1

            # =================================================
            # 병원 Edge
            # =================================================

            hospital_edge = (
                selected_hospital.get(
                    "hospital_edge"
                )
            )

            if hospital_edge is None:

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

                total_reward += reward

                completed_missions += 1

                continue

            # =================================================
            # 실제 Route
            # =================================================

            route = find_route(
                patient_edge,
                hospital_edge,
            )

            if route is None:

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

                total_reward += reward

                completed_missions += 1

                continue

            if len(route.edges) == 0:

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

                total_reward += reward

                completed_missions += 1

                continue

            # =================================================
            # 구급차 생성
            # =================================================

            spawned_count += 1

            ambulance_id = (
                f"amb_eval_"
                f"{step}_"
                f"{spawned_count}"
            )

            route_id = (
                f"route_eval_"
                f"{step}_"
                f"{spawned_count}"
            )

            try:

                if (
                    route_id
                    in traci.route.getIDList()
                ):

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
                    depart="now",
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

                total_reward += reward

                completed_missions += 1

                continue

            # =================================================
            # 실제 목적지 좌표
            # =================================================

            try:

                target_edge = (
                    env.net.getEdge(
                        hospital_edge
                    )
                )

                target_pos = (
                    target_edge
                    .getToNode()
                    .getCoord()
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

            # =================================================
            # Mission 저장
            # =================================================

            active_missions[
                ambulance_id
            ] = {

                "severity":
                    severity,

                "hospital":
                    selected_hospital,

                "elapsed_time":
                    0.0,

                "rejection_count":
                    rejection_count,

                "occupancy_at_selection":
                    occupancy_at_selection,

                "state":
                    state,

                "action":
                    action,

                "target_pos":
                    target_pos,

                "golden_limit":
                    GOLDEN_TIME_LIMITS.get(
                        severity,
                        300.0,
                    ),
            }

        # =====================================================
        # Simulation 종료
        # =====================================================

        for (
            ambulance_id,
            mission,
        ) in list(
            active_missions.items()
        ):

            travel_time = (
                mission[
                    "elapsed_time"
                ]
            )

            reward = calculate_reward(
                travel_time=travel_time,
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

            total_reward += reward

            completed_missions += 1

            total_transfer_time += (
                travel_time
            )

        active_missions.clear()

    except Exception as exc:

        return {
            "reward":
                total_reward,

            "avg_reward":
                0.0,

            "golden_success_rate":
                0.0,

            "rejections":
                total_rejections,

            "avg_time":
                0.0,

            "assignment_std":
                0.0,

            "completed_missions":
                completed_missions,

            "zero_reason":
                str(exc),
        }

    finally:

        try:

            if traci.isLoaded():

                traci.close()

        except Exception:
            pass

    # =========================================================
    # 평가 지표 계산
    # =========================================================

    if completed_missions > 0:

        avg_reward = (
            total_reward
            / completed_missions
        )

        avg_time = (
            total_transfer_time
            / completed_missions
        )

    else:

        avg_reward = 0.0

        avg_time = 0.0

    # ---------------------------------------------------------
    # 골든타임 성공률
    # ---------------------------------------------------------

    if completed_missions > 0:

        golden_success_rate = (
            golden_success_count
            / completed_missions
            * 100.0
        )

    else:

        golden_success_rate = 0.0

    # ---------------------------------------------------------
    # 병원 배정 편차
    # ---------------------------------------------------------

    if hospital_assignment_counts:

        assignment_std = float(
            np.std(
                list(
                    hospital_assignment_counts.values()
                )
            )
        )

    else:

        assignment_std = 0.0

    return {
        "reward":
            total_reward,

        "avg_reward":
            avg_reward,

        "golden_success_rate":
            golden_success_rate,

        "rejections":
            total_rejections,

        "avg_time":
            avg_time,

        "assignment_std":
            assignment_std,

        "completed_missions":
            completed_missions,

        "zero_reason":
            "",
    }


# ============================================================
# 결과 출력
# ============================================================

def print_result(
    label,
    result,
):
    """
    하나의 평가 결과를 출력한다.
    """

    print(
        f"  · {label:<16}"
        f"| 보상 {result['reward']:9.1f} "
        f"| 평균보상 {result['avg_reward']:7.2f} "
        f"| 골든타임 "
        f"{result['golden_success_rate']:6.1f}% "
        f"| 거부 {result['rejections']:4d} "
        f"| 평균시간 "
        f"{result['avg_time']:7.1f}s "
        f"| 완료 "
        f"{result['completed_missions']:4d} "
        f"| 배정편차 "
        f"{result['assignment_std']:6.2f}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 100
    )

    print(
        "시나리오 기반 응급실 병원 선택 평가"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # Environment
    # ========================================================

    temp_env = (
        SumoMedicalEnvironment()
    )

    valid_edges = (
        temp_env.get_valid_edges()
    )

    initial_hospitals = copy.deepcopy(
        temp_env.get_hospitals()
    )

    action_dim = len(
        initial_hospitals
    )

    # ========================================================
    # State Dimension
    # ========================================================

    state_dim = (
        3
        + action_dim * 4
    )

    print(
        f"[State] 차원       : {state_dim}"
    )

    print(
        f"[Action] 병원 수   : {action_dim}"
    )

    print(
        f"[Model]            : {MODEL_PATH}"
    )

    # ========================================================
    # DQN Agent
    # ========================================================

    dqn_agent = (
        DQNAmbulanceAgent(
            state_dim,
            action_dim,
        )
    )

    # --------------------------------------------------------
    # 모델 로드
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        print()
        print(
            "[오류]"
        )

        print(
            f"모델 파일이 없습니다: "
            f"{MODEL_PATH}"
        )

        print(
            "먼저 train_agent.py를 실행하여 "
            "모델을 학습하십시오."
        )

        sys.exit(1)

    try:

        loaded = (
            dqn_agent.load_model(
                MODEL_PATH
            )
        )

    except Exception as exc:

        print()
        print(
            "[오류] DQN 모델 로드 실패"
        )

        print(
            exc
        )

        sys.exit(1)

    if not loaded:

        print()
        print(
            "[오류] DQN 모델을 불러오지 못했습니다."
        )

        sys.exit(1)

    # ========================================================
    # 평가 모드
    # ========================================================

    dqn_agent.epsilon = 0.0

    try:

        dqn_agent.model.eval()

    except Exception:
        pass

    print(
        "[DQN] 평가 모드: "
        "epsilon = 0.0"
    )

    # ========================================================
    # 평가 전략
    # ========================================================

    strategies = [
        (
            "shortest_distance",
            "최단거리",
        ),
        (
            "shortest_time",
            "최단시간",
        ),
        (
            "bed",
            "병상기반",
        ),
        (
            "rule",
            "치료역량기반",
        ),
        (
            "heuristic",
            "종합휴리스틱",
        ),
        (
            "dqn",
            "DQN",
        ),
    ]

    print()
    print(
        "[평가 전략]"
    )

    for mode, label in strategies:

        print(
            f"  - {label}"
        )

    # ========================================================
    # 시나리오 선택
    # ========================================================

    print()
    print(
        "[시나리오]"
    )

    for scenario in SCENARIOS:

        print(
            f"  [{scenario['id']}] "
            f"환자={scenario['patient_freq']} / "
            f"교통={scenario['traffic']} / "
            f"복잡도={scenario['complexity']}"
        )

    print(
        "  [0] 전체"
    )

    selected_input = input(
        "\n실행할 시나리오 번호 "
        "(0~9): "
    ).strip()

    # ========================================================
    # 전체 시나리오
    # ========================================================

    if selected_input == "0":

        selected_scenarios = (
            SCENARIOS
        )

    else:

        try:

            scenario_id = int(
                selected_input
            )

            selected_scenarios = [
                scenario
                for scenario in SCENARIOS
                if scenario["id"]
                == scenario_id
            ]

        except ValueError:

            selected_scenarios = []

    if not selected_scenarios:

        print()
        print(
            "[오류] "
            "올바른 시나리오 번호가 아닙니다."
        )

        sys.exit(1)

    # ========================================================
    # 전체 결과 저장
    # ========================================================

    overall = {}

    for mode, label in strategies:

        overall[mode] = {

            "reward":
                0.0,

            "golden":
                0.0,

            "rejections":
                0,

            "avg_time":
                0.0,

            "assignment_std":
                0.0,

            "completed":
                0,
        }

    # ========================================================
    # 시나리오 실행
    # ========================================================

    for scenario in selected_scenarios:

        print()
        print(
            "=" * 100
        )

        print(
            f"시나리오 {scenario['id']} "
            f"평가 시작"
        )

        print(
            "=" * 100
        )

        # ----------------------------------------------------
        # train_agent.py와 동일한 seed 체계
        # ----------------------------------------------------

        seed_val = (
            10000
            + scenario["id"]
        )

        # ----------------------------------------------------
        # 환자 발생 스케줄
        # ----------------------------------------------------

        patient_schedule = (
            generate_fixed_patient_schedule(
                scenario,
                valid_edges,
                seed_val,
            )
        )

        print(
            f"환자 발생 이벤트: "
            f"{len(patient_schedule)}"
        )

        # ====================================================
        # 각 전략 평가
        # ====================================================

        for mode, label in strategies:

            print()

            print(
                f"[{label}]"
            )

            result = (
                run_single_scenario(
                    scenario=scenario,

                    initial_hospitals=
                        initial_hospitals,

                    seed_val=seed_val,

                    patient_schedule=
                        patient_schedule,

                    mode=mode,

                    agent=(
                        dqn_agent
                        if mode == "dqn"
                        else None
                    ),
                )
            )

            print_result(
                label,
                result,
            )

            # ------------------------------------------------
            # 전체 결과 누적
            # ------------------------------------------------

            overall[
                mode
            ]["reward"] += (
                result["reward"]
            )

            overall[
                mode
            ]["golden"] += (
                result[
                    "golden_success_rate"
                ]
            )

            overall[
                mode
            ]["rejections"] += (
                result[
                    "rejections"
                ]
            )

            overall[
                mode
            ]["avg_time"] += (
                result[
                    "avg_time"
                ]
            )

            overall[
                mode
            ]["assignment_std"] += (
                result[
                    "assignment_std"
                ]
            )

            overall[
                mode
            ]["completed"] += (
                result[
                    "completed_missions"
                ]
            )

            # ------------------------------------------------
            # 예외적인 평가 실패 표시
            # ------------------------------------------------

            if result.get(
                "zero_reason",
                "",
            ):

                print(
                    f"    평가 메시지: "
                    f"{result['zero_reason']}"
                )

    # ========================================================
    # 최종 결과
    # ========================================================

    scenario_count = len(
        selected_scenarios
    )

    print()
    print(
        "=" * 110
    )

    print(
        "최종 평가 결과"
    )

    print(
        "=" * 110
    )

    print(
        f"{'전략':<18}"
        f"| {'총보상':>12}"
        f"| {'평균 골든타임':>13}"
        f"| {'총 거부':>8}"
        f"| {'평균 이송시간':>15}"
        f"| {'총 완료':>10}"
        f"| {'배정편차':>10}"
    )

    print(
        "-" * 110
    )

    for mode, label in strategies:

        data = overall[
            mode
        ]

        average_golden = (
            data["golden"]
            / scenario_count
        )

        average_time = (
            data["avg_time"]
            / scenario_count
        )

        average_assignment_std = (
            data["assignment_std"]
            / scenario_count
        )

        print(
            f"{label:<18}"
            f"| {data['reward']:12.1f}"
            f"| {average_golden:12.1f}%"
            f"| {data['rejections']:8d}"
            f"| {average_time:14.1f}s"
            f"| {data['completed']:10d}"
            f"| {average_assignment_std:10.2f}"
        )

    print(
        "=" * 110
    )

    print()
    print(
        "평가 완료"
    )