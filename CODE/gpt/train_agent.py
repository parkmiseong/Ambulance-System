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
# 학습 설정
# ============================================================

# 수정된 모델은 v4로 별도 저장
MODEL_PATH = "dqn_ambulance_model_v4.pth"

# 한 Episode에서 진행할 최대 SUMO step
MAX_STEPS = 500

# 동시에 운용할 수 있는 최대 구급차 수
MAX_AMBULANCES = 30

# 하나의 이송이 지나치게 오래 지속되는 것을 방지
MISSION_TIMEOUT_SECONDS = 350.0

# 병원 도착 판정 거리
ARRIVAL_DISTANCE = 120.0

# 병원 혼잡도 기준
# 이 값 이상이면 일반적인 선택 후보에서 제외
HOSPITAL_FULL_THRESHOLD = 0.85

# 상태 벡터에서 사용하는 예상 이동시간의 최대값
# 600초 이상은 600초로 clipping
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
    """
    SUMO/TraCI에서 발생하는 불필요한 stdout/stderr 출력 억제.
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

    # --------------------------------------------------------
    # 출발 Edge가 지정된 경우 먼저 확인
    # --------------------------------------------------------

    if from_edge is not None:

        try:

            from_edge_obj = net.getEdge(
                from_edge
            )

            lanes = from_edge_obj.getLanes()

            allowed = False

            for lane in lanes:

                try:

                    permissions = (
                        lane.getPermissions()
                    )

                    # permission 정보가 없으면
                    # 일반적으로 접근 가능하다고 판단
                    if not permissions:
                        allowed = True
                        break

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

    # --------------------------------------------------------
    # 주변 Edge 탐색
    # --------------------------------------------------------

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

            # SUMO 내부 Edge 제외
            if edge_id.startswith(":"):
                continue

            # 이미 검사한 Edge 제외
            if edge_id in visited_edges:
                continue

            visited_edges.add(edge_id)

            # ------------------------------------------------
            # 출발 Edge와 동일한 경우
            # ------------------------------------------------

            if from_edge is not None:

                if edge_id == from_edge:
                    return edge_id

            # ------------------------------------------------
            # 출발 Edge가 없는 경우
            # ------------------------------------------------

            if from_edge is None:
                return edge_id

            # ------------------------------------------------
            # 실제 접근 가능 여부 확인
            # ------------------------------------------------

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
# 환자 발생 스케줄
# ============================================================

def generate_fixed_patient_schedule(
    scenario,
    valid_edges,
    seed_val,
):
    """
    동일 seed를 사용하면 동일한 환자 발생 스케줄을 생성한다.

    학습 결과의 재현성을 확보하기 위한 함수.
    """

    random.seed(seed_val)
    np.random.seed(seed_val)

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

def encode_hospital_type(
    hospital,
):
    """
    병원 유형을 수치화한다.

    권역응급의료센터 → 1.0
    지역응급의료센터 → 0.8
    일반 응급실     → 0.3
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
# 환자-병원 치료 적합도
# ============================================================

def get_hospital_suitability(
    severity,
    hospital,
):
    """
    환자 중증도와 병원 치료역량의 적합도 계산.

    중증 환자:
        권역 = 1.0
        지역 = 0.8
        일반 = 0.0

    경증/중등도:
        권역 = 0.8
        지역 = 1.0
        일반 = 0.7
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
    # 경증 / 중등도 환자
    # --------------------------------------------------------

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
    예상 이동시간 경로 계산.

    반환값:

    [
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

        x = hospital.get(
            "sumo_x"
        )

        y = hospital.get(
            "sumo_y"
        )

        result = {
            "edge_id": None,
            "route": None,
            "travel_time":
                MAX_ESTIMATED_TRAVEL_TIME,
        }

        if x is None or y is None:

            hospital_routes.append(
                result
            )

            continue

        # ----------------------------------------------------
        # 병원 좌표 → SUMO Edge
        # ----------------------------------------------------

        hospital_edge = (
            get_reachable_edge_id(
                net,
                x,
                y,
                from_edge=patient_edge,
                vtype="ambulance_custom",
            )
        )

        if hospital_edge is None:

            hospital_routes.append(
                result
            )

            continue

        result["edge_id"] = (
            hospital_edge
        )

        # ----------------------------------------------------
        # 실제 route 계산
        # ----------------------------------------------------

        try:

            with suppress_sumo_stdout():

                route = (
                    traci.simulation.findRoute(
                        patient_edge,
                        hospital_edge,
                        vType="ambulance_custom",
                        depart=-1,
                        routingMode=0,
                    )
                )

            if (
                route
                and len(route.edges) > 0
            ):

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

        hospital_routes.append(
            result
        )

    return hospital_routes


# ============================================================
# State Vector
# ============================================================

def build_state_vector(
    patient_pos,
    severity,
    hospitals,
    hospital_routes,
):
    """
    DQN 상태 벡터.

    상태 구성:

    [환자 정보]
        1. 환자 X
        2. 환자 Y
        3. 중증도

    [병원별 정보]
        4. 예상 이동시간
        5. 병상 점유율
        6. 치료역량
        7. 환자-병원 적합도

    따라서 전체 State Dimension은

        3 + (병원 수 × 4)

    이 된다.
    """

    hospital_count = len(
        hospitals
    )

    # --------------------------------------------------------
    # 환자 위치
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
            travel_time
            / MAX_ESTIMATED_TRAVEL_TIME,
            1.0,
        )

        state.append(
            normalized_time
        )

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
    """
    SUMO 실행 및 ambulance_custom 차량 유형 생성.
    """

    sumo_binary = (
        sumolib.checkBinary(
            "sumo"
        )
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
        # 구급차 전용 차량 유형
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

    env.update_hospital_coordinates()

    step_length = float(
        traci.simulation.getDeltaT()
    )

    if step_length <= 0:
        step_length = 1.0

    return step_length


# ============================================================
# 실제 이동용 최단시간 경로
# ============================================================

def _find_route(
    from_edge,
    to_edge,
):
    """
    출발 Edge → 목적지 Edge의
    이동시간 최소 경로를 계산한다.

    오류 발생 시 None 반환.
    SUMO 내부 오류 출력은 억제한다.
    """

    if not from_edge:
        return None

    if not to_edge:
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
# Valid Actions
# ============================================================

def _get_valid_actions(
    hospitals,
    severity,
):
    """
    현재 상태에서 선택 가능한 병원 Action.

    병상 점유율이 85% 이상이면
    일반적인 선택 후보에서 제외한다.

    치료역량은 강제로 mask하지 않고
    State + Reward를 통해 학습하도록 한다.
    """

    valid_actions = []

    for index, hospital in enumerate(
        hospitals
    ):

        occupancy = float(
            hospital.get(
                "occupancy",
                0.0,
            )
        )

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

def _get_fallback_actions(
    hospitals,
):
    """
    모든 병원이 가득 찬 경우
    가장 여유 있는 병원을 선택 후보로 반환.
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
        for index, hospital in enumerate(
            hospitals
        )
        if float(
            hospital.get(
                "occupancy",
                1.0,
            )
        )
        <= min_occupancy + 1e-6
    ]


# ============================================================
# 학습
# ============================================================

def train_dqn_fast(
    num_episodes=50,
):
    """
    DQN 학습 함수.

    주요 변경사항:

    1. 기존 v4 모델이 있으면 이어서 학습
    2. epsilon을 강제로 1.0으로 초기화하지 않음
    3. 병원 occupancy 초기값을 seed 기반으로 생성
    4. 평균 이송시간 계산
    5. 골든타임 성공률 계산
    """

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = SumoMedicalEnvironment()

    valid_edges = (
        env.get_valid_edges()
    )

    temp_hospitals = (
        env.get_hospitals()
    )

    action_dim = len(
        temp_hospitals
    )

    # --------------------------------------------------------
    # State Dimension
    # --------------------------------------------------------

    state_dim = (
        3
        + action_dim * 4
    )

    # --------------------------------------------------------
    # DQN Agent
    # --------------------------------------------------------

    agent = DQNAmbulanceAgent(
        state_dim,
        action_dim,
    )

    # --------------------------------------------------------
    # 기존 모델이 있으면 이어서 학습
    # --------------------------------------------------------

    if os.path.exists(
        MODEL_PATH
    ):

        print()
        print(
            f"[모델 발견] {MODEL_PATH}"
        )

        try:

            loaded = (
                agent.load_model(
                    MODEL_PATH
                )
            )

            if loaded:

                print(
                    "기존 모델의 학습 상태를 "
                    "불러와 이어서 학습합니다."
                )

            else:

                print(
                    "기존 모델을 불러오지 못했습니다."
                )

                print(
                    "새로운 모델로 학습합니다."
                )

        except Exception as exc:

            print(
                "기존 모델 로드 실패:"
            )

            print(
                f"  {exc}"
            )

            print(
                "새로운 모델로 학습합니다."
            )

    else:

        print(
            "기존 모델이 없습니다."
        )

        print(
            "새로운 DQN 모델로 학습을 시작합니다."
        )

    # --------------------------------------------------------
    # 학습 시작 출력
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "DQN 구급차 병원 선택 학습 시작"
    )
    print(
        "=" * 70
    )
    print(
        f"Model Path      : {MODEL_PATH}"
    )
    print(
        f"State Dimension : {state_dim}"
    )
    print(
        f"Action Dimension: {action_dim}"
    )
    print(
        f"Episodes        : {num_episodes}"
    )
    print(
        f"Hospital Count  : {action_dim}"
    )
    print(
        "=" * 70
    )

    # ========================================================
    # Episode
    # ========================================================

    for episode in range(
        1,
        num_episodes + 1,
    ):

        # ----------------------------------------------------
        # Scenario 선택
        # ----------------------------------------------------

        scenario = random.choice(
            SCENARIOS
        )

        # ----------------------------------------------------
        # Episode별 seed
        # ----------------------------------------------------

        seed_val = (
            10000
            + episode
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

        # ----------------------------------------------------
        # 병원 정보 복사
        # ----------------------------------------------------

        hospitals = copy.deepcopy(
            temp_hospitals
        )

        # ----------------------------------------------------
        # 수정:
        # seed 기반으로 occupancy 생성
        #
        # 기존:
        # random.uniform(0.2, 0.6)
        #
        # 문제:
        # 실행할 때마다 병원 상태가 달라짐
        #
        # 수정:
        # episode seed + hospital index
        # ----------------------------------------------------

        for hospital_index, hospital in enumerate(
            hospitals
        ):

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

        # ----------------------------------------------------
        # Mission
        # ----------------------------------------------------

        active_missions = {}

        # ----------------------------------------------------
        # Episode 통계
        # ----------------------------------------------------

        episode_reward = 0.0

        episode_losses = []

        completed_missions = 0

        failed_missions = 0

        rejection_total = 0

        # ----------------------------------------------------
        # 수정:
        # 평균 이송시간 계산용
        # 실제 도착한 임무만 포함
        # ----------------------------------------------------

        episode_travel_times = []

        # ----------------------------------------------------
        # 수정:
        # 골든타임 성공 횟수
        # ----------------------------------------------------

        golden_success_count = 0

        step_length = 1.0

        # ====================================================
        # SUMO 실행
        # ====================================================

        try:

            step_length = _start_sumo(
                env,
                scenario,
                seed_val,
            )

            # ------------------------------------------------
            # Simulation Step
            # ------------------------------------------------

            for step in range(
                1,
                MAX_STEPS + 1,
            ):

                traci.simulationStep()

                # ============================================
                # 병원 점유율 자연 감소
                # ============================================

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

                # ============================================
                # 현재 구급차 확인
                # ============================================

                ambulance_ids = [
                    vid
                    for vid in (
                        traci.vehicle.getIDList()
                    )
                    if "ambulance"
                    in vid.lower()
                ]

                # ============================================
                # 기존 Mission 처리
                # ============================================

                finished_ids = []

                for (
                    ambulance_id,
                    mission,
                ) in list(
                    active_missions.items()
                ):

                    # ----------------------------------------
                    # 차량이 SUMO에서 사라진 경우
                    # ----------------------------------------

                    if (
                        ambulance_id
                        not in ambulance_ids
                    ):

                        reward = (
                            calculate_reward(
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

                        loss = (
                            agent.train_step()
                        )

                        if loss is not None:

                            episode_losses.append(
                                loss
                            )

                        episode_reward += (
                            reward
                        )

                        failed_missions += 1

                        finished_ids.append(
                            ambulance_id
                        )

                        continue

                    # ----------------------------------------
                    # 경과시간
                    # ----------------------------------------

                    mission[
                        "elapsed_time"
                    ] += step_length

                    # ----------------------------------------
                    # 구급차 현재 위치
                    # ----------------------------------------

                    try:

                        vehicle_pos = (
                            traci.vehicle.getPosition(
                                ambulance_id
                            )
                        )

                        target_pos = (
                            mission[
                                "target_pos"
                            ]
                        )

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

                        distance = float(
                            "inf"
                        )

                    # ----------------------------------------
                    # 도착 여부
                    # ----------------------------------------

                    arrived = (
                        distance
                        <= ARRIVAL_DISTANCE
                    )

                    # ----------------------------------------
                    # Timeout 여부
                    # ----------------------------------------

                    timeout = (
                        mission[
                            "elapsed_time"
                        ]
                        >= MISSION_TIMEOUT_SECONDS
                    )

                    # ----------------------------------------
                    # 도착 또는 Timeout
                    # ----------------------------------------

                    if arrived or timeout:

                        elapsed_time = (
                            mission[
                                "elapsed_time"
                            ]
                        )

                        reward = (
                            calculate_reward(
                                travel_time=elapsed_time,
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

                        loss = (
                            agent.train_step()
                        )

                        if loss is not None:

                            episode_losses.append(
                                loss
                            )

                        episode_reward += (
                            reward
                        )

                        # ------------------------------------
                        # 성공
                        # ------------------------------------

                        if arrived:

                            completed_missions += 1

                            # 평균 이송시간
                            episode_travel_times.append(
                                elapsed_time
                            )

                            # --------------------------------
                            # 골든타임 성공 여부
                            # --------------------------------

                            if (
                                elapsed_time
                                <= mission[
                                    "golden_limit"
                                ]
                            ):

                                golden_success_count += 1

                        # ------------------------------------
                        # 실패
                        # ------------------------------------

                        else:

                            failed_missions += 1

                        finished_ids.append(
                            ambulance_id
                        )

                # ============================================
                # 완료된 Mission 제거
                # ============================================

                for ambulance_id in finished_ids:

                    active_missions.pop(
                        ambulance_id,
                        None,
                    )

                # ============================================
                # 새로운 환자 발생
                # ============================================

                patient = (
                    patient_schedule.get(
                        step
                    )
                )

                if patient is None:
                    continue

                # ============================================
                # 최대 구급차 수 제한
                # ============================================

                if (
                    len(ambulance_ids)
                    >= MAX_AMBULANCES
                ):
                    continue

                # ============================================
                # 환자 정보
                # ============================================

                patient_edge = (
                    patient["edge_id"]
                )

                severity = int(
                    patient["severity"]
                )

                # ============================================
                # 환자 위치
                # ============================================

                try:

                    edge_obj = (
                        env.net.getEdge(
                            patient_edge
                        )
                    )

                    patient_pos = (
                        edge_obj
                        .getFromNode()
                        .getCoord()
                    )

                except Exception:

                    continue

                # ============================================
                # 병원별 예상 경로
                # ============================================

                hospital_routes = (
                    calculate_hospital_routes(
                        env.net,
                        patient_edge,
                        hospitals,
                    )
                )

                # ============================================
                # DQN State
                # ============================================

                state = (
                    build_state_vector(
                        patient_pos,
                        severity,
                        hospitals,
                        hospital_routes,
                    )
                )

                # ============================================
                # Valid Actions
                # ============================================

                valid_actions = (
                    _get_valid_actions(
                        hospitals,
                        severity,
                    )
                )

                # ============================================
                # 모든 병원이 가득 찬 경우
                # ============================================

                if not valid_actions:

                    valid_actions = (
                        _get_fallback_actions(
                            hospitals
                        )
                    )

                if not valid_actions:
                    continue

                # ============================================
                # 초기 Exploration
                # ============================================

                if (
                    episode <= 10
                    and random.random()
                    < 0.25
                ):

                    # 초기에는 일부를
                    # shortest-time 방식으로 선택
                    selected_action = min(
                        valid_actions,
                        key=lambda index:
                            hospital_routes[
                                index
                            ][
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

                selected_hospital = (
                    hospitals[
                        selected_action
                    ]
                )

                # ============================================
                # 병상 거부 처리
                # ============================================

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
                        if action
                        != selected_action
                    ]

                    if not alternative_actions:
                        break

                    selected_action = (
                        agent.select_action(
                            state,
                            alternative_actions,
                        )
                    )

                    selected_hospital = (
                        hospitals[
                            selected_action
                        ]
                    )

                # ============================================
                # 선택 당시 occupancy 기록
                # ============================================

                occupancy_at_selection = float(
                    selected_hospital.get(
                        "occupancy",
                        0.0,
                    )
                )

                # ============================================
                # 병원 수용 처리
                # ============================================

                selected_hospital[
                    "occupancy"
                ] = min(
                    1.0,
                    occupancy_at_selection
                    + 0.08,
                )

                # ============================================
                # 선택 병원 Route
                # ============================================

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

                # --------------------------------------------
                # Route가 없으면 다시 계산
                # --------------------------------------------

                if (
                    route is None
                    and hospital_edge
                    is not None
                ):

                    route = _find_route(
                        patient_edge,
                        hospital_edge,
                    )

                # ============================================
                # Route 실패
                # ============================================

                if (
                    hospital_edge is None
                    or route is None
                    or len(route.edges) == 0
                ):

                    reward = (
                        calculate_reward(
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

                    loss = (
                        agent.train_step()
                    )

                    if loss is not None:

                        episode_losses.append(
                            loss
                        )

                    episode_reward += (
                        reward
                    )

                    failed_missions += 1

                    continue

                # ============================================
                # 구급차 생성
                # ============================================

                ambulance_id = (
                    f"ambulance_"
                    f"{episode}_"
                    f"{step}"
                )

                route_id = (
                    f"route_"
                    f"{episode}_"
                    f"{step}"
                )

                # --------------------------------------------
                # 기존 route ID 제거
                # --------------------------------------------

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

                # ============================================
                # SUMO 구급차 등록
                # ============================================

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

                    reward = (
                        calculate_reward(
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

                    loss = (
                        agent.train_step()
                    )

                    if loss is not None:

                        episode_losses.append(
                            loss
                        )

                    episode_reward += (
                        reward
                    )

                    failed_missions += 1

                    continue

                # ============================================
                # 예상 이동시간
                # ============================================

                estimated_travel_time = float(
                    max(
                        0.0,
                        route.travelTime,
                    )
                )

                # ============================================
                # 목표 위치
                # ============================================

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

                # ============================================
                # Mission 등록
                # ============================================

                active_missions[
                    ambulance_id
                ] = {

                    "hospital":
                        selected_hospital,

                    "severity":
                        severity,

                    "elapsed_time":
                        0.0,

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

        # ====================================================
        # SUMO 오류
        # ====================================================

        except traci.TraCIException as exc:

            print(
                f"[Episode {episode}] "
                f"SUMO 오류: {exc}"
            )

        # ====================================================
        # Episode 종료 처리
        # ====================================================

        finally:

            # ------------------------------------------------
            # 아직 완료되지 않은 Mission
            # ------------------------------------------------

            for (
                ambulance_id,
                mission,
            ) in list(
                active_missions.items()
            ):

                reward = (
                    calculate_reward(
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

                loss = (
                    agent.train_step()
                )

                if loss is not None:

                    episode_losses.append(
                        loss
                    )

                episode_reward += (
                    reward
                )

                failed_missions += 1

            active_missions.clear()

            # ------------------------------------------------
            # SUMO 종료
            # ------------------------------------------------

            if traci.isLoaded():

                try:

                    traci.close()

                except Exception:

                    pass

        # ====================================================
        # Target Network 업데이트
        # ====================================================

        if episode % 5 == 0:

            agent.update_target_network()

        # ====================================================
        # 모델 저장
        # ====================================================

        if episode % 5 == 0:

            agent.save_model(
                MODEL_PATH
            )

            print(
                f"[Checkpoint] "
                f"{MODEL_PATH} 저장 완료"
            )

        # ====================================================
        # Episode 통계
        # ====================================================

        # ----------------------------------------------------
        # 평균 이송시간
        # ----------------------------------------------------

        if episode_travel_times:

            avg_travel_time = float(
                np.mean(
                    episode_travel_times
                )
            )

        else:

            avg_travel_time = 0.0

        # ----------------------------------------------------
        # 골든타임 성공률
        # ----------------------------------------------------

        if completed_missions > 0:

            golden_success_rate = (
                golden_success_count
                / completed_missions
            )

        else:

            golden_success_rate = 0.0

        # ----------------------------------------------------
        # 평균 Loss
        # ----------------------------------------------------

        if episode_losses:

            avg_loss = float(
                np.mean(
                    episode_losses
                )
            )

        else:

            avg_loss = 0.0

        # ====================================================
        # Episode 결과 출력
        # ====================================================

        print(
            f"[Episode {episode:03d}] "
            f"Scenario={scenario['id']} | "
            f"Reward={episode_reward:8.2f} | "
            f"Completed={completed_missions:3d} | "
            f"Failed={failed_missions:3d} | "
            f"Reject={rejection_total:3d} | "
            f"AvgTime={avg_travel_time:7.2f}s | "
            f"GoldenRate={golden_success_rate:6.2%} | "
            f"Loss={avg_loss:.5f} | "
            f"Epsilon={agent.epsilon:.4f}"
        )

    # ========================================================
    # 최종 모델 저장
    # ========================================================

    agent.save_model(
        MODEL_PATH
    )

    print()
    print(
        "=" * 70
    )
    print(
        "DQN 학습 완료"
    )
    print(
        f"최종 모델: {MODEL_PATH}"
    )
    print(
        "=" * 70
    )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    train_dqn_fast(
        num_episodes=50
    )