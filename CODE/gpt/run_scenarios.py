#!/usr/bin/env python3

import contextlib
import copy
import os
import random
import sys
import time

import numpy as np
import sumolib
import torch
import traci

from baselines import (
    HospitalRouters,
    build_state_vector,
)
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import calculate_reward


# ============================================================
# 설정
# ============================================================

MAX_STEPS = 600
MAX_AMBULANCES = 30

MISSION_TIMEOUT_SECONDS = 350.0
ARRIVAL_DISTANCE = 120.0

MODEL_PATH = "dqn_ambulance_model_v3.pth"

MAX_ESTIMATED_TRAVEL_TIME = 600.0

HOSPITAL_FULL_THRESHOLD = 0.85

GOLDEN_TIME_LIMITS = {
    4: 120.0,
    3: 180.0,
    2: 240.0,
    1: 300.0,
}


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

    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    saved_stdout_fd = os.dup(
        original_stdout_fd
    )
    saved_stderr_fd = os.dup(
        original_stderr_fd
    )

    with open(os.devnull, "w") as devnull:

        os.dup2(
            devnull.fileno(),
            original_stdout_fd
        )

        os.dup2(
            devnull.fileno(),
            original_stderr_fd
        )

        try:
            yield

        finally:

            os.dup2(
                saved_stdout_fd,
                original_stdout_fd
            )

            os.dup2(
                saved_stderr_fd,
                original_stderr_fd
            )

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


# ============================================================
# SUMO 실행
# ============================================================

def start_sumo(
    env,
    scenario,
    seed_val
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

        traci.vehicletype.copy(
            "DEFAULT_VEHTYPE",
            "ambulance_custom"
        )

        traci.vehicletype.setVehicleClass(
            "ambulance_custom",
            "emergency"
        )

        traci.vehicletype.setShapeClass(
            "ambulance_custom",
            "emergency"
        )

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
    seed_val
):

    random.seed(seed_val)
    np.random.seed(seed_val)

    schedule = {}

    usable_edges = [
        edge_id
        for edge_id in valid_edges
        if edge_id
        and not str(edge_id).startswith(":")
    ]

    if not usable_edges:
        return schedule

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        if random.random() < scenario["prob"]:

            schedule[step] = {
                "edge_id": random.choice(
                    usable_edges
                ),
                "severity": random.randint(
                    1,
                    4
                ),
            }

    return schedule


# ============================================================
# 접근 가능한 Edge 검색
# ============================================================

def get_reachable_edge_id(
    net,
    x,
    y,
    from_edge=None,
    vtype="ambulance_custom"
):

    if x is None or y is None:
        return None

    radius = 100
    visited = set()

    while radius <= 5000:

        nearby = net.getNeighboringEdges(
            x,
            y,
            radius
        )

        nearby = sorted(
            nearby,
            key=lambda item: item[1]
        )

        for edge, _distance in nearby:

            edge_id = edge.getID()

            if (
                edge_id.startswith(":")
                or edge_id in visited
            ):
                continue

            visited.add(edge_id)

            if from_edge is None:
                return edge_id

            if edge_id == from_edge:
                return edge_id

            try:

                with suppress_sumo_stdout():

                    route = traci.simulation.findRoute(
                        from_edge,
                        edge_id,
                        vType=vtype,
                    )

                if route and len(
                    route.edges
                ) > 0:

                    return edge_id

            except traci.exceptions.TraCIException:
                continue

            except Exception:
                continue

        radius += 200

    return None


# ============================================================
# Route 계산
# ============================================================

def find_route(
    from_edge,
    to_edge
):

    if not from_edge or not to_edge:
        return None

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
            )

        if route and len(
            route.edges
        ) > 0:

            return route

    except Exception:
        pass

    return None


# ============================================================
# 병원 Route 정보 계산
# ============================================================

def calculate_hospital_routes(
    env,
    patient_edge,
    hospitals
):

    """
    모든 병원에 대해

    - hospital edge
    - route length
    - estimated travel time

    을 계산합니다.

    이 정보가 바로 새로운 State의 병원별 정보가 됩니다.
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

        try:

            hospital_edge = (
                get_reachable_edge_id(
                    env.net,
                    hospital.get(
                        "sumo_x"
                    ),
                    hospital.get(
                        "sumo_y"
                    ),
                    from_edge=patient_edge,
                )
            )

            if hospital_edge is None:
                continue

            route = find_route(
                patient_edge,
                hospital_edge
            )

            if route is None:
                continue

            route_length = float(
                getattr(
                    route,
                    "length",
                    0.0
                )
            )

            travel_time = float(
                getattr(
                    route,
                    "travelTime",
                    0.0
                )
            )

            if travel_time <= 0:

                # SUMO route의 travelTime이
                # 제대로 제공되지 않는 경우
                # route length를 fallback으로 사용합니다.

                travel_time = max(
                    route_length / 13.9,
                    1.0
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
# 병원 좌표 최신화
# ============================================================

def prepare_hospitals(
    initial_hospitals,
    env
):

    hospitals = copy.deepcopy(
        initial_hospitals
    )

    env_hospitals = env.get_hospitals()

    coordinate_map = {
        h["id"]: (
            h.get("sumo_x"),
            h.get("sumo_y")
        )
        for h in env_hospitals
    }

    for hospital in hospitals:

        if hospital["id"] in coordinate_map:

            x, y = coordinate_map[
                hospital["id"]
            ]

            hospital[
                "sumo_x"
            ] = x

            hospital[
                "sumo_y"
            ] = y

    return hospitals


# ============================================================
# Action 선택
# ============================================================

def select_action(
    mode,
    router,
    patient_pos,
    severity,
    hospitals,
    state,
    agent,
    valid_actions
):

    if not valid_actions:
        valid_actions = list(
            range(len(hospitals))
        )

    if mode == "dqn":

        action = agent.select_action(
            state,
            valid_actions
        )

        return int(action)

    if mode == "shortest_time":

        return router.shortest_time_strategy(
            valid_actions
        )

    if mode == "shortest_distance":

        return router.shortest_distance_strategy(
            valid_actions
        )

    if mode == "bed":

        return router.bed_strategy(
            severity,
            valid_actions
        )

    if mode == "rule":

        return router.rule_based_strategy(
            severity,
            valid_actions
        )

    if mode == "heuristic":

        return router.heuristic_strategy(
            severity,
            valid_actions
        )

    raise ValueError(
        f"알 수 없는 평가 전략: {mode}"
    )


# ============================================================
# 단일 시나리오
# ============================================================

def run_single_scenario(
    scenario,
    initial_hospitals,
    seed_val,
    patient_schedule,
    mode="dqn",
    agent=None
):

    env = SumoMedicalEnvironment()

    random.seed(seed_val)
    np.random.seed(seed_val)

    total_reward = 0.0

    completed_missions = 0
    golden_success_count = 0

    total_rejections = 0
    total_transfer_time = 0.0

    hospital_assignment_counts = {}

    active_missions = {}

    spawned_count = 0

    try:

        # ----------------------------------------------------
        # SUMO 시작
        # ----------------------------------------------------

        step_length = start_sumo(
            env,
            scenario,
            seed_val
        )

        hospitals = prepare_hospitals(
            initial_hospitals,
            env
        )

        router = HospitalRouters(
            hospitals
        )

        hospital_assignment_counts = {
            h["id"]: 0
            for h in hospitals
        }

        # ----------------------------------------------------
        # Simulation
        # ----------------------------------------------------

        for step in range(
            1,
            MAX_STEPS + 1
        ):

            try:

                traci.simulationStep()

            except traci.exceptions.FatalTraCIError:

                break

            # ------------------------------------------------
            # 병원 점유율 자연 감소
            # ------------------------------------------------

            for hospital in hospitals:

                hospital["occupancy"] = max(
                    0.1,
                    hospital["occupancy"]
                    - 0.0005
                )

            ambulance_ids = set(
                traci.vehicle.getIDList()
            )

            # =================================================
            # 1. 기존 미션 처리
            # =================================================

            for amb_id in list(
                active_missions.keys()
            ):

                mission = active_missions[
                    amb_id
                ]

                # ---------------------------------------------
                # 차량이 사라진 경우
                # ---------------------------------------------

                if amb_id not in ambulance_ids:

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
                        golden_limit=GOLDEN_TIME_LIMITS[
                            mission["severity"]
                        ],
                        rejection_count=mission[
                            "rejections"
                        ],
                        occupancy_at_selection=mission[
                            "occupancy_at_selection"
                        ],
                        route_failed=True,
                    )

                    total_reward += reward

                    completed_missions += 1

                    total_transfer_time += (
                        mission[
                            "elapsed_time"
                        ]
                    )

                    del active_missions[
                        amb_id
                    ]

                    continue

                # ---------------------------------------------
                # 실제 SUMO 시간 증가
                # ---------------------------------------------

                mission[
                    "elapsed_time"
                ] += step_length

                try:

                    ambulance_pos = (
                        traci.vehicle.getPosition(
                            amb_id
                        )[:2]
                    )

                    target_pos = (
                        mission[
                            "target_pos"
                        ]
                    )

                    distance = float(
                        np.hypot(
                            ambulance_pos[0]
                            - target_pos[0],

                            ambulance_pos[1]
                            - target_pos[1]
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

                # ---------------------------------------------
                # 실제 미션 종료
                # ---------------------------------------------

                travel_time = mission[
                    "elapsed_time"
                ]

                severity = mission[
                    "severity"
                ]

                golden_limit = (
                    GOLDEN_TIME_LIMITS[
                        severity
                    ]
                )

                success = (
                    arrived
                    and travel_time
                    <= golden_limit
                )

                if success:

                    golden_success_count += 1

                reward = calculate_reward(
                    travel_time=travel_time,
                    severity=severity,
                    hospital=mission[
                        "hospital"
                    ],
                    golden_limit=golden_limit,
                    rejection_count=mission[
                        "rejections"
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

                try:

                    traci.vehicle.remove(
                        amb_id
                    )

                except Exception:
                    pass

                del active_missions[
                    amb_id
                ]

            # =================================================
            # 2. 신규 환자
            # =================================================

            if step not in patient_schedule:
                continue

            if (
                len(active_missions)
                >= MAX_AMBULANCES
            ):
                continue

            patient_info = (
                patient_schedule[step]
            )

            patient_edge = (
                patient_info["edge_id"]
            )

            severity = (
                patient_info["severity"]
            )

            # ---------------------------------------------
            # 환자 Edge 검증
            # ---------------------------------------------

            try:

                patient_edge_obj = (
                    env.net.getEdge(
                        patient_edge
                    )
                )

            except Exception:

                patient_edge_obj = None

            if patient_edge_obj is None:
                continue

            if str(
                patient_edge
            ).startswith(":"):

                continue

            # ---------------------------------------------
            # 환자 위치
            # ---------------------------------------------

            patient_pos = (
                patient_edge_obj
                .getFromNode()
                .getCoord()[:2]
            )

            # =================================================
            # 핵심
            # 모든 병원의 예상 이송시간 계산
            # =================================================

            hospitals = calculate_hospital_routes(
                env,
                patient_edge,
                hospitals
            )

            # ---------------------------------------------
            # 실제 route가 존재하는 병원만 후보
            # ---------------------------------------------

            valid_actions = [
                i
                for i, hospital in enumerate(
                    hospitals
                )
                if (
                    hospital.get(
                        "reachable",
                        False
                    )
                    and hospital.get(
                        "estimated_travel_time"
                    ) is not None
                    and hospital.get(
                        "occupancy",
                        1.0
                    ) < HOSPITAL_FULL_THRESHOLD
                )
            ]

            # ------------------------------------------------
            # 모든 병원이 포화/경로불가인 경우
            # 가장 낮은 점유율의 reachable 병원을 사용
            # ------------------------------------------------

            if not valid_actions:

                valid_actions = [
                    i
                    for i, hospital in enumerate(
                        hospitals
                    )
                    if hospital.get(
                        "reachable",
                        False
                    )
                ]

            if not valid_actions:
                continue

            # =================================================
            # 새로운 3 + 4N State
            # =================================================

            state = build_state_vector(
                patient_pos,
                severity,
                hospitals
            )

            # =================================================
            # DQN / Baseline 병원 선택
            # =================================================

            try:

                action = select_action(
                    mode=mode,
                    router=router,
                    patient_pos=patient_pos,
                    severity=severity,
                    hospitals=hospitals,
                    state=state,
                    agent=agent,
                    valid_actions=valid_actions,
                )

            except Exception:

                continue

            action = int(action)

            # ------------------------------------------------
            # 유효 action이 아닌 경우
            # ------------------------------------------------

            if action not in valid_actions:

                # DQN이 포화 병원을 선택했다면
                # 평가에서 다른 병원으로 몰래 바꾸지 않습니다.
                #
                # 해당 선택은 rejection으로 처리합니다.

                selected_hospital = hospitals[
                    action
                ]

                rejection_count = 1

                total_rejections += 1

            else:

                selected_hospital = hospitals[
                    action
                ]

                rejection_count = 0

            # =================================================
            # 선택 시점 정보 저장
            # =================================================

            occupancy_at_selection = float(
                selected_hospital.get(
                    "occupancy",
                    0.0
                )
            )

            # ------------------------------------------------
            # 병원 점유율 증가
            # ------------------------------------------------

            selected_hospital[
                "occupancy"
            ] = min(
                1.0,
                occupancy_at_selection
                + 0.08
            )

            hospital_assignment_counts[
                selected_hospital["id"]
            ] += 1

            # =================================================
            # Route 확인
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
                    golden_limit=GOLDEN_TIME_LIMITS[
                        severity
                    ],
                    rejection_count=rejection_count,
                    occupancy_at_selection=occupancy_at_selection,
                    route_failed=True,
                )

                total_reward += reward

                completed_missions += 1

                continue

            route = find_route(
                patient_edge,
                hospital_edge
            )

            if route is None:

                reward = calculate_reward(
                    travel_time=0.0,
                    severity=severity,
                    hospital=selected_hospital,
                    golden_limit=GOLDEN_TIME_LIMITS[
                        severity
                    ],
                    rejection_count=rejection_count,
                    occupancy_at_selection=occupancy_at_selection,
                    route_failed=True,
                )

                total_reward += reward

                completed_missions += 1

                continue

            # =================================================
            # Ambulance 생성
            # =================================================

            spawned_count += 1

            ambulance_id = (
                f"amb_eval_"
                f"{step}_"
                f"{spawned_count}"
            )

            route_id = (
                f"route_"
                f"{ambulance_id}"
            )

            try:

                traci.route.add(
                    route_id,
                    list(route.edges)
                )

                traci.vehicle.add(
                    vehID=ambulance_id,
                    routeID=route_id,
                    typeID="ambulance_custom",
                    depart="now",
                )

            except Exception:

                reward = calculate_reward(
                    travel_time=0.0,
                    severity=severity,
                    hospital=selected_hospital,
                    golden_limit=GOLDEN_TIME_LIMITS[
                        severity
                    ],
                    rejection_count=rejection_count,
                    occupancy_at_selection=occupancy_at_selection,
                    route_failed=True,
                )

                total_reward += reward

                completed_missions += 1

                continue

            # =================================================
            # Mission 저장
            # =================================================

            active_missions[
                ambulance_id
            ] = {

                "severity": severity,

                "hospital": selected_hospital,

                "elapsed_time": 0.0,

                "rejections": rejection_count,

                "occupancy_at_selection":
                    occupancy_at_selection,

                "state_vector": state,

                "action": action,

                "target_pos": (
                    selected_hospital[
                        "sumo_x"
                    ],
                    selected_hospital[
                        "sumo_y"
                    ],
                ),
            }

        # =====================================================
        # 시뮬레이션 종료
        # 미완료 미션은 실패 처리
        # =====================================================

        for (
            amb_id,
            mission
        ) in list(
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
                golden_limit=GOLDEN_TIME_LIMITS[
                    mission["severity"]
                ],
                rejection_count=mission[
                    "rejections"
                ],
                occupancy_at_selection=mission[
                    "occupancy_at_selection"
                ],
                route_failed=True,
            )

            total_reward += reward

            completed_missions += 1

            total_transfer_time += (
                mission[
                    "elapsed_time"
                ]
            )

            try:

                traci.vehicle.remove(
                    amb_id
                )

            except Exception:
                pass

    except Exception as exc:

        return {
            "reward": 0.0,
            "avg_reward": 0.0,
            "golden_success_rate": 0.0,
            "rejections": total_rejections,
            "avg_time": 0.0,
            "load_std": 0.0,
            "completed_missions":
                completed_missions,
            "zero_reason": str(exc),
        }

    finally:

        try:
            traci.close()
        except Exception:
            pass

    # =========================================================
    # 결과 계산
    # =========================================================

    golden_success_rate = (
        golden_success_count
        / completed_missions
        * 100.0
        if completed_missions > 0
        else 0.0
    )

    avg_time = (
        total_transfer_time
        / completed_missions
        if completed_missions > 0
        else 0.0
    )

    avg_reward = (
        total_reward
        / completed_missions
        if completed_missions > 0
        else 0.0
    )

    load_std = (
        float(
            np.std(
                list(
                    hospital_assignment_counts.values()
                )
            )
        )
        if hospital_assignment_counts
        else 0.0
    )

    return {
        "reward": total_reward,
        "avg_reward": avg_reward,
        "golden_success_rate":
            golden_success_rate,
        "rejections":
            total_rejections,
        "avg_time":
            avg_time,
        "load_std":
            load_std,
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
    result
):

    print(
        f"  · {label:<16}"
        f"| 보상 {result['reward']:8.1f} "
        f"| 평균보상 {result['avg_reward']:6.2f} "
        f"| 골든타임 "
        f"{result['golden_success_rate']:5.1f}% "
        f"| 거부 {result['rejections']:3d} "
        f"| 평균시간 "
        f"{result['avg_time']:6.1f}s "
        f"| 완료 "
        f"{result['completed_missions']:3d} "
        f"| 부하편차 "
        f"{result['load_std']:.2f}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 90
    )

    print(
        "시나리오 기반 응급실 병원 선택 평가"
    )

    print(
        "=" * 90
    )

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    temp_env = SumoMedicalEnvironment()

    valid_edges = (
        temp_env.get_valid_edges()
    )

    initial_hospitals = copy.deepcopy(
        temp_env.get_hospitals()
    )

    action_dim = len(
        initial_hospitals
    )

    # 새로운 State:
    # 3 + 4N

    state_dim = (
        3
        + action_dim * 4
    )

    print(
        f"[State] 차원 = {state_dim}"
    )

    print(
        f"[Action] 병원 수 = {action_dim}"
    )

    # ---------------------------------------------------------
    # DQN
    # ---------------------------------------------------------

    dqn_agent = DQNAmbulanceAgent(
        state_dim,
        action_dim
    )

    loaded = dqn_agent.load_model(
        MODEL_PATH
    )

    if not loaded:

        print(
            f"[경고] 모델을 불러오지 못했습니다: "
            f"{MODEL_PATH}"
        )

        sys.exit(1)

    # 평가에서는 탐험 제거

    dqn_agent.epsilon = 0.0

    dqn_agent.model.eval()

    print(
        "[DQN] 평가 모드 "
        "(epsilon = 0.0)"
    )

    # ---------------------------------------------------------
    # 전략
    # ---------------------------------------------------------

    strategies = [
        ("shortest_distance", "최단거리"),
        ("shortest_time", "최단시간"),
        ("bed", "병상기반"),
        ("rule", "치료역량기반"),
        ("heuristic", "종합휴리스틱"),
        ("dqn", "DQN"),
    ]

    print(
        "\n[평가 전략]"
    )

    for mode, label in strategies:

        print(
            f"  - {label}"
        )

    # ---------------------------------------------------------
    # Scenario 선택
    # ---------------------------------------------------------

    print(
        "\n[시나리오]"
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
                s
                for s in SCENARIOS
                if s["id"] == scenario_id
            ]

        except ValueError:

            selected_scenarios = []

    if not selected_scenarios:

        print(
            "[오류] 올바른 시나리오 번호가 아닙니다."
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # 전체 결과
    # ---------------------------------------------------------

    overall = {
        mode: {
            "reward": 0.0,
            "golden": 0.0,
            "rejections": 0,
            "avg_time": 0.0,
            "load_std": 0.0,
        }
        for mode, _label in strategies
    }

    # ---------------------------------------------------------
    # 시나리오 실행
    # ---------------------------------------------------------

    for scenario in selected_scenarios:

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"시나리오 {scenario['id']} "
            f"평가 시작"
        )

        print(
            "=" * 90
        )

        seed_val = (
            10000
            + scenario["id"]
            * 100
        )

        patient_schedule = (
            generate_fixed_patient_schedule(
                scenario,
                valid_edges,
                seed_val
            )
        )

        print(
            f"환자 발생 이벤트: "
            f"{len(patient_schedule)}"
        )

        for mode, label in strategies:

            print(
                f"\n[{label}]"
            )

            result = run_single_scenario(
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

            print_result(
                label,
                result
            )

            overall[
                mode
            ]["reward"] += result[
                "reward"
            ]

            overall[
                mode
            ]["golden"] += result[
                "golden_success_rate"
            ]

            overall[
                mode
            ]["rejections"] += result[
                "rejections"
            ]

            overall[
                mode
            ]["avg_time"] += result[
                "avg_time"
            ]

            overall[
                mode
            ]["load_std"] += result[
                "load_std"
            ]

    # =========================================================
    # 최종 결과
    # =========================================================

    count = len(
        selected_scenarios
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "최종 평가 결과"
    )

    print(
        "=" * 100
    )

    for mode, label in strategies:

        data = overall[
            mode
        ]

        print(
            f"{label:<18}"
            f"| 총보상 "
            f"{data['reward']:10.1f}"
            f"| 평균 골든타임 "
            f"{data['golden']/count:6.1f}%"
            f"| 총거부 "
            f"{data['rejections']:5d}"
            f"| 평균 이송시간 "
            f"{data['avg_time']/count:7.1f}s"
            f"| 평균 부하편차 "
            f"{data['load_std']/count:6.2f}"
        )

    print(
        "=" * 100
    )