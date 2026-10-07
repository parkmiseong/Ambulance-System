import contextlib
import copy
import os
import random
import sys
import sumolib
import traci

from baseline import HospitalRouters, build_state_vector
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import GOLDEN_TIME_LIMITS, calculate_reward, normalize_ktas

## 상수 ##
MODEL_PATH = "dqn_ambulance_model_ktas_v1.pth"
NUM_EPISODES = 100
MAX_STEPS = 500
HOSPITAL_FULL_THRESHOLD = 0.85
MAX_ESTIMATED_TRAVEL_TIME = 600.0
BATCH_SIZE = 32
TARGET_UPDATE_EVERY = 10

SCENARIOS = [
    {"id": 1, "prob": 0.20, "scale": 0.3},
    {"id": 2, "prob": 0.20, "scale": 0.5},
    {"id": 3, "prob": 0.20, "scale": 0.8},
    {"id": 4, "prob": 0.30, "scale": 0.3},
    {"id": 5, "prob": 0.30, "scale": 0.5},
    {"id": 6, "prob": 0.30, "scale": 0.8},
    {"id": 7, "prob": 0.40, "scale": 0.3},
    {"id": 8, "prob": 0.40, "scale": 0.5},
    {"id": 9, "prob": 0.40, "scale": 0.8},
]

@contextlib.contextmanager
def suppress_sumo_stdout():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout, sys.stderr = devnull, devnull
            yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def start_sumo(env, scenario, seed):
    binary = sumolib.checkBinary("sumo")
    cmd = [
        binary, "-c", str(env.cfg_file),
        "--scale", str(scenario["scale"]),
        "--no-warnings", "true", "--no-step-log", "true",
        "--duration-log.disable", "true", "--seed", str(seed), "--start"
    ]
    with suppress_sumo_stdout():
        traci.start(cmd)
        try:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_custom")
            traci.vehicletype.setVehicleClass("ambulance_custom", "emergency")
            traci.vehicletype.setShapeClass("ambulance_custom", "emergency")
        except Exception:
            pass
    env.update_hospital_coordinates()

def get_reachable_edge_id(env, from_edge, x, y):
    if x is None or y is None:
        return None
    visited = set()
    radius = 100.0
    while radius <= 5000.0:
        try:
            nearby = env.net.getNeighboringEdges(x, y, radius)
        except Exception:
            nearby = []
        for edge, _ in sorted(nearby, key=lambda z: z[1]):
            edge_id = edge.getID()
            if not edge_id or edge_id.startswith(":") or edge_id in visited:
                continue
            visited.add(edge_id)
            try:
                with suppress_sumo_stdout():
                    route = traci.simulation.findRoute(
                        from_edge, edge_id, vType="ambulance_custom", depart=-1
                    )
                if route is not None and len(route.edges) > 0:
                    return edge_id
            except Exception:
                continue
        radius += 200.0
    return None

def calculate_hospital_routes(env, patient_edge, hospitals):
    result = copy.deepcopy(hospitals)
    for hospital in result:
        hospital["reachable"] = False
        hospital["hospital_edge"] = None
        hospital["route_length"] = None
        hospital["estimated_travel_time"] = MAX_ESTIMATED_TRAVEL_TIME
        try:
            edge = get_reachable_edge_id(
                env, patient_edge, hospital.get("sumo_x"), hospital.get("sumo_y")
            )
            if edge is None:
                continue
            with suppress_sumo_stdout():
                route = traci.simulation.findRoute(
                    patient_edge, edge, vType="ambulance_custom", depart=-1
                )
            if route is None or len(route.edges) == 0:
                continue
            length = float(getattr(route, "length", 0.0))
            travel = float(getattr(route, "travelTime", 0.0))
            if travel <= 0:
                travel = max(length / 13.9, 1.0)
            hospital["reachable"] = True
            hospital["hospital_edge"] = edge
            hospital["route_length"] = length
            hospital["estimated_travel_time"] = min(travel, MAX_ESTIMATED_TRAVEL_TIME)
        except Exception:
            continue
    return result

def generate_schedule(scenario, valid_edges, seed):
    rng = random.Random(seed)
    edges = [e for e in valid_edges if e and not str(e).startswith(":")]
    schedule = {}
    for step in range(1, MAX_STEPS + 1):
        if rng.random() < scenario["prob"]:
            schedule[step] = {
                "edge_id": rng.choice(edges),
                "ktas": rng.randint(1, 5),
            }
    return schedule


def valid_actions(hospitals):
    reachable = [
        i for i, h in enumerate(hospitals)
        if h.get("reachable", False)
        and float(h.get("occupancy", 1.0)) < HOSPITAL_FULL_THRESHOLD
    ]
    if reachable:
        return reachable
    return [i for i, h in enumerate(hospitals) if h.get("reachable", False)]


def train(num_episodes=NUM_EPISODES):
    env = SumoMedicalEnvironment()
    hospitals0 = copy.deepcopy(env.get_hospitals())
    action_dim = len(hospitals0)
    state_dim = 3 + action_dim * 4
    agent = DQNAmbulanceAgent(state_dim, action_dim)
    router = HospitalRouters()

    print(f"[KTAS DQN] state_dim={state_dim}, action_dim={action_dim}")
    print(f"[KTAS DQN] 모델={MODEL_PATH}")

    best_reward = -float("inf")

    for episode in range(1, num_episodes + 1):
        scenario = SCENARIOS[(episode - 1) % len(SCENARIOS)]
        seed = 10000 + episode
        hospitals = copy.deepcopy(hospitals0)
        for h in hospitals:
            h["occupancy"] = random.uniform(0.20, 0.60)

        schedule = generate_schedule(scenario, env.get_valid_edges(), seed)
        episode_reward = 0.0
        losses = []
        missions = 0

        try:
            start_sumo(env, scenario, seed)
            for step in range(1, MAX_STEPS + 1):
                if step not in schedule:
                    traci.simulationStep()
                    continue

                patient = schedule[step]
                ktas = normalize_ktas(patient["ktas"])
                patient_edge = patient["edge_id"]

                try:
                    edge_obj = env.net.getEdge(patient_edge)
                    patient_pos = edge_obj.getFromNode().getCoord()[:2]
                except Exception:
                    continue

                hospitals = calculate_hospital_routes(env, patient_edge, hospitals)
                actions = valid_actions(hospitals)
                if not actions:
                    continue

                state = build_state_vector(patient_pos, ktas, hospitals)
                action = agent.select_action(state, actions)
                selected = hospitals[action]
                occupancy = float(selected.get("occupancy", 0.0))
                travel = float(selected.get("estimated_travel_time", MAX_ESTIMATED_TRAVEL_TIME))
                failed = not selected.get("reachable", False)

                reward = calculate_reward(
                    travel_time=travel,
                    ktas=ktas,
                    hospital=selected,
                    golden_limit=GOLDEN_TIME_LIMITS[ktas],
                    occupancy_at_selection=occupancy,
                    route_failed=failed,
                )

                # 한 환자에 대한 병원 선택이 종료되므로 terminal transition
                next_state = np.zeros(state_dim, dtype=np.float32)
                agent.store_transition(state, action, reward, next_state, True)
                loss = agent.train_step(BATCH_SIZE)
                if loss is not None:
                    losses.append(loss)

                episode_reward += reward
                missions += 1
                selected["occupancy"] = min(1.0, occupancy + 0.08)
                traci.simulationStep()

            if episode % TARGET_UPDATE_EVERY == 0:
                agent.update_target_network()

        finally:
            try:
                traci.close()
            except Exception:
                pass

        avg_reward = episode_reward / missions if missions else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"Episode {episode:3d}/{num_episodes} | "
            f"Scenario {scenario['id']} | 환자 {missions:3d} | "
            f"Reward {episode_reward:8.2f} | Avg {avg_reward:7.2f} | "
            f"Loss {avg_loss:.5f} | Epsilon {agent.epsilon:.4f}"
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(MODEL_PATH)

    agent.save_model(MODEL_PATH)
    print(f"\n[완료] KTAS 모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    train(NUM_EPISODES)
