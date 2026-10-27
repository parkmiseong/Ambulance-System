# -*- coding: utf-8 -*-
"""동일 DQN 및 동일 시나리오/시드에서 대기열 정책만 비교한다."""
import argparse
import copy
import time
from experiment_log import append_record, new_run_id, LOG_PATH
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from run_scenarios import SCENARIOS, MODEL_PATH, run_single


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ambulances', type=int, default=10)
    parser.add_argument('--scenario', type=int, default=1)
    parser.add_argument('--aging-seconds', type=float, default=300)
    parser.add_argument('--model', default=MODEL_PATH)
    args = parser.parse_args()
    env = SumoMedicalEnvironment()
    hospitals = copy.deepcopy(env.get_hospitals())
    scenario = next((s for s in SCENARIOS if s['id'] == args.scenario), None)
    if scenario is None:
        parser.error('존재하지 않는 시나리오 번호입니다')
    agent = DQNAmbulanceAgent(3 + len(hospitals) * 4, len(hospitals))
    if not agent.load_model(args.model):
        parser.error(f'모델을 찾을 수 없습니다: {args.model}')
    agent.epsilon = 0.0
    agent.model.eval()
    seed = 30000 + scenario['id']
    run_id = new_run_id('queue_compare')
    started = time.monotonic()
    append_record(run_id, 'queue_comparison', 'run_start', scenario_id=scenario['id'],
                  scenario_name=scenario['name'], ambulance_limit=args.ambulances,
                  model_path=args.model, metrics={'seed': seed, 'aging_seconds': args.aging_seconds})
    print(f'[통합 로그] {LOG_PATH} | 실행 ID: {run_id}')
    for policy in ('fifo', 'severity', 'hybrid'):
        result = run_single(env, hospitals, scenario, seed, 'dqn', agent=agent,
                            ambulance_count=args.ambulances, queue_policy=policy,
                            aging_seconds=args.aging_seconds)
        append_record(run_id, 'queue_comparison', 'queue_policy_result',
                      scenario_id=scenario['id'], scenario_name=scenario['name'],
                      strategy=f'DQN_{policy}', status='completed',
                      ambulance_limit=args.ambulances, model_path=args.model,
                      duration_seconds=round(time.monotonic()-started, 2),
                      metrics={**result, 'seed': seed, 'aging_seconds': args.aging_seconds})
        print(f"[{policy}] 도착={result['arrived']}, 실패={result['failed']}, "
              f"골든타임={result['golden_rate']:.1f}%, 평균 대기={result['avg_wait']:.1f}초, "
              f"평균 보상={result['avg_reward']:.2f}")
    append_record(run_id, 'queue_comparison', 'run_end', status='completed',
                  ambulance_limit=args.ambulances, model_path=args.model,
                  duration_seconds=round(time.monotonic()-started, 2))


if __name__ == '__main__':
    main()
