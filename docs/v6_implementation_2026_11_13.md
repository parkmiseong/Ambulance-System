# v6 구현 변경과 실험 실행 안내

작성일: 2026-11-13

이 문서는 합의한 DQN 개선안을 현재 코드에 적용한 내용이다. 새 학습은 v6로 저장하며, 기존 v3·v4·v5 모델은 해당 관측·네트워크로 계속 평가한다. v5의 18회·54회 결과는 [당시 상세 설명](current_code_details_2026_11_11.md)에 남긴다.

## 1. 적용 범위

| 항목 | 현재 구현 |
|---|---|
| 실험/관측 버전 | 실험 6, 관측 스키마 5 |
| 네트워크 | hospital_shared_v6 |
| 9개 병원 입력 | 공통 16 + 병원별 11 = 115차원 |
| 타깃 동기화 기본 | 실제 gradient update 200회마다 hard sync |
| 비교용 동기화 | episodes: 기본 누적 10회마다 / soft: 업데이트마다 tau=0.005 |
| gamma·보상 | gamma=0.95, 보상 v4와 선택 구간 귀속 유지 |
| 보상 스케일 | 기본 /100 유지 |
| 추가 특징 | 골든타임 여유·후보별 예상 보상, 각각 기본 0인 대조군 |
| 비교 정책 | 기존 정책 + greedy_reward |
| 검증 기본 시드 | 30001,30002,30003 |
| 평가 기본 시드 | 32001,32002,32003 |
| 모델 선택 | 항상 best. 지정한 last 경로도 같은 실험의 best로 연결 |
| 평가 통계 | 보상 분해, 환자 대응 차이, 에피소드 시드 단위 95% bootstrap 구간 |

보상 소급 배정, 예상 보상의 즉시 지급, 잔차 Q 구조, 입력 삭제, 마스크된 병원 제외는 이번 기본 설정에 적용하지 않았다. gamma를 크게 낮추거나 스케일을 /10으로 바꾸는 실험도 기본값으로 채택하지 않았다.

## 2. 학습 설정과 재개

[LearningConfig](../CODE/experiment_config.py)에 gamma, 학습률, 배치 크기, replay 크기, 보상 스케일, 타깃 방식·주기·tau, epsilon 시작·최저·감소 방식, gradient 제한을 모은다. 이 설정은 manifest와 체크포인트에 저장하고 signature에 포함한다.

같은 관측 크기라도 gamma·타깃 주기·특징 스위치가 다르면 서로 다른 실험이다. 재개 시 설정 일치를 확인하고, 새로운 v6 실험을 과거 v5 체크포인트에 이어 붙이지 않는다.

누적 통계는 다음과 같이 구분한다.

| 필드 | 의미 |
|---|---|
| total_decisions | 실제 학습 병원 선택 수 |
| exploratory_decisions | 그중 탐색으로 고른 수 |
| total_transitions | 저장한 경험의 누적 수. replay가 가득 차도 계속 증가 |
| training_updates | 실제 optimizer.step 횟수 |
| target_updates | 학습 중 타깃 갱신 횟수. 최초 복사는 제외 |
| episode_updates | 이번 에피소드의 실제 업데이트 수 |
| episode_decisions | 이번 에피소드의 병원 선택 수 |
| exploration_rate | 이번 에피소드의 탐색 선택 비율 |
| counters_complete | 누적 통계가 학습 시작부터 보존됐는지 여부 |

카운터가 없는 과거 모델은 counters_complete=false로 표시한다. 이 경우 0인 내부 카운터를 과거 실제 학습량 0으로 해석하지 않는다. v6 재개는 온전한 카운터를 요구한다.

학습 중 검증은 가중치와 학습 카운터를 변경하지 않는다. 타깃 갱신은 updates·episodes·soft 중 한 방식만 사용하며, 재개 후에도 누적 경계에서 이어진다.

## 3. 탐색과 업데이트 예산

기본 epsilon 일정은 기존과 같이 실제 업데이트마다 0.9995를 곱한다. 배치가 부족해 업데이트하지 않으면 이 일정도 진행하지 않는다.

선택 횟수 기준 대조 실험은 다음 옵션을 사용한다.

```text
--epsilon-schedule decisions_linear --epsilon-decay-decisions 5000
```

epsilon은 실제 학습 선택 횟수에 따라 1.0에서 0.05까지 선형 감소한다. 평가 선택은 이 카운터에 포함하지 않는다. 기존 일정과 비교할 때 다른 설정은 유지한다.

`--max-updates 5000`은 모델 전체의 누적 업데이트 상한이다. 상한 이후 추가 업데이트는 중단하고 현재 에피소드의 환자 처리는 끝낸 뒤 저장·검증하고 종료한다. 따라서 에피소드 수·선택 수는 설정에 따라 달라질 수 있다. 재개 시에는 현재 누적 횟수보다 큰 상한을 지정한다.

episodes와 max-updates는 실행 예산으로 기록하며 signature에는 포함하지 않는다. epsilon 감소 일정 자체의 값은 학습 설정이므로 signature에 포함한다.

## 4. 추가 관측 특징

기존 공통 특징 16개와 병원 특징 9개를 유지하고 병원별로 두 칸을 추가한다. 병원 h의 블록 시작은 16+11h다.

| 블록 위치 | 특징 | 계산 |
|---:|---|---|
| 9 | 골든타임 여유 | clip((L−현재 대기−예상 이동시간)/L, −3, 1) |
| 10 | 후보별 예상 보상 | v4(현재 대기+예상 이동시간, KTAS, 해당 병원, 선택 전 점유율)/100 |

`--golden-margin-feature`, `--candidate-reward-feature`로 각각 켠다. 끄면 해당 칸을 0으로 고정한다. 네트워크와 입력 크기가 같아 특징 하나의 효과를 비교할 수 있다. 도달 불가 병원의 추가 특징은 0이다.

예상 보상은 배정 시 알 수 있는 정보만 사용한다. 실제 미래 도착시간·보상은 관측에 넣지 않는다. 예상 보상을 학습 보상에 더 지급하지도 않는다. 병원별 학습 평가는 공통 16 + 해당 병원 11 + 전체 병원 평균·최소·최대 33 = 60차원 입력을 사용한다.

## 5. 실험 순서와 학습 명령

프로젝트 루트의 PowerShell에서 실행한다. 기존 모델과 구분할 새 저장 폴더를 사용한다. 학습 횟수 54는 기존 실험과 같은 에피소드 예산을 잡은 예시이며, 성능에 충분한 최소 횟수라는 의미는 아니다.

### 5.1 v6 대조군: 10에피소드 동기화

```powershell
.venv/Scripts/python.exe -u CODE/train_agent.py --episodes 54 --train-scenarios 6 --eval-every 9 --ambulances 10 --virtual-count 2 --virtual-strategy far --placement-seed 42 --seed 42 --curriculum 0.1,0.3 --cycle-curriculum --validation-scenarios 6 --validation-demand-levels 0.1,0.3 --validation-seeds 30001,30002,30003 --target-sync-mode episodes --target-sync-episodes 10 --quiet --output-root results/v6_s6_episode_s42
```

### 5.2 첫 변경: 업데이트 200회 동기화

```powershell
.venv/Scripts/python.exe -u CODE/train_agent.py --episodes 54 --train-scenarios 6 --eval-every 9 --ambulances 10 --virtual-count 2 --virtual-strategy far --placement-seed 42 --seed 42 --curriculum 0.1,0.3 --cycle-curriculum --validation-scenarios 6 --validation-demand-levels 0.1,0.3 --validation-seeds 30001,30002,30003 --target-sync-mode updates --target-sync-interval 200 --quiet --output-root results/v6_s6_updates_s42
```

두 명령은 모두 v6의 115차원·추가 특징 0 대조군을 사용한다. 과거 v5의 97차원 모델과 직접 비교해 타깃 주기 하나의 효과라고 해석하지 않는다.

이후 5.2 명령에서 하나씩만 바꾸고 새 폴더에 저장한다.

| 실험 | 추가할 옵션 | 저장 폴더 예시 |
|---|---|---|
| 골든타임 여유 | --golden-margin-feature | results/v6_s6_margin_s42 |
| 후보 예상 보상 | --candidate-reward-feature | results/v6_s6_reward_s42 |
| 선택 기준 epsilon | --epsilon-schedule decisions_linear --epsilon-decay-decisions 5000 | results/v6_s6_epsilon_s42 |
| 학습량 확대 | --episodes 1000 --max-updates 5000 | 별도 실험 폴더 |

첫 학습에는 resume을 넣지 않는다. 같은 실험을 이어 학습할 때만 동일 설정에 `--resume`을 추가하고 episodes를 추가 횟수로 지정한다. 검증 시드 3개·부하 2개이므로 검증 시점마다 SUMO를 6회 실행한다.

유망한 설정은 학습 seed를 42·43·44로 바꿔 독립 반복한다. placement-seed는 42로 고정하고 다른 설정·검증·평가 시드는 동일하게 둔다. 이 세 반복만으로 일반화 우위를 확정하지 않는다.

## 6. best 모델과 독립 평가

```powershell
.venv/Scripts/python.exe -u CODE/run_scenarios.py --output-root results/v6_s6_updates_s42 --scenario 6 --seeds 32001,32002,32003 --demand-levels 0.1,0.3 --strategies shortest_time,heuristic,greedy_reward,dqn --quiet
```

이 명령은 지정 폴더의 best를 고르고 실제 학습 회차를 출력한다. best가 없으면 다른 last로 대신 평가하지 않는다. 학습 시나리오가 하나면 scenario 생략 시 그 번호를 사용하고, 수요 생략 시 manifest의 검증 수요를 사용한다.

최종 평가에서는 학습·검증에서 사용하지 않은 시드를 사용한다. 이미 관찰한 평가 결과로 설정을 계속 고르면 그 패널은 추가 검증용으로 취급하고, 최종 보고에는 별도 시드를 남긴다. 최종 표본은 시드 수를 늘려 부하별 수백 명을 확보하고 실제 환자 수를 함께 보고한다.

중복 시드나 학습·검증과 겹친 시드는 기본 평가에서 거부한다. 점검 목적으로 재사용하려면 `--diagnostic-seeds`를 명시하며, 로그를 diagnostic으로 표시한다.

greedy_reward는 현재 유효 병원 중 예상 환자 보상이 가장 큰 병원을 고른다. 예상 이동시간과 현재 대기·점유율을 사용하며 실제 미래 결과를 아는 정책이 아니다.

## 7. 보상 분해와 대응 비교

mission 로그에 edge_id, 선택 점유율, 적합도, 8개 reward_components를 저장한다. 에피소드·정책 요약에는 항목별 합계와 환자당 평균을 남긴다.

같은 실행에 shortest_time이 있으면 다른 정책과 환자 ID·발생 시각·KTAS·도로가 일치하는지 확인하고 후보 정책−기준 정책의 차이를 기록한다. `--reference-strategy heuristic`으로 기준을 바꿀 수 있다.

평균 보상, 도착률, 골든타임률, 대기시간의 95% percentile bootstrap 구간은 에피소드 시드를 묶음으로 재표집한다. paired_comparison에는 보상 항목별 차이도 포함한다. 기본 재표집 횟수는 2000이다.

- 같은 에피소드의 환자들을 독립 표본으로 취급하지 않는다.
- 시드가 하나뿐이거나 분모가 0이면 구간을 보고하지 않는다.
- 같은 시드가 반복 집계돼도 요약 평균은 구할 수 있으나 구간은 생략한다.
- 이 구간은 평가 시드 변동을 나타내며, 독립 학습 시드에 따른 불확실성을 포함하지 않는다.
- 3개 시드의 구간은 탐색적 결과다. 작은 표본의 안정성을 과장하지 않는다.

## 8. 서로 다른 best 모델 비교

각 모델을 동일 평가 조건으로 실행한 뒤 로그의 run_id를 사용한다.

```powershell
.venv/Scripts/python.exe CODE/compare_evaluations.py --logs results/v6_s6_episode_s42/experiment_history.csv results/v6_s6_updates_s42/experiment_history.csv --run-ids BASELINE_RUN_ID,CANDIDATE_RUN_ID --reference-strategy dqn --output results/reports/v6_target_comparison.json
```

BASELINE_RUN_ID와 CANDIDATE_RUN_ID를 실제 평가 ID로 바꾼다. 한 파일에 모두 있으면 `--log`를 사용한다. 도구는 완료된 best·test 평가만 비교하며 학습 회차를 보존한다. last 평가를 best 결과로 재해석하지 않는다. 채점 버전과 환자 패널이 달라지면 비교를 중단한다.

18회 학습 구간의 best는 18회, 기존 54회 학습 구간의 best는 27회다. 두 모델도 새 best 평가 로그를 만든 뒤 같은 방법으로 비교한다.

## 9. 검증 범위

전체 회귀 테스트에서 타깃 갱신 경계, warm-up, soft update, 예산 상한, 체크포인트 재개, 입력 대조군, 미래 결과 비사용, 병원 순서 동등성, best 선택, 평가 시드 분리와 대응 통계를 확인한다.

실제 SUMO 점검은 `results/verification/v6_20261113_smoke/`에서 수행한다. 환자 발생 60초·이송 제한 180초·후보 도로 30개·배치 크기 2·타깃 주기 2의 짧은 기능 검사다. 두 학습 회차에서 선택 8→14개, 업데이트 7→13회, 타깃 갱신 3→6회, epsilon 0.620→0.335가 이어졌다.

이 제한 설정에서는 모든 학습 환자가 이송 시간 초과였다. 학습·재개·실패 보상 처리를 확인하는 결과이며 성능 향상이나 임상적 효과의 근거가 아니다. 실제 성능 비교에는 위 표준 설정과 독립 평가를 사용한다.

### 검증 완료 기록

- 회귀 테스트 104개 통과.
- 기존 v3·v4·v5 best 파일 4개의 로드와 가중치 무변경 확인.
- 실제 SUMO 학습·재개 후 누적 14개 선택, 13회 업데이트, 6회 타깃 갱신 확인.
- 총 학습 2회와 별개로 best인 1회 모델이 평가에 사용됐음을 확인.
- 독립 평가 시드 2개·정책 4개, 32건의 평가 환자 기록과 학습 14건을 합한 46건의 보상 항목 합계 일치.
- 3개 정책의 대응 비교와 항목별 차이 합계 일치 확인.

[기능 검증 보고서](../results/verification/v6_20261113_smoke/functional_verification.json)에 원본 경로·선택 모델 회차·통계·검증 범위를 저장했다. 이 보고서는 짧은 기능 점검이며 정식 정책 성능 비교가 아니다.
