# 학습·평가 조건 복원과 결과 재현성 (2026-11-16)

이번 변경은 웹·CLI의 실행 조건을 저장·복원하고, 같은 환경에서 모델 성능을 비교하기 위한 보완입니다. 관측 스키마·보상 함수·DQN 학습 알고리즘의 변경 실험과는 별도로 관리합니다. 기존 모델과 결과 파일을 유지하며, 배경 교통 규칙을 바꾸는 학습은 새 실험으로 시작합니다.

## 적용 내용

| 항목 | 적용 내용 |
|---|---|
| 웹 모델 평가 조건 | 저장된 수요·웹 시나리오·고정 환자 수·초기 병원 점유율을 CLI 기본 평가에 복원합니다. |
| 배경 교통 | 새 CLI 학습과 웹에 원본 1시간 승용차 패턴의 주기적 반복을 적용합니다. 기존 모델은 원래 규칙을 복원합니다. |
| 동일 조건 비교 | 실제 실행 환경을 기록하고 구급차 수·병원·교통·수요·운영 설정 등이 같은지 검사합니다. |
| best 선정 | 수요 프로필과 관계없이 실제 가중치 업데이트가 0회이면 새 best로 선정하지 않습니다. |
| 평가 모델 보관 | 평가 전에 best와 manifest를 복사하고, 사본의 SHA256을 기록하여 사용한 모델을 보존합니다. |

## 저장된 조건과 명시적 변경

`run_scenarios.py`와 `compare_queue_policies.py`는 기본적으로 `--scenario-profile saved`를 사용합니다.

- 웹 모델에서 `--scenario`를 생략하면 저장된 웹 시나리오와 수요를 복원합니다. 고정 총 환자 수와 초기 병원 점유율도 함께 복원합니다.
- 웹의 기본 세 단계 수요를 저장한 모델에서 `--scenario N` 또는 `--scenario 0`을 지정하면 저장된 세 단계 발생률과 해당 교통 단계를 사용합니다. 직접 입력한 시간당 발생률·고정 총 환자 수는 시나리오를 바꾸어도 유지합니다.
- CLI 학습 모델은 저장된 CLI 수요 설정을 사용합니다. `run_scenarios.py --scenario 0`은 전체 9개 시나리오 평가입니다.
- `--scenario-profile standard`를 명시하면 수요·교통을 기존 CLI 시나리오 정의로 바꿉니다. 초기 점유율은 저장된 웹 값이 있으면 유지하고, 없으면 기본 20~60%를 사용합니다. 과거 웹 모델처럼 `config.demand_rates_per_hour`가 없는 모델은 초당 확률 0.2~0.4의 고밀도 수요가 될 수 있으므로 실제 적용 프로필을 확인해 주세요.
- `--emergency-yield` / `--no-emergency-yield`, `--repeat-background-traffic` / `--no-repeat-background-traffic`는 해당 평가 조건을 명시적으로 변경합니다. 기본값은 모델의 저장된 설정입니다.
- `--demand-levels`는 시간당 기대 수요에 적용하는 배율입니다. 고정 총 환자 수를 복원한 평가에서는 `--demand-levels 1`만 사용합니다.

조건을 바꾼 평가는 원래 학습 조건을 재현한 평가와 구분해 기록합니다. 모델 가중치는 평가 중 업데이트하지 않습니다. 웹 화면에서 수행하는 시뮬레이션은 화면의 선택값이 우선하므로, CLI의 저장 조건 복원과 같은 의미로 해석하지 않도록 해 주세요.

## 새 학습과 기존 학습 재개

새 CLI 학습은 긴급차 양보와 배경 교통 반복이 기본 ON입니다. 반복은 시간대별 실측 교통량이 아니라 원본 1시간 승용차 패턴을 3,600초 간격으로 재사용하는 가정입니다. 환자 발생 종료 후 남은 이송을 마무리하는 구간에도 적용합니다. 반복 여부와 규칙은 실험 signature에 포함합니다.

같은 실험을 이어갈 때는 원래 설정·출력 폴더를 유지하고 `--resume`을 추가합니다. `--episodes`는 추가 학습 회차입니다.

| 기존 CLI 실험의 실제 조건 | 재개 시 필요한 설정 |
|---|---|
| 양보 OFF, 배경 교통 반복 OFF | `--no-emergency-yield --no-repeat-background-traffic --resume` |
| 양보 ON, 배경 교통 반복 OFF | `--emergency-yield --no-repeat-background-traffic --resume` |
| 양보 ON, 배경 교통 반복 ON | 원래 설정을 유지하고 `--resume`을 추가합니다. |

표의 옵션은 원래 수요·기간·병원·시드·학습 옵션에 추가하는 설정입니다. 기존 OFF 체크포인트를 ON 환경에서 바로 재개하지 않습니다. 새 조건을 학습하려면 다른 출력 폴더를 사용해 주세요. 실행 중인 프로세스에는 코드 수정이 소급 적용되지 않습니다.

모든 수요 프로필에서 실제 업데이트가 0회이면 새로운 best를 만들지 않습니다. 짧은 학습에서는 replay 배치가 차지 않아 업데이트가 없을 수 있으므로 로그의 `updates`를 확인해 주세요. 검증을 끈 `--eval-every 0` 실행도 best를 선정하지 않습니다. 기존에 저장된 best는 자동으로 삭제하지 않습니다.

시간당 기대 환자 10/15/20명·전체 9개 시나리오의 새 학습 명령은 [README의 학습 명령](../README.md#웹cli-차량-이동-규칙-통일-2026-11-16)을 사용해 주세요. 반복 도입 전 결과와 구분하여 `results/all_scenarios_normal15_yield_repeat_v6`에 저장합니다.

## 평가에 사용한 모델과 환경 기록

평가는 항상 best를 사용하며, `_last.pth`를 지정하면 같은 실험의 `_best.pth`로 연결합니다. 평가 시작 전에 다음 사본을 만들고 실제로 그 사본을 로드합니다.

```text
<실험 폴더>/evaluations/<평가 실행 ID>/
  models/<원래 모델명>_best.pth
  experiment.json
  snapshot.json
  evaluation.json
```

CLI 학습 모델의 실험 폴더는 `results/<실험명>/`이며, 웹 모델은 해당 `results/web_dashboard/<웹 실행 ID>/`입니다. 원본 best가 추가 학습으로 갱신되어도 평가 사본은 유지합니다.

웹 화면에서 DQN 시뮬레이션을 실행할 때도 해당 웹 실행 폴더의 `evaluation_snapshot/`에 best·manifest·해시를 보관하고 사본을 로드합니다. `run.json`과 각 `episodeN.json`에 `model_snapshot`과 실제 `evaluation_environment`를 저장합니다. 웹의 다른 정책도 실제 환경은 기록하며, 새 모델을 학습하는 실행에는 평가용 사본이 필요하지 않습니다.

`snapshot.json`과 평가 시작 기록의 `model_snapshot`에 사본 경로·SHA256·원본 경로를 남깁니다. 실제 환경은 `evaluation.json`과 로그의 `evaluation_environment`에 설정·fingerprint를 함께 기록합니다. 비교 대상인 모델 가중치·학습 시드·학습 하이퍼파라미터는 환경의 동일 여부와 분리해서 관리합니다.

환경 검사에는 환자 수요·발생 기간·초기 점유율, 구급차 수·대기열 정책, 대기·임무 제한, 병상·치료 설정, 병원 구성·접근 도로·환자 발생 도로, 보상 채점·수용 방식, 양보·배경 교통 반복, 도로·교통 입력 파일과 SUMO 실행 환경 등이 포함됩니다. 따라서 같은 환자 시드를 사용하더라도 구급차 1대와 10대의 결과를 동일 조건의 모델 성능 차이로 비교하지 않습니다.

서로 다른 모델의 학습률·학습 회차가 다르더라도 평가 환경을 같게 맞추면 비교할 수 있습니다. 환경이 다른 실험은 목적과 변경 항목을 따로 명시해 해석해 주세요.

## 교수님께 전달할 결과 준비

1. 비교할 각 학습 모델의 best를 같은 시나리오·수요·교통·병원·구급차 수에서 평가합니다. 학습·검증과 분리된 동일 평가 시드를 사용합니다.
2. `shortest_time`, `heuristic`, `dqn`을 같은 조건에서 실행하고, 필요하면 `greedy_reward`도 포함합니다. 총 보상과 환자당 보상, 도착률·골든타임·대기 시간, 보상 항목별 차이를 함께 확인합니다.
3. 각 평가의 `run_id`를 기록하고 `compare_evaluations.py`로 환자·시드가 대응되는 결과를 비교합니다. 환경 조건이 다르면 먼저 같은 조건으로 다시 평가합니다.
4. 평가별 모델 사본·manifest·원본 로그·비교 JSON을 함께 보관합니다. 보고서에는 모델의 실제 학습 회차, 평가 환자 수·시드 수, 적용 환경과 시드 단위 신뢰구간을 적습니다.

예를 들어 두 실험의 평가가 완료되면 다음과 같이 비교합니다. `RUN_A`, `RUN_B`는 각 로그에 저장된 실제 평가 `run_id`로 바꾸어 주세요.

```powershell
.venv/Scripts/python.exe -B CODE/compare_evaluations.py `
  --logs results/experiment_A/experiment_history.csv results/experiment_B/experiment_history.csv `
  --run-ids "RUN_A,RUN_B" --reference-strategy dqn `
  --output results/comparison_A_B.json
```

평가 환경 fingerprint가 없는 과거 로그는 기본 비교에서 거부합니다. 과거 기록을 확인할 목적으로 비교해야 하면 위 명령에 `--allow-legacy-environment`를 추가할 수 있지만, 결과는 `environment_verified=False`로 표시됩니다. 이 옵션은 환경이 같음을 증명하거나 과거 로그의 누락 정보를 복원하지 않습니다. 동일 조건을 입증해야 하는 최종 보고에는 새 평가 기록을 사용하는 편이 적절합니다.

## 변경 사항 검증

회귀 테스트는 다음 명령으로 실행합니다.

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests
```

실제 SUMO를 연결하는 작은 도로망 검증은 다음 명령으로 실행합니다. 별도 검증 폴더에 테스트용 모델을 만들고 CLI 평가·대기열 비교의 저장 조건 복원, 사본 로딩·해시, 환자 수와 도착, 원본 보존을 확인합니다. 이 테스트용 모델은 학습 성능을 평가하기 위한 모델이 아닙니다.

```powershell
.venv/Scripts/python.exe -B tests/verify_evaluation_pipeline.py
```

검증 결과는 `results/verification/evaluation_reproducibility_20261116/native_cli/proof.json`과 해당 실행의 하위 폴더에 보관합니다. 배경 교통의 1시간 경계 ON/OFF 대조 결과는 `results/verification/background_parity_20261116/native/proof.json`에 보관합니다.

2026-11-17 최종 확인에서 전체 회귀 테스트 259개가 통과했습니다. 실제 SUMO 평가에서는 `shortest_time`·`dqn` 및 대기열 세 정책이 각각 저장된 고정 환자 1명을 이송했고, 모델·manifest 사본의 해시 일치와 원본 보존을 확인했습니다. 이는 실행 경로 검증이며 학습 모델의 성능 비교 결과는 아닙니다. 검증 요약은 `results/verification/evaluation_reproducibility_20261116/verification_summary.json`에 저장했습니다.
