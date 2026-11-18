# SUMO 1.28.0 에피소드 27 중단 분석 (2026-11-17)

SUMO 1.28.0에서 에피소드 25·26은 완료했지만 **27회차에서 SUMO 프로세스가 다시 접근 위반으로 종료되었습니다.** 현재는 원인 추적 중이며, 버전 교체로 전체 문제가 해결된 상태가 아닙니다. 같은 재개 명령을 반복하면 해결된다고 안내할 근거도 없습니다.

기존 1.27.1 실험의 **24회 체크포인트**와 새 1.28.0 실험의 **26회·342 업데이트 체크포인트**를 각각 보존했습니다. 중단된 27회차는 완료 회수에 포함하지 않습니다. 새 폴더에도 아직 best는 없습니다.

## 실제 학습에서 확인한 사실

| 항목 | 기록 |
|---|---|
| 실행 | `train_20261117_213050_c46a1a` |
| 중단 시각 | 2026-11-17 21:42:00 KST |
| 실제 엔진 | 프로젝트의 `.runtime/sumo-1.28.0/bin/sumo.exe` |
| 에피소드 / 시나리오 / 시드 | 27 / 9 / 10069 |
| 교통량 배율 | 0.8 |
| 설정 | 긴급차 양보 ON, 배경 교통 반복 ON |
| 실패 문맥의 마지막 loop 시각 | 1,128초 |
| 그때 완료 / 전체 발생 예정 환자 | 4 / 15명 |
| 관리 중 활성 구급차 | `amb_e_5`, `amb_e_6` |
| 네이티브 종료 코드 | `3221225477` = `0xc0000005` |
| Python 후속 오류 | `FatalTraCIError: Connection closed by SUMO.` |
| 마지막 저장 회차 / 업데이트 | 26 / 342 |

해당 학습은 상세 tick 추적을 켜지 않았으므로 `last_successful_sim_time`은 비어 있습니다. 1,128초는 실패 처리에서 보존한 loop 시각이며, 별도로 측정한 마지막 정상 SUMO 시각으로 표현하지 않습니다. 활성 구급차 두 대가 기록되었다고 해서 그중 하나가 엔진 충돌의 원인 차량이라는 뜻은 아닙니다.

[실제 실패 기록](../results/all_scenarios_normal15_yield_repeat_sumo128_v6/sumo/train_20261117_213050_c46a1a/episode27/sumo_failure.json)과 [실제 실행 명령](../results/all_scenarios_normal15_yield_repeat_sumo128_v6/sumo/train_20261117_213050_c46a1a/episode27/sumo_command.json)을 보존했습니다.

## 덤프에서 직접 확인한 내용

새 Windows minidump `sumo.exe.7784.dmp`를 읽기 전용으로 분석했습니다. 덤프에 기록된 실행 모듈 경로·SHA256은 앞서 검증한 portable SUMO 1.28.0과 일치합니다.

| 항목 | 확인값 |
|---|---|
| 예외 종류 | 읽기 접근 위반 `0xc0000005` |
| 잘못 읽으려던 주소 | `0x70` |
| SUMO 모듈 내 위치 | RVA `0x4d6789` |
| 기계 명령 | `41 8b 53 70` |
| 디스어셈블 결과 | `mov edx, dword ptr [r11 + 0x70]` |
| 해당 시점 R11 | `0x0` |
| 해당 RIP를 포함한 `.pdata` 범위 | `0x4d6710` ~ `0x4d6ada` |

따라서 **SUMO 네이티브 코드가 NULL 포인터의 `+0x70` 필드를 읽다가 종료된 것**은 확인했습니다. 기존 1.27.1 덤프에도 NULL+0x70 읽기 형태가 있었지만, 빌드가 다르므로 동일 함수라는 결론까지 내리지는 않습니다.

PE의 `.pdata`·unwind 정보와 덤프에 포함된 스택 메모리로 아래 14프레임을 복원했습니다. 직접 호출은 반환 주소 앞의 `call` 명령과 대상 주소도 확인했고, 간접 호출은 해당 명령 바이트를 확인했습니다. 아래 주소는 함수명이 없는 원시 증거입니다.

| 프레임 | 모듈 | RIP의 모듈 내 RVA |
|---:|---|---|
| 0 | sumo.exe | `0x4d6789` |
| 1 | sumo.exe | `0x4caa4c` |
| 2 | sumo.exe | `0x4d646e` |
| 3 | sumo.exe | `0x330d3e` |
| 4 | sumo.exe | `0x3300b2` |
| 5 | sumo.exe | `0x32fe71` |
| 6 | sumo.exe | `0x32e361` |
| 7 | sumo.exe | `0x2bf405` |
| 8 | sumo.exe | `0x25cd26` |
| 9 | sumo.exe | `0x25c38e` |
| 10 | sumo.exe | `0x16a0b` |
| 11 | sumo.exe | `0x48f8cf` |
| 12 | kernel32.dll | `0x2dda7` |
| 13 | ntdll.dll | `0xc9c5c` |

원시 레지스터·명령 바이트는 [덤프 분석 JSON](../results/diagnostics/sumo128_episode27_native_dump_20261117.json), unwind 연산·호출 대상·디스어셈블과 소스 대조는 [call chain 분석 JSON](../results/diagnostics/sumo128_episode27_unwind_20261117.json)에 저장했습니다.

## 공식 소스와 대조한 강한 추정

SUMO PDB가 없고 덤프의 CodeView 레코드도 비어 있으므로, 아래 함수명은 심볼로 확정한 값이 아닙니다. 다만 인자 배치, 분기 순서, 구조체의 크기·복사 위치와 직접 호출 관계가 공식 `v1_28_0` 소스와 일치합니다.

| 함수 시작 RVA | 대응하는 소스 후보 |
|---|---|
| `0x330940` | `MSLaneChangerSublane::checkChangeSublane` |
| `0x4d6290` | `MSLCM_SL2015::wantsChangeSublane` |
| `0x4ca430` | `MSLCM_SL2015::_wantsChangeSublane` |
| `0x4d6710` | `MSLCHelper::getRoundaboutDistBonus` |

충돌 함수는 두 구조체의 첫 포인터를 읽고, 그 포인터가 가리키는 `+0x70` 값을 비교한 뒤, 구조체의 `+0x30`·`+0x38`에 있는 포인터 차이로 벡터 크기를 비교합니다. 이는 [회전교차로 보너스 계산 소스](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/lcmodels/MSLCHelper.cpp#L39-L68)의 차선 인덱스·연속 차선 수 비교와 대응합니다. [LaneQ 정의](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/MSVehicle.h#L777-L798)도 바이너리의 첫 포인터·벡터 위치와 72바이트 배열 간격에 맞습니다.

호출자에는 세 `LaneQ`의 포인터·벡터를 0으로 초기화한 뒤 best-lane 배열을 검색해 복사하는 명령이 있습니다. 덤프의 해당 스택 영역에서도 **curr·neigh·best 후보 세 개의 lane 포인터와 continuation vector가 모두 0**입니다. [SL2015 검색 부분](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/lcmodels/MSLCM_SL2015.cpp#L1038-L1083)은 검색 성공 시 이 값을 채운 뒤 `assert`로 확인합니다. Release 실행에서 그 확인이 유효한 방어로 남지 않으면, 검색이 실패한 상태가 후속 helper의 NULL 역참조로 이어질 수 있습니다. 현재 상태는 이 가설과 일치하지만 **검색이 실패한 최초 이유는 아직 확인하지 못했습니다.**

검색 전에는 내부 차선을 일반 차선에 대응시키는 처리도 있습니다. 따라서 best-lane 배열의 내용, 차량의 현재 차선, 내부 차선의 대응 결과가 다음 관측 대상입니다. 저장된 현재·이웃 edge 주소는 같고, [상위 호출자](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/MSLaneChangerSublane.cpp#L611-L683)도 일반 sublane 검사에 대응하므로, opposite 전용 경로라고 단정하지 않습니다.

함수명에 회전교차로가 들어간다고 해서 **실제 충돌 차량이 회전교차로에 있었다고 확인한 것은 아닙니다.** 해당 함수는 실제 회전교차로 여부를 판단하기 전 차선 정보를 읽습니다. 또한 보너스 계수를 마지막 계산에 사용하는 구조이므로, `lcCooperativeRoundabout=0`만 설정하면 이 초기 NULL 접근이 사라진다고 볼 수 없습니다. 이 값을 해결책으로 적용하지 않았습니다.

## 관리 구급차와 배경 구급차의 구분

입력 교통에는 출발 시각 0초, `type="ambulance"`, blue-light 장치가 켜진 **배경 구급차 180대**도 있습니다. 이는 `--ambulances 10`으로 생성·관리하는 환자 이송 구급차와 별도입니다. 따라서 `--ambulances 10`은 에이전트가 배차하는 구급차 수이며, 전체 입력에 있는 구급차 차량의 총수를 뜻하지 않습니다.

현재 덤프에는 차량 객체 후보 주소 `0x2cdf1155210`의 첫 바이트조차 포함되어 있지 않습니다. Lane-change model·이웃 차선 객체 후보의 heap도 포함되지 않아 차량 ID를 복원할 수 없었습니다. 관리 구급차인지 배경 구급차인지, 또는 주변 일반 차량인지 아직 특정하지 않았습니다. 배경 구급차 180대라는 사실만으로 충돌 원인을 정하지 않습니다.

## 1차 사본 재생 결과와 재현 한계

26회·342 업데이트의 full checkpoint를 별도 진단 폴더에 복사하고 다음 회차인 27회·시나리오 9·시드 10069를 재생했습니다.

| 항목 | 1차 진단 결과 |
|---|---|
| 진단 실행 | `sumo_crash_20261117_214851_ad10e9` |
| 상태 | 정상 완료 |
| 발생 / 도착 | 15 / 15명 |
| 실패 / 검열 | 0 / 0명 |
| 종료 시각 | 3,601초 |
| 원본 체크포인트·manifest | 실행 전후 SHA256 동일 |
| 진단 중 가중치 변경 | 메모리에서만 진행, 저장하지 않음 |

[1차 진단 요약](../results/diagnostics/sumo128_episode27_replay/sumo_crash_20261117_214851_ad10e9/diagnostic_summary.json)에 이 결과를 남겼습니다. 실제 실패 실행과 차량 도착 시각·미시적인 이동 궤적이 달랐으므로, 이는 **기존 충돌 재현에 성공한 결과가 아닙니다.** 같은 환자 시드와 체크포인트를 사용했다는 사실만으로 전체 native 시뮬레이션 상태가 동일했다고 판단하지 않습니다. 왜 궤적이 달라졌는지도 아직 확정하지 않았습니다.

2차 probe에서 사용하는 `traci.vehicle.getBestLanes()`의 내부 동작도 확인했습니다. 공식 [libsumo getter 구현](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/libsumo/Vehicle.cpp#L455-L475)은 저장된 best-lane 정보를 응답으로 구성하며, 그 아래 [MSVehicle getter](https://github.com/eclipse-sumo/sumo/blob/v1_28_0/src/microsim/MSVehicle.cpp#L5796-L5799)도 저장된 첫 벡터를 반환합니다. 확인한 경로에는 `updateBestLanes()` 호출이 없습니다. 따라서 **이 getter가 캐시를 갱신해서 충돌을 회피한다고 설명할 근거는 없습니다.** 관측을 추가한 진단이 완료되더라도, 기존 실패 궤적과의 일치 및 반복 재현 없이 문제 해결로 집계하지 않습니다. 진단 getter 호출을 일반 학습의 충돌 우회책으로 적용하지 않았습니다.

## 2차 진단: 연결 종료·접근 위반 재현

양보 규칙을 유지한 2차 진단 `sumo_crash_20261117_220133_06ce2c`에서 충돌을 다시 확인했습니다. 마지막 정상 시각은 **1,124초**, 실패 명령은 다음 `traci.simulationStep()`이며 SUMO 종료 코드는 **`0xc0000005`**입니다. 원래 학습의 1,128초와는 미시적인 궤적·시각이 다르므로 완전히 동일한 상태·명령 주소의 재현으로 표현하지 않습니다.

- 1,100~1,124초의 차로 스냅샷 25개와 1,100초의 native 상태를 저장했습니다.
- 관리 차량 50개·배경 구급차 75개·내부 차로 차량 5,509개의 시점별 관측을 확보했습니다. 이는 고유 차량 수가 아닌 누적 관측 수입니다.
- 빈 best-lane 또는 현재 일반 차로 미포함 관측 flag는 0개였습니다. 일반 도로의 모든 승용차를 상세 조회한 것은 아니므로 정상 상태의 증거로 확대하지 않습니다.
- 원본 체크포인트·manifest는 변경되지 않았고, 진단 중 가중치는 저장하지 않았습니다.
- 진단 코드 회귀 테스트는 **291개** 통과했습니다. 이 결과는 SUMO 충돌 해결을 뜻하지 않습니다.

[2차 진단 요약](../results/diagnostics/sumo128_episode27_lane_probe/sumo_crash_20261117_220133_06ce2c/diagnostic_summary.json) · [native 실패 기록](../results/diagnostics/sumo128_episode27_lane_probe/sumo_crash_20261117_220133_06ce2c/sumo/episode27/sumo_failure.json)

관리 차량 `amb_e_6`의 마지막 내부 차로는 일반 차로 `251118731#0_0`으로 연결되며, 관측된 best-lane ID와도 일치했습니다. 이 일반 차로의 길이는 0.69m이고 앞선 내부 차로의 남은 길이는 약 2.03m였습니다. 한 시뮬레이션 단계에서 여러 차로 경계를 통과할 수 있어 연속 전환의 갱신 처리가 추가 조사 후보입니다. **이 차량이 충돌 원인이라고 확정하지 않았습니다.** [정적 연결 대조](../results/diagnostics/sumo128_episode27_lane_probe/sumo_crash_20261117_220133_06ce2c/native_state_replay_1/internal_lane_mapping.json)

1,100초 상태만 로드한 짧은 native 재생은 1,125초까지 정상 완료했습니다. 상태 파일에 RNG 정보가 없고 실제 차량 차로도 달라졌으므로, 충돌을 해결하거나 원래 상태를 완전히 복원했다는 근거가 아닙니다. [짧은 재생 결과](../results/diagnostics/sumo128_episode27_lane_probe/sumo_crash_20261117_220133_06ce2c/native_state_replay_1/result.json)

## 별도로 확인한 입력 오류와 사용자가 선택한 대조

배경 구급차 원본 180대를 전수 검사한 결과 **정상 73대·문제 107대**였습니다. 정적 네트워크 검사와 SUMO 1.28.0 `duarouter`의 strict 검증이 같은 107대의 경로를 거부했습니다. 긴급차 불허 도로·출발 차로 부재·허용 차로 연결 부재가 포함되며 오류 범주는 서로 중복됩니다. 180대는 원본 정의 수이고 실제 시나리오 삽입 수에는 교통 배율이 적용됩니다.

현재 설정의 `ignore-route-errors=true`는 이러한 입력 오류를 허용하고 있었습니다. **경로 오류가 이번 NULL 참조를 직접 일으켰는지는 입증하지 못했습니다.** [입력 전수 검증](../results/diagnostics/episode27_background_route_audit_20261117/audit_summary.json)

사용자는 **문제 107대를 제외하고 정상 73대의 원래 경로·차종·출발 시각을 유지한 사본 검증**을 선택했습니다. 원본 입력과 모델을 수정하지 않고 진단용 설정 사본에만 적용합니다. 실제 학습 재개와 새 환경에서의 모델 저장은 이 대조 시험에 포함하지 않습니다.

## 정상 73대 사본: 3회 반복 결과

같은 26회 체크포인트·시나리오 9·시드 10069로 별도 Python/SUMO 프로세스 3개를 실행했습니다. **3회 모두 엔진 중단 없이 시뮬레이션 3,601초까지 완료했습니다.** 시드는 모두 같으므로 서로 다른 검증 시드 3개를 평가한 것으로 표현하지 않습니다.

| 진단 실행 | 발생 | 도착 | 임무 실패 | 검열 | 엔진 중단 |
|---|---:|---:|---:|---:|---|
| 1 | 15 | 14 | 1 | 0 | 없음 |
| 2 | 15 | 15 | 0 | 0 | 없음 |
| 3 | 15 | 14 | 1 | 0 | 없음 |

임무 실패 2건의 이유는 `hospital_stop_missing`입니다. 이는 차량의 병원 정차를 확인하지 못한 경우이며, Python/SUMO 프로세스 중단과 구분합니다. 진단 중 탐색·학습을 메모리에서 수행한 결과로, best 모델의 정책 성능 평가에 해당하지 않습니다.

모든 실행에서 구급차 양보 ON, 관리 구급차 상한 10대, 승용차 원본 입력·반복을 유지했습니다. 정상 73대의 원래 정의만 남긴 candidate를 각 진단 폴더에 고정 복사했고, 원본 cfg·원본 route·candidate·적용 cfg·적용 route의 실행 전후 SHA256 검사도 모두 통과했습니다. 원본 24회 모델과 현재 26회·342 업데이트 모델은 보존되었으며, 진단을 학습 이력에 합치지 않았습니다.

- [3회 비교·입력 보존·테스트 종합 기록](../results/diagnostics/sumo128_valid73_comparison_20261117.json)
- [73대 후보·strict 경로 검증·정상 차량 원본 일치](../results/diagnostics/episode27_background_route_audit_20261117/valid73_candidate/candidate_manifest.json)
- [1회차 진단 결과](../results/diagnostics/sumo128_valid73_trial1/sumo_crash_20261117_222040_020d39/diagnostic_summary.json)
- [2회차 진단 결과](../results/diagnostics/sumo128_valid73_trial2/sumo_crash_20261117_222040_b5d831/diagnostic_summary.json)
- [3회차 진단 결과](../results/diagnostics/sumo128_valid73_trial3/sumo_crash_20261117_222040_b4307c/diagnostic_summary.json)
- [전체 코드 회귀 테스트: 296개 통과](../results/diagnostics/sumo128_episode27_tests_20261117_utf8.txt)

기존 원본 입력에서의 native NULL 참조 결함이 수정되었다거나, 잘못된 경로 107대가 그 결함의 직접 원인이라고 확정한 결과는 아닙니다. 정상 73대 조건에서 같은 실패 회차를 3회 완료했다는 근거입니다. 나머지 시나리오와 다른 시드까지의 무중단은 아직 검증하지 않았습니다.

현재 이 입력 변경은 **진단 사본에서만 적용**됩니다. 기존 `resume_sumo128.py`와 학습·평가·웹은 원래 180대 입력을 사용합니다. 실제 학습에 채택하려면 공통 환경 계층에서 73대 프로필을 복원하고 입력 해시를 새 실험 signature에 포함해야 합니다. 기존 26회 모델에 signature만 덮어써 동일 실험의 재개로 취급하지 않습니다. 새 학습 또는 가중치 전이 여부도 별도로 기록해야 합니다.

진단을 다시 실행하려면 프로젝트 PowerShell에서 아래 명령을 사용합니다. 이 명령은 학습 재개 명령이 아니며, 새 진단 폴더만 만듭니다.

```powershell
$env:SUMO_BINARY = (Resolve-Path '.runtime/sumo-1.28.0/bin/sumo.exe').Path
.venv/Scripts/python.exe -B -u CODE/diagnose_sumo_crash.py `
  --checkpoint results/all_scenarios_normal15_yield_repeat_sumo128_v6/training/checkpoint_v6_f560a1f65da40cc8.pth `
  --output-root results/diagnostics/valid73_recheck `
  --background-route-override results/diagnostics/episode27_background_route_audit_20261117/valid73_candidate/ambulance_routes.valid73.rou.xml
```

원본 24회 모델과 새 실험의 26회 모델은 계속 보존하고, 실패한 회차를 건너뛰거나 진단 성공을 학습 완료로 기록하지 않습니다.
