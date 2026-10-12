공개 55k EMA 평가 결과 — DINOv3 특징 호환성 수정 전

2026-09-28 09:28:48 KST에 `turbovla_released55k_eval10_20260927_2237`이
`stage=completed`, `exit_code=0`으로 종료됐다. 시작부터 사전 검사를 포함해 약 10시간 52분이 걸렸다.

| 모델 | 범위 | 성공 | 성공률 |
|---|---|---:|---:|
| 로컬 5k EMA | Clean 50개 작업 × 10회 | 140/500 | 28.0% |
| 공개 55k EMA, 수정 전 환경 | Clean 50개 작업 × 10회 | 49/500 | **9.8%** |

두 실행 모두 EMA 가중치다. 공개 모델은 29개 작업에서 0/10, 21개 작업에서 1회 이상 성공했고,
10/10 성공한 작업은 없었다. 작업 범위는 같지만 episode seed 목록은 50개 중 33개 작업에서만
완전히 일치한다. instruction 문자열의 동일성은 기록으로 검증하지 않았다.

| 상대적으로 높은 성공률의 작업 | 성공 |
|---|---:|
| grab_roller | 7/10 |
| shake_bottle_horizontally | 6/10 |
| shake_bottle | 6/10 |
| place_container_plate | 4/10 |
| click_bell | 3/10 |
| move_can_pot | 3/10 |
| stack_bowls_two | 3/10 |

50개 작업 로그의 1~10회 순차 집계가 CSV와 일치하고, 합계는 summary.json과 일치했다.
50개 정책 서버가 모두 지정한 공개 55k EMA를 사용했다. 시뮬레이터 결과 파일 50개와
빈 파일이 아닌 평가 영상 500개가 존재한다. 실패 451회는 모두 제어 step 제한에 도달했다.
expert 장면 검증 과정의 caught traceback 18개가 있었지만 이들은 graded policy trial 이전의
seed 검증이며, 전체 평가와 집계는 정상 종료됐다.

**이 결과는 DINOv3 특징 호환성 문제를 수정하기 전의 점수다.**
실행 당시와 현재의 vision_encoder.py가 바이트 단위로 같고,
Transformers 4.57.6에서 `hidden_states[-1]`를 사용하는 것을 재확인했다.
분리된 진단 검사에서만 `last_hidden_state` 선택을 시험했으며,
그 수정으로 시뮬레이터를 평가한 결과는 아직 없다.

따라서 이번 9.8%를 공개 모델의 논문 조건 성능으로 해석하거나,
5k 모델이 55k 공개 모델보다 본질적으로 우수하다고 결론 내릴 수 없다.
또한 이번은 작업당 10회이며 원래 논문의 작업당 100회 평가와 구분한다.
다음 검증은 기존 55k 가중치와 나머지 조건을 고정하고 시각 특징 선택만 수정한 실제 rollout 비교다.

- [전체 작업별 결과](/data/jack/robotwin2_turbovla/evaluation/turbovla_released55k_eval10_20260927_2237_seed0/task_results.csv)
- [5k 모델과의 작업별 비교](/data/jack/robotwin2_turbovla/evaluation/turbovla_released55k_eval10_20260927_2237_seed0/comparison_to_pilot5k.csv)
- [완료 검증 기록](setup_records/released_model_20260927/final_evaluation_verification.json)
- [호환성 원인 분석과 별도 예측 검사](PUBLIC_MODEL_DIAGNOSIS_20260928_KO.md)
