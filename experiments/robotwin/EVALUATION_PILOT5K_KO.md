# RoboTwin Clean50: 5k 예비 학습 평가 결과

**50개 작업에서 각각 10회 평가하여 총 500회 중 140회 성공, 평균 성공률 28.0%를 기록했다.**
학습 후 자동 평가가 2026-09-27 19:05:03 KST에 완료되었고, 이후 결과를 검증했다.
학습과 전체 평가에 16시간 28분이 걸려 하루 이내 결과 확인이라는 목표를 충족했다.

| 항목 | 결과 |
|---|---|
| 학습 | 5,000 optimizer update, 전역 배치 192, ZeRO-1, RTX 3090 두 장 |
| 평가 가중치 | `steps_5000_ema_model.safetensors`, EMA 0.999 |
| 평가 조건 | RoboTwin 2.0 `demo_clean`, unseen instruction, seed 0 |
| 작업·시도 | 50개 작업 × 10회 = 500회 |
| 성공·실패 | 140회 성공 / 360회 실패 |
| 작업 평균 성공률 | 28.0%; 작업별 시도 수가 같아 전체 성공률과 동일 |
| 한 번 이상 성공한 작업 | 36개 / 50개 |
| 10회 모두 성공한 작업 | 4개 / 50개 |
| 성공이 없었던 작업 | 14개 / 50개 |
| 마지막 학습 loss | 5,000 step에서 L1 0.0138334 |

이번 실행은 5,000 step·작업당 10회로 제한한 예비 실험이다.
기존 [재현 계획](ONE_DAY_PILOT_KO.md)의 55,000 step·작업당 100회 조건과 구분해야 한다.
작업당 성공 한 번이 10%p 차이이므로 작은 성공률 차이로 작업 난이도나 모델 우열을 단정할 수 없다.
학습 loss는 낮아졌지만, 실제 평가에서 14개 작업은 0/10이었다.
이 결과만으로 학습량 부족 등 특정 실패 원인을 확정할 수는 없다.

| 단계 | 시각(KST, 2026-09-27) | 소요 시간 |
|---|---|---|
| 학습 | 02:36:44 → 09:26:44 | 약 6시간 50분 |
| 평가: 사전 검사 2회 포함 | 09:26:55 → 19:05:03 | 약 9시간 38분 |
| 전체 | 02:36:44 → 19:05:03 | 약 16시간 28분 |

사전 검사 `adjust_bottle` 2회는 1회 성공했다. 이 검사는 본 평가 500회 및 28.0%에 포함하지 않았다.

**결과 검증**

50개 작업이 중복·누락 없이 존재하고, 각 로그의 누적 시도 횟수가 1부터 10까지 기록되었음을 확인했다.
성공 횟수는 작업 로그, 시뮬레이터의 `_result.txt`, 작업별 CSV, 최종 JSON에서 모두 일치한다.
모든 정책 서버가 5,000 step EMA 체크포인트를 사용했고, 학습과 전체 파이프라인 종료 코드는 0이다.
평가 직전 파이프라인이 체크포인트의 878개 tensor 유한값과 SHA256을 검사했다.
이번 사후 검증에서는 그 기록과 현재 파일 크기를 확인했으며, 대용량 가중치를 다시 해시하거나 로드하지 않았다.

평가 로그에는 9개 작업에서 총 20건의 예외가 있다.
모두 모델 rollout 전에 `TASK_ENV.play_once()`로 장면의 전문가 경로를 검사하다 발생한 예외였다.
평가 코드가 예외를 처리하고 다음 seed로 넘어갔으며, 이후 작업마다 채점 대상 10회를 완료했다.
이 20건은 전체 seed 제외 건수와 같지는 않다. 예외 없이 전문가 검사에 실패한 장면도 제외된다.
따라서 이 결과는 RoboTwin 평가 코드의 전문가 검사를 통과한 장면에 대한 성공률이다.
모델 정책 서버의 traceback이나 GPU 메모리 부족·연결 거부 등 검색한 치명적 오류 패턴은 발견되지 않았다.

500개 episode 영상의 존재와 파일 크기가 0보다 큼을 확인했다.
이번 검증에서 영상 500개 전체를 디코딩하거나 실패 행동을 시각적으로 분류한 것은 아니다.
검증 JSON에는 작업별 seed, 각 episode의 성공 여부, 영상 경로, 예외 건수 및 검증 파일 SHA256을 저장했다.

- [작업별 결과 CSV](evaluation_records/turbovla_clean50_pilot5k_2x3090_20260927_0236/task_results.csv)
- [검증 기록과 500개 영상 경로](evaluation_records/turbovla_clean50_pilot5k_2x3090_20260927_0236/validation.json)
- [학습 loss CSV](evaluation_records/turbovla_clean50_pilot5k_2x3090_20260927_0236/training_loss.csv)
- [원본 평가 로그 및 집계 폴더](/data/jack/robotwin2_turbovla/evaluation/turbovla_clean50_pilot5k_2x3090_20260927_0236_seed0)

학습 loss CSV는 50 update마다 기록된 rank 0의 마지막 소배치 L1 오차다.
누적 전역 배치 평균이나 별도 검증 데이터의 loss가 아니다.

**50개 작업별 결과**

| 작업 | 성공 / 시도 | 성공률 |
|---|---:|---:|
| adjust_bottle | 7 / 10 | 70% |
| beat_block_hammer | 6 / 10 | 60% |
| blocks_ranking_rgb | 0 / 10 | 0% |
| blocks_ranking_size | 0 / 10 | 0% |
| click_alarmclock | 1 / 10 | 10% |
| click_bell | 4 / 10 | 40% |
| dump_bin_bigbin | 6 / 10 | 60% |
| grab_roller | 10 / 10 | 100% |
| handover_block | 0 / 10 | 0% |
| handover_mic | 4 / 10 | 40% |
| hanging_mug | 0 / 10 | 0% |
| lift_pot | 1 / 10 | 10% |
| move_can_pot | 1 / 10 | 10% |
| move_pillbottle_pad | 1 / 10 | 10% |
| move_playingcard_away | 2 / 10 | 20% |
| move_stapler_pad | 0 / 10 | 0% |
| open_laptop | 1 / 10 | 10% |
| open_microwave | 3 / 10 | 30% |
| pick_diverse_bottles | 1 / 10 | 10% |
| pick_dual_bottles | 3 / 10 | 30% |
| place_a2b_left | 0 / 10 | 0% |
| place_a2b_right | 0 / 10 | 0% |
| place_bread_basket | 3 / 10 | 30% |
| place_bread_skillet | 2 / 10 | 20% |
| place_burger_fries | 6 / 10 | 60% |
| place_can_basket | 4 / 10 | 40% |
| place_cans_plasticbox | 2 / 10 | 20% |
| place_container_plate | 9 / 10 | 90% |
| place_dual_shoes | 2 / 10 | 20% |
| place_empty_cup | 5 / 10 | 50% |
| place_fan | 0 / 10 | 0% |
| place_mouse_pad | 0 / 10 | 0% |
| place_object_basket | 5 / 10 | 50% |
| place_object_scale | 0 / 10 | 0% |
| place_object_stand | 3 / 10 | 30% |
| place_phone_stand | 0 / 10 | 0% |
| place_shoe | 1 / 10 | 10% |
| press_stapler | 10 / 10 | 100% |
| put_bottles_dustbin | 1 / 10 | 10% |
| put_object_cabinet | 2 / 10 | 20% |
| rotate_qrcode | 2 / 10 | 20% |
| scan_object | 0 / 10 | 0% |
| shake_bottle_horizontally | 10 / 10 | 100% |
| shake_bottle | 10 / 10 | 100% |
| stack_blocks_three | 0 / 10 | 0% |
| stack_blocks_two | 2 / 10 | 20% |
| stack_bowls_three | 3 / 10 | 30% |
| stack_bowls_two | 6 / 10 | 60% |
| stamp_seal | 0 / 10 | 0% |
| turn_switch | 1 / 10 | 10% |
