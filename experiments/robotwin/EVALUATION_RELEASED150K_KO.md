# 공개 150k EMA 평가 결과 — 2026-09-29

**Clean·Randomized 평가가 모두 정상 완료됐다. 총 882/1,000 성공, 평균 88.2%다.**
평가에 사용한 150k 배포의 공식 평균 88.06%와 0.14%p 차이다.
호환성을 수정한 환경에서 공식 성공률 수준에 근접한 결과를 얻었다.
이번에는 작업당 10회를 실행했으므로, 공식 작업당 100회와 같은 규모의 검증은 아니다.

50개 태스크의 한글 설명·평가 방식·로그 링크는 [평가 로그 안내](EVALUATION_LOG_GUIDE_KO.md)에 정리했다.
[통합 결과 JSON](evaluation_records/released150k_20260929/evaluation_results.json)과
[1,000회 회차별 결과 JSON](evaluation_records/released150k_20260929/episode_results.json)에서 점수, 지시문, seed 및 영상 경로를 확인할 수 있다.

## 결과와 공식 보고값

| 조건 | 평가 규모 | 성공 | 이번 성공률 | 공식 보고값 | 차이 |
|---|---|---:|---:|---:|---:|
| Clean | 50개 작업 × 10회 | 443/500 | **88.60%** | 88.02% | +0.58%p |
| Randomized | 50개 작업 × 10회 | 439/500 | **87.80%** | 88.10% | −0.30%p |
| 평균 / 합계 | 총 1,000회 | 882/1,000 | **88.20%** | 88.06% | +0.14%p |

공식 값은 평가 당시 고정한 코드 commit `6727c875666f8d5dda8d8cca0043da200738fe73`의
[공식 성능표](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/assets/figures/performance.png)와 비교했다.
당시 보관한 [성능표 이미지](setup_records/compatibility_20260928/official_performance_6727c875.png)에서도 확인했다.
[공식 평가 안내](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/experiments/robotwin/README.md)는
두 조건 각각 50개 작업 × 100회, 합계 10,000회를 사용한다.

각 작업의 시도 수가 같으므로 작업별 성공률 평균과 전체 성공 수/시도 수가 같다.
Clean과 Randomized의 평가 규모도 같아, 두 조건의 평균은 전체 882/1,000과 같다.
원본 통합 JSON의 `88.19999999999999`는 부동소수점 표현이며 표에는 88.2%로 표시했다.

## 종료 및 소요 시간

모든 시각은 KST다. 두 RTX 3090에서 GPU당 한 작업씩 평가했고, Clean 이후 Randomized를 순서대로 실행했다.

| 조건 | 시작 | 종료 | 소요 시간 |
|---|---|---|---|
| Clean | 9/28 15:15:58 | 9/28 20:42:32 | 5시간 26분 33초 |
| Randomized | 9/28 20:42:37 | 9/29 03:10:01 | 6시간 27분 24초 |
| 전체 큐 | 9/28 15:15:53 | 9/29 03:10:01 | 약 11시간 54분 |

두 pipeline 모두 `stage=completed`, `exit_code=0`이고 통합 큐도 `completed`다.
[통합 결과](setup_records/compatibility_20260928/latest150k_eval10_summary.json),
[큐 기록](setup_records/compatibility_20260928/benchmark_queue.json).

## 확인한 모델과 평가 조건

- 모델: 저자가 공개한 **150,000-step EMA**, 추가 학습 없이 평가.
- Hugging Face revision: `9cdb576ed9c56039b9c1149142ed654e876bad76`.
- 실제 가중치: `pretrained/converted/turbovla_robotwin_9cdb576e_150k_ema_postnorm/steps_150000_ema_pytorch_model.safetensors`.
- 변환된 가중치 SHA256: `a62501e2f424275bf71a763ccaff90b1c0b94e65b6fc58e69a4e97821171d7d7`.
- 해당 배포의 설정 및 정규화 통계 사용. 원본 가중치와 변환본의 tensor 값 보존 검사를 앞서 통과했다.
- DINOv3 `feature_source=last_hidden_state`, 공식 `compatibility_profile=robotwin`.
- 정책 환경: Torch 2.6.0+cu124, Transformers 4.57.6, BF16.
- RoboTwin commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`.
- 평가 seed=0, unseen instruction, instruction 생성 개수=100, rollout은 작업당 10회.
- 3개 RGB 카메라, 절대 관절 위치 제어, action horizon 50, temporal ensemble 설정 유지.

이 모델은 Clean+Randomized를 함께 학습한 새 배포다. 기존 v1 Clean 55k의 60.2% 실험과 구분한다.
이전 55k·수정 전 9.8%와 이번 150k·88.2%의 차이는 가중치와 학습 데이터도 달라,
그 차이 전부를 특징 수정의 효과로 계산할 수 없다.
동일 55k에서의 수치 검사와 작은 rollout 개선은 [호환성 보고서](PUBLIC_MODEL_COMPATIBILITY_KO.md)에 별도 기록했다.

## 완료 및 무결성 검사

2026-09-29에 [검사 스크립트](setup_records/compatibility_20260928/audit_full_evaluation.py)를 실행했다.
기존 평가 로그와 집계 파일은 수정하지 않았다.

- 각 조건에서 정확히 50개 작업, 작업당 10개 graded trial을 확인했다.
- 모든 trial의 누적 성공 수·seed·지시문·성공/실패 표식이 일관되며 CSV 및 통합 JSON과 일치한다.
- 시뮬레이터가 저장한 100개 `_result.txt`도 각각의 작업 점수와 일치한다.
- 서버 로그 100개에서 올바른 150k 가중치와 수정된 특징/profile 사용을 확인했다.
- 가중치·설정·통계의 SHA256을 검증 기록과 대조했다. 실행별 보존 파일 56개씩의 해시가 일치한다.
  보관된 추론·평가 source도 검사 시점의 실제 source와 같다.
- **영상 1,000개**가 존재하며 모두 양수의 frame 수·재생 길이와 유효한 video stream 정보를 갖는다.
  이는 `ffprobe` 스트림 정보 검사이며 전체 프레임 디코딩 검사는 아니다.
- 실패는 Clean 57회, Randomized 61회이며, **118회 모두 제어 step 제한에 도달**했다.
  graded rollout 내부에는 traceback이 없었다.
- Expert/scene 검증 중 caught traceback은 각 조건 16개였다. 모두 graded trial 시작 전이며,
  원래 simulator의 seed 검증 절차에 따라 다음 장면을 선택했다. 1,000회 policy trial 수에는 포함하지 않는다.

검사 자료: [완료·점수·가중치·에피소드 감사](evaluation_records/released150k_20260929/completion_audit.json),
[영상 스트림 검사](evaluation_records/released150k_20260929/video_stream_audit.json),
[50개 작업별 비교 CSV](evaluation_records/released150k_20260929/task_comparison.csv).

## 작업별 결과에서 확인할 부분

두 조건 모두 10/10 성공 작업이 27개이고, 0/10 작업은 없다.
다음은 두 조건을 합했을 때 성공 수가 가장 적은 다섯 작업이다.

| 작업 | Clean | Randomized | 합계 |
|---|---:|---:|---:|
| turn_switch | 5/10 | 5/10 | 10/20 |
| hanging_mug | 4/10 | 7/10 | 11/20 |
| place_a2b_left | 6/10 | 6/10 | 12/20 |
| place_dual_shoes | 7/10 | 5/10 | 12/20 |
| put_object_cabinet | 6/10 | 6/10 | 12/20 |

작업별 10회로는 순위와 조건 간 차이의 불확실성이 크다. 공식의 작업별 점수나 같은 episode 입력과
직접 비교한 결과는 아니다. 공식 시뮬레이터가 Python `random`을 고정하지 않으므로 같은 scene seed도
지시문을 동일하게 보장하지 않는다. 실제 선택한 seed와 지시문은 이번 실행 전체에서 기록했다.

## 재현 판단과 남은 범위

**공개 150k EMA의 성공률 수준은 이 축소 평가에서 재현됐다고 볼 근거가 충분하다.**
두 조건의 차이가 각각 +0.58%p, −0.30%p이므로 기존의 큰 성능 격차는 나타나지 않았다.
다만 작은 점수 차이가 우열이나 통계적 동등성을 증명하는 것은 아니다.
동일 규모의 재현 검증에는 공식처럼 작업당 100회 평가가 남아 있다.
이번 결과는 공개 가중치의 추론·평가 재현이며, 우리의 학습 과정 재현이나 기존 5k 모델의 성능 향상을 뜻하지 않는다.

결과 경로:

- [Clean 결과](/data/jack/robotwin2_turbovla/evaluation/turbovla_release150000_postnorm_demo_clean_eval10_20260928_151553_seed0)
- [Randomized 결과](/data/jack/robotwin2_turbovla/evaluation/turbovla_release150000_postnorm_demo_randomized_eval10_20260928_151553_seed0)
