# 공개 RoboTwin 55k EMA 모델 비교 평가

**후속 조치(2026-09-28):** 공개 모델의 특징 호환성을 수정하고 새 150k EMA도 내려받았다.
두 모델 모두 Transformers 4.56.0 참조와 고정 관측에서 출력이 일치한다.
현재 실행·재평가 명령과 실제 검증 결과는 [공개 모델 호환성 준비 기록](PUBLIC_MODEL_COMPATIBILITY_KO.md)을 따른다.
아래 9.8%와 실행 설명은 **수정 전 평가의 보존 기록**이다.

**완료:** 2026-09-28 09:28:48 KST, 50개 작업 × 10회에서 **49/500 성공, 9.8%**.
전체 평가와 집계는 정상 종료됐다. 아래 호환성 수정은 이번 실행에 적용되지 않았다.
[최종 결과와 검증](EVALUATION_RELEASED55K_KO.md).

2026-09-28 진단: 현재 환경의 Transformers 4.57.6에서 DINOv3 특징 선택이 저자의 기존 환경과
달라지는 문제를 확인했다. 동일 55k 가중치의 별도 관측 검사에서 정규화 후 특징을 선택하면
행동 오차가 약 83% 감소했다. 당시 실행 중이던 평가 설정은 유지했다.
이번 결과를 논문 성능으로 해석하기 전에 [진단 기록](PUBLIC_MODEL_DIAGNOSIS_20260928_KO.md)을 확인한다.
또한 평가 시작 뒤 공식 배포가 Clean+Randomized 150k EMA로 변경됐다. 이번 실행은 고정된 기존 55k다.

2026-09-27 22:36:45 KST에 저자가 공개한 55,000 step EMA 모델의 평가를 시작했다.
Run ID: `turbovla_released55k_eval10_20260927_2237`.
tmux 세션: `turbovla_release55k_20260927`.
완료 여부와 결과는 아래 상태 명령 또는 결과 파일로 확인한다.

기존 28.0%는 우리가 학습한 `steps_5000_ema_model.safetensors`의 결과다.
이번 공개 모델도 `steps_55000_ema_model.safetensors`로, 두 모델 모두 EMA 가중치를 사용한다.
두 실행의 EMA decay 설정은 0.999다. 이번에는 추가 학습 없이 공개 가중치를 평가한다.

| 항목 | 기존 모델 | 이번 공개 모델 |
|---|---|---|
| 학습 주체 | 로컬 5k 예비 학습 | 저자 공개 가중치 |
| 학습 step | 5,000 | 55,000 |
| 평가 가중치 | EMA | EMA |
| 평가 범위 | Clean50 × 10회 | Clean50 × 10회 |
| 평가 seed / instruction | 0 / unseen | 0 / unseen |
| 추론 정밀도 | BF16 | BF16 |
| 성공률 | 140/500, 28.0% | 49/500, 9.8% — 특징 호환성 수정 전 |

평가 규모 질문에 답변이 없어, 먼저 기존 실행과 같은 50개 작업 × 10회로 비교하는 옵션을 적용했다.
이는 논문의 작업당 100회·총 5,000회 평가와 구분한다.
기존 평가에서는 500회에 약 9시간 38분이 걸렸지만, 공개 모델은 성공 시 조기 종료하는 비율이 달라
소요 시간이 같다고 보장할 수 없다.

**공개 가중치와 로딩 검증**

- 공식 배포: [H-EmbodVis/TurboVLA](https://huggingface.co/H-EmbodVis/TurboVLA).
- 고정 revision: `cb5300544693013164c4bb251a13036002a55c81`.
- 원본 가중치: 868,178,034 bytes, 878개 BF16 tensor.
- 공식 LFS SHA256과 일치: `d0183df6bafd44507b6c797da5c5ab080ef8446cde4a8127d7280546d9f7c034`.
- 구형 `GroundingDINODiT`의 tensor 이름을 현재 `TurboVLA` 이름으로 변환했다. 원본 파일은 보존한다.
- 878개 tensor의 이름 대응, shape, dtype, 값의 정확한 보존, 유한값 및 strict loading을 확인했다.
- 배포된 `dataset_statistics.json`을 변환본에 그대로 복사했다.
- 실제 평가에 쓰는 state/action min·max와 mask는 기존 5k 모델의 통계와도 정확히 같다.
- 공식 코드 commit `b29ab1420baa5c663ec935df513f2012430beb67`의 core model을 별도 이름으로 불러
  같은 가중치와 입력에서 비교했다. 실제 데이터의 3개 작업 이미지를 이용한 batch 1·3에서
  `(B, 50, 14)` 출력이 모두 유한하고 bitwise 동일했다(최대 차이 0).

이 출력 비교는 현재 공개된 공식 코드에 대한 검사다. 논문 작성 당시의 미공개 runtime이나
저자의 정확한 시뮬레이터 commit까지 동일하다는 검증은 아니다. 기존 RoboTwin 환경을 그대로 사용한다.
배포 `config.yaml`의 `max_train_steps=100000`은 원본 설정값으로 보존했다.
평가 가중치의 step은 배포 파일명과 run ID에 명시된 55,000이며, 이번 실행에서 학습을 수행하지 않는다.

검증 기록:
[다운로드·체크섬](../../pretrained/releases/turbovla_robotwin_cb530054/download-verification.json),
[변환 기록](../../pretrained/converted/turbovla_robotwin_cb530054_55k_ema/conversion.json),
[추론 동등성 검사](setup_records/released_model_20260927/inference_validation.json),
[정규화 통계 대조](setup_records/released_model_20260927/normalization_comparison.json).

**실행과 모니터링**

당시 `evaluate_released_model.sh`가 시뮬레이터 사전 검사 2회를 먼저 수행했다.
이 검사가 정상 종료한 뒤 GPU당 한 작업씩, 두 GPU에서 전체 500회를 평가했다.
사전 검사 결과는 본 평가 점수에 포함하지 않는다.
전체 작업이 완료된 경우에만 `summary.json`과 `task_results.csv`를 만들고,
기존 5k 평가와 비교한 `comparison_to_pilot5k.json`·`.csv`를 생성한다.
작업별 첫 10회의 seed 일치 여부도 기록하며, 동일한 instruction 문자열까지 확인했다고 해석하지 않는다.

```bash
cd /home/jack/TurboVLA
watch -n 10 python3 scripts/robotwin/pipeline_status.py turbovla_released55k_eval10_20260927_2237
```

`partial_evaluation`은 현재까지 완료된 시도 수이며 최종 50개 작업 평균이 아니다.
모든 작업과 집계가 끝나면 `stage=completed`와 `evaluation_summary`, `comparison_to_pilot5k`가 표시된다.

```bash
tmux attach -t turbovla_release55k_20260927
```

`Ctrl-b` 후 `d`를 누르면 평가를 유지한 채 화면에서 나온다.
학습·평가를 실행하는 tmux 화면에서 `Ctrl-C`를 누르면 실행이 중단될 수 있다.
별도 `watch` 조회 화면의 `Ctrl-C`는 조회만 종료한다.

- [실행 식별자와 가중치 경로](setup_records/active-evaluation.json)
- [평가 결과 폴더](/data/jack/robotwin2_turbovla/evaluation/turbovla_released55k_eval10_20260927_2237_seed0)
- [제어 로그·상태·코드 보존 폴더](/data/jack/robotwin2_turbovla/pipelines/turbovla_released55k_eval10_20260927_2237)

가중치는 SSD의 `pretrained/releases/`와 `pretrained/converted/`에 보관하고,
평가 로그·영상은 기존과 같이 HDD에 저장한다. 기존 5k 결과는 변경하지 않는다.

현재 `evaluate_released_model.sh`는 호환성 수정 번들과 검증된 새 실행기를 사용하도록 바뀌었다.
다음 명령은 **수정 후 새 평가를 시작**하며 위 9.8% 조건을 재실행하지 않는다.
GPU 사용 중에는 실행을 거부한다. 새 실행은 코드·설정·통계 검증 후 평가 및 집계를 수행한다.

```bash
cd /home/jack/TurboVLA
ROBOTWIN_TEST_NUM=10 bash scripts/robotwin/evaluate_released_model.sh
```

작업당 100회로 확대할 때는 `ROBOTWIN_TEST_NUM=100`을 지정할 수 있다.
이는 추가 실행이며, 이번 10회 평가가 자동으로 100회로 늘어나지는 않는다.
