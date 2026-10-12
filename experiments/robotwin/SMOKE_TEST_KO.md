**RoboTwin Clean50 학습 스모크: 통과**

2026-09-27에 첫 실행이 정상 종료했다(exit code 0). 시작 01:37:03 KST, 종료 01:41:52 KST.
초기화·학습·저장을 포함해 4분 49초였으며, 별도로 체크포인트 검증과 재로딩·추론을 수행했다.
이 스모크 실행에서는 본학습과 RoboTwin 시뮬레이터 평가를 수행하지 않았다.
이후 시작한 55k 실행은 02:25 KST에 중단하고 속도 최적화와 단축 실행으로 전환했다.
현재 구성은 [5k 예비 실험 기록](ONE_DAY_PILOT_KO.md)을 참고한다.

실행 디렉터리:
`/data/jack/robotwin2_turbovla/runs/turbovla_clean50_smoke_20260927_a`

| 항목 | 결과 |
| --- | --- |
| 환경 | `turbovla-robotwin`, PyTorch 2.6.0+cu124, Accelerate 1.5.2, DeepSpeed 0.16.9 |
| 모델 / 데이터 | DINOv3 ViT-L TurboVLA / 50개 작업, 2,500개 시연, 549,787 transitions |
| 실행 구성 | RTX 3090 × 2, GPU당 batch 6, accumulation 16, global batch 192 |
| 학습 길이 | optimizer update 20회, warmup 5회, 최대 LR 5e-5, EMA decay 0.999 |
| 각 rank의 집계 | micro-batch 320개, optimizer/scheduler/EMA update 각각 20회 |
| 학습 샘플 수 | 전체 GPU 합계 3,840개; 별도 batch 평가 입력 제외 |
| loss | 첫 update 평균 0.4650 → 마지막 update 평균 0.2134 |
| batch 평가 | update 10·20에서 총 2회, simulator rollout은 아님 |
| 파라미터 변경 | action head 표본 파라미터의 최대 변화 0.00081316, 두 rank 일치 |
| 최대 GPU 할당 | rank 0: 11.475 GiB / rank 1: 11.492 GiB |
| 최대 GPU 예약 | rank 0: 13.369 GiB / rank 1: 13.389 GiB |
| 저장 | step-20 raw/EMA와 final raw/EMA, 총 4개 safetensors |
| 재로딩 | EMA strict 로딩 성공, GPU BF16 추론 출력 `(1, 50, 14)`, 모든 값 유한 |

loss는 각 update에 사용한 16개 micro-batch와 두 rank를 평균한 값이다.
GPU 메모리는 이 PyTorch 학습 프로세스의 측정치이며 기존 작업의 메모리는 포함하지 않는다.
스모크 시작 시 기존 GPU 작업이 있었으므로 실행 속도는 단독 GPU 성능 측정으로 해석하지 않는다.

**검증 근거와 범위**

[rank 0 기록](smoke_records/smoke_audit_rank0.json), [rank 1 기록](smoke_records/smoke_audit_rank1.json).
검사용 진입점은 실제 `EMAVLATrainer`를 상속해 원래 학습 경로를 실행한다.
각 micro-batch마다 DeepSpeed `global_steps`, scheduler `last_epoch`, EMA 호출 횟수를 비교했다.
누적 경계에서만 각각 1씩 증가하고, optimizer skip이나 비유한 loss가 없음을 확인했다.
실행은 최대 20 update로 제한되며 모델 구조와 loss 계산을 바꾸지 않는다.

체크포인트는 각각 1,736,262,180 bytes, 878개 tensor이다.
4개 파일 전체 tensor의 유한값과 SHA-256을 확인했으며 step-20과 final의 raw 파일끼리,
EMA 파일끼리 각각 해시가 일치했다. raw와 EMA의 action head 표본 값은 서로 다르다.
[체크포인트 검증 결과](smoke_records/checkpoint_validation.json).

사용할 EMA 파일:
`checkpoints/steps_20_ema_model.safetensors`

해당 파일의 SHA-256:
`0d075e022ca0c0f11c9cf7f6e069bb2d1f783b22126a211ad6d663f432662efd`.
재로딩 후 합성 RGB 3장과 0 state를 입력해 inference 입출력을 확인했다.
이 검사는 시뮬레이터 작업 성공률이나 20-step 모델의 정책 품질을 검증한 결과가 아니다.

**시간과 데이터 대기**

rank 0에서 20 update 경계까지 약 239.3초, 첫 update를 제외한 중앙값은 update당 약 12.18초였다.
이 시간에는 data 대기와 update 10의 batch 평가 등 경계 사이 작업이 포함된다.
기존 GPU 작업과 초기 warmup의 영향을 받으므로 55k 학습 시간을 확정하는 측정으로 사용하지 않는다.

`next(dataloader)`에서 대기한 시간은 rank 0 약 0.951초, rank 1 약 0.994초였다.
학습 320개 + 평가 2개 batch 기준 평균 약 3ms이다. worker의 영상 디코딩은 미리 진행되므로
이 값은 디코딩 자체의 전체 처리시간이 아니라 학습 루프가 기다린 시간이다.
이번 SSD 기반 RoboTwin 실행에서는 데이터 대기가 작았다. HDD나 LIBERO와의 직접 비교는 수행하지 않았다.

**적용한 변경**

1. `reproduction_accumulation.patch`를 적용했다. scheduler·평가·로그·저장을 optimizer update 기준으로 묶고,
   gradient 초기화·clipping 시점과 ZeRO-2의 accumulation 설정을 보정했으며 모델 생성 전에 seed를 고정했다.
2. `scripts/robotwin/smoke_train.sh`, `smoke_train.py`를 추가했다. 실행 범위를 제한하고 rank별 수치를 검증·기록한다.
3. 통계 캐시를 데이터 원본과 분리했다. 초기 실행에서 로더가 배포된 `meta/stats_gr00t.json` 50개를
   자체 캐시 형식으로 다시 기록하는 동작을 확인했다. 생성한 통계는
   `/home/jack/datasets/RoboTwin/.cache/turbovla_statistics`에 보존하고, 배포 메타파일은 원래 revision으로 복원했다.
   `datasets.py`에 `statistics_cache_dir` 지원을 추가하고 `clean50.yaml`에 경로를 지정했다.

통계 캐시 경로 변경은 학습 시작 후에 적용했으므로, 완료된 학습의 코드와 설정은 run의 `provenance/`가 기준이다.
수정된 로더는 별도로 50개 작업 전체를 로드해 검증했다. 정규화 결과는 실제 학습에 사용한 결과와 완전히 같았고,
원본 메타파일이 변경되지 않았으며 실제 샘플의 3개 view, state `(1,14)`, action `(50,14)` 읽기도 통과했다.
[캐시 분리 검사](smoke_records/statistics-cache-check.json).
마지막으로 배포 원본 10,404개 파일 전체가 다운로드 manifest의 크기·해시와 일치함을 재확인했다.
[학습 후 데이터 무결성](smoke_records/dataset-integrity-after-smoke.json).

모델 구조, action horizon, 입력 view 수, L1 loss는 변경하지 않았다.
현재 남은 단계는 검증된 설정을 바탕으로 본학습을 시작하거나, RoboTwin 평가 환경을 준비하는 것이다.

재실행 명령은 다음과 같다. 자동으로 새 run 이름을 만들며 본학습을 시작하지 않는다.

```bash
cd /home/jack/TurboVLA
bash scripts/robotwin/smoke_train.sh
```

표준 학습 진입점의 55k 실행 명령은 [재현 가이드](REPRODUCE_KO.md)에 있다.
현재 GPU당 batch 6 / accumulation 16이 실제로 검증된 설정이다. batch 12는 아직 측정하지 않았다.
