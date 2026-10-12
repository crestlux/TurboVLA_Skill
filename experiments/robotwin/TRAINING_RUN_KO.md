# RoboTwin Clean50 본학습 및 자동 평가

**현재 상태 변경:** 이 문서의 55k 실행은 2026-09-27 02:25:56 KST에 사용자의
하루 이내 평가 요청으로 174 update에서 중단했다. 로그와 설정은 보존했다.
새 구성과 속도 측정은 [5k 예비 실험 기록](ONE_DAY_PILOT_KO.md)을 참고한다.
아래 내용은 중단된 55k 실행의 당시 기록이다.

2026-09-27 01:53:31 KST에 본학습을 시작했다. 이 문서는 실행 구성 기록이며,
실시간 상태는 아래 확인 명령으로 조회한다. 학습이나 5,000회 평가가 완료되었다는 의미가 아니다.

## 실행 정보

- Run ID: `turbovla_clean50_55k_2x3090_20260927_0156`
- tmux: `turbovla_clean50_55k_20260927`
- 실행 스크립트: `scripts/robotwin/train_then_evaluate.sh`
- 제어 기록 및 로그: `/data/jack/robotwin2_turbovla/pipelines/turbovla_clean50_55k_2x3090_20260927_0156`
- 학습 결과: `/data/jack/robotwin2_turbovla/runs/turbovla_clean50_55k_2x3090_20260927_0156`
- 전체 평가 로그: `/data/jack/robotwin2_turbovla/evaluation/turbovla_clean50_55k_2x3090_20260927_0156_seed0`
- 평가 영상과 simulator 원본 결과: `/data/jack/robotwin2_turbovla/simulator_eval_result`

실행 이름의 마지막 숫자는 식별자이다. 실제 시작 시간은 위 timestamp와 `events.jsonl`을 따른다.

## 학습 설정

| 항목 | 설정 |
|---|---|
| 데이터 | Clean50, 50개 작업, 2,500개 에피소드; SSD |
| 모델 | TurboVLA, DINOv3 ViT-L, action horizon 50 |
| 초기값 | 준비된 GroundingDINO/BERT/DINOv3; 20-step smoke에서 이어 학습하지 않음 |
| GPU | RTX 3090 두 장, GPU 0/1 |
| 배치 | GPU당 6 × GPU 2 × accumulation 16 = 전역 192 |
| 학습량 | optimizer update 55,000회 |
| 학습률 | 5e-5, warmup 1,000 update, 이후 constant |
| EMA | decay 0.999 |
| seed | 42 |
| 저장 | 5,000 update마다 raw/EMA safetensors; HDD |
| 데이터 로더 | rank당 worker 4, prefetch 2, 원본 metadata와 분리한 statistics cache |
| 로그 | 50 update마다 loss; tqdm은 매 optimizer update; W&B 비활성 |

초기 관측은 optimizer update당 약 11초였다. 이 속도를 유지하면 학습은 약 7일이며,
설치 중 CPU/디스크 부하, 다른 GPU 작업, 평가와 저장 시간에 따라 바뀐다.
학습 로그의 `model_time`은 마지막 micro-batch 시간으로, 전체 optimizer update 시간이 아니다.

## 자동 실행 순서

1. 본학습을 55,000 update까지 수행한다.
2. 학습이 exit 0으로 끝난 경우에만 `steps_55000_ema_model.safetensors`를 검사한다.
   저장 step, tensor 878개, 전체 tensor 유한값, SHA256을 확인하고 기록한다.
3. 평가 환경 설치 기록과 RoboTwin commit을 확인한다.
4. GPU가 학습에서 해제된 뒤 `adjust_bottle`을 **2회 실제 평가**한다.
   이 검사는 최대 30분이며 서버 연결, 시뮬레이터, 결과 저장을 확인한다.
   성공률이 0/2여도 실행과 결과 저장이 정상이라면 검사를 통과한다.
5. 짧은 검사를 통과하면 동일 55k EMA로 Clean50 전체를 작업당 100회, 총 5,000회 평가한다.
   seed 0, unseen instruction, min-max 정규화, temporal ensemble 50/alpha 0.1,
   binary threshold 0.49를 사용한다. GPU당 server/simulator 한 쌍이다.
6. 모든 task에 최종 `/100`과 저장 완료 문구가 있어야 `task_results.csv`, `summary.json`을 생성하고
   pipeline 상태를 `completed`로 바꾼다. 두 번의 사전 검사는 5,000회 결과에 포함하지 않는다.

어느 단계에서든 오류가 나면 이후 단계를 실행하지 않고 `status.json`에 `*_failed`를 기록한다.
부분 평가를 최종 성공률로 보고하지 않는다. 자동 재시작은 설정하지 않았다.

**GPU 평가 경로는 본학습 종료 후 검증한다.** 본학습과 동시 SAPIEN 실행 요청은 자동 승인 검토가
추가 GPU 메모리에 따른 OOM/학습 중단 위험을 이유로 거절했다. 따라서 현재 단계의 평가 준비 확인은
설치, CPU import/연산, 에셋 무결성, 경로 및 자동 실행 구성에 한정한다.
실제 Vulkan 렌더링·CuRobo GPU 계획·정책 연결 rollout 통과를 미리 주장하지 않는다.

## 상태 확인

```bash
cd /home/jack/TurboVLA
python3 scripts/robotwin/pipeline_status.py turbovla_clean50_55k_2x3090_20260927_0156
```

위 명령은 stage, 프로세스 생존 여부, 마지막 완료 update, 최근 최대 100 update의 실제 경과시간,
추정 잔여 시간과 예상 종료 시각을 출력한다. 평가가 끝나면 성공률 summary도 출력한다.
프로세스 생존 여부는 동일 호스트의 터미널에서 확인해야 한다. 격리된 PID namespace에서는 보이지 않을 수 있다.

실행 화면:

```bash
tmux attach -t turbovla_clean50_55k_20260927
```

`Ctrl-b`를 누른 다음 `d`로 분리하면 실행은 계속된다.
로그만 볼 때는 아래 명령을 사용한다.

```bash
tail -f /data/jack/robotwin2_turbovla/pipelines/turbovla_clean50_55k_2x3090_20260927_0156/pipeline.log
```

연결 해제에는 지속되지만 재부팅/전원 차단에는 지속되지 않는다. 현재 runtime의 저장은 모델 가중치이며,
optimizer/RNG/누적 EMA 전체 상태를 복원하는 재개 기능은 보장되지 않는다.
중단된 실행을 가중치만 읽어 이어 실행한 결과를 원래의 연속 55k 학습과 동일하게 취급하지 않는다.

## 평가 환경과 재현 범위

정책 서버는 기존 `turbovla-robotwin` 환경을 사용하고, 시뮬레이터는 별도 `robotwin-turbovla` 환경을 사용한다.
02:15 KST에 평가 환경 설치 및 CPU 검증을 완료했다.
[검증 기록](setup_records/evaluation_ready.json)의 `status=ready`는 설치·에셋 준비 완료를 의미하며,
`gpu_rollout_validation=pending_after_training`을 함께 확인해야 한다.
SAPIEN/mplib/Open3D/정책 클라이언트 import, PyTorch3D C++ CPU 연산,
CuRobo 확장 5개 import, 50개 task 소스 문법 및 instruction 파일, `pip check`를 확인했다.
CuRobo `MotionGen`은 import 과정에서 CUDA tensor를 만들므로 상위 planner와 task runtime import는
실제 평가 단계로 미뤘다. 전체 패키지는 [simulator freeze](setup_records/simulator-packages.txt)에 보관했다.
시뮬레이터 소스는 `/home/jack/practice/RoboTwin-TurboVLA`에 있다.
RoboTwin commit `c3ddfa8b97d5519efa828b075999bd0006778e5e`는 현재 TurboVLA 평가 API와의 호환성으로 선택했다.
논문 저자가 사용한 정확한 simulator commit이 공개 자료에서 확인되지 않았으므로 결과 일치를 보장하지 않는다.

평가 의존성은 `scripts/robotwin/requirements-evaluation.txt`,
빌드 제약은 `scripts/robotwin/constraints-evaluation.txt`에 기록한다.
PyTorch3D는 공식 stable commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`(0.7.8)이며,
[공식 설치 문서](https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/INSTALL.md)의
PyTorch 2.4.1 지원 조합을 사용한다. CuRobo는 v0.7.8 (`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`)이다.
SAPIEN/mplib의 두 수정은 RoboTwin 공식 `_install.sh` 내용과 같다.
시뮬레이터 평가 코드에는 평가 횟수 환경 변수, 예외 traceback 기록,
20회 연속 일반 setup 예외 시 task 중단을 추가했다. 원래 코드는 setup 오류를 계속 숨기며 seed를 건너뛸 수 있었다.
정상 episode의 평가 횟수·성공 판정·동작은 바꾸지 않았다.

에셋은 `TianxingChen/RoboTwin2.0` revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`의
`objects.zip`, `embodiments.zip`, `background_texture.zip`이다.
압축 합계 14,928,313,317 bytes(약 14.93 GB / 13.90 GiB)를 HDD에 보관하고
SHA256 검증 및 ZIP CRC 검사를 거쳐 압축 해제까지 완료했다. 압축 해제 후 16,605,181,555 bytes이며,
압축본과 함께 HDD에서 약 30 GiB를 사용한다. 로봇 설정의 경로 6개도 갱신했다.
상세 manifest는 `setup_records/asset-manifest.json`에 있다.
에셋/평가 영상의 HDD 저장을 위한 심볼릭 링크는 학습 데이터의 SSD 위치를 바꾸지 않는다.

이 실행에서 학습 알고리즘은 스모크 통과 상태로 유지했다. 새 변경은 장기 실행 감독,
평가 순서/집계, 상태 조회, 별도 평가 환경 준비에 해당한다.
