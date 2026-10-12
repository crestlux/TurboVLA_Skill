# 하루 이내 첫 평가를 위한 5k 예비 실험

2026-09-27, 사용자가 긴 실행 대신 하루 안에 첫 평가 결과를 요청하여 구성을 변경했다.
기존 55k 실행은 02:25:56 KST에 174 update에서 정상 인터럽트(exit 130)로 중단했다.
로그와 설정은 기존 디렉터리에 남겼으며, 상태는 `superseded_for_one_day_pilot`이다.
첫 저장 지점 5,000 update 이전이어서 기존 실행의 가중치는 저장되지 않았다.

새 실행은 **2026-09-27 02:36:44 KST**에 시작했다.
Run ID는 `turbovla_clean50_pilot5k_2x3090_20260927_0236`,
tmux 세션은 `turbovla_pilot5k_20260927`이다.
**완료: 2026-09-27 19:05:03 KST.** 5,000 update 학습과 Clean50 × 10회 평가가 모두 끝났다.
총 500회 중 140회 성공하여 평균 성공률은 **28.0%**다.
2026-09-27에 50개 작업의 로그·시뮬레이터 결과 파일·CSV·집계 JSON을 대조했다.
세부 결과와 검증 범위는 [5k 평가 보고서](EVALUATION_PILOT5K_KO.md)에 정리했다.

추가 요청으로 [전처리 최적화와 측정](FURTHER_OPTIMIZATION_KO.md)을 수행했다.
실행 중인 학습은 재시작하지 않았으며, 소스 변경의 적용 시점과 전체 속도 측정 한계는 해당 기록에 명시했다.

## 시간과 재현 범위

기존 실측 약 11초/update × 55,000 = 약 168시간이다. 당시 166시간 표시는 진행 상황에 따른
잔여 시간 추정치이며, 이번 실측에서도 약 7일이라는 계산은 맞았다.
[논문 §5.1–5.2](https://arxiv.org/html/2607.27205v1#S5)는 RTX 4090 네 장,
RoboTwin 55k update / warmup 1k / 전역 배치 192 및 작업당 100회 평가를 사용한다.
여기서는 RTX 3090 두 장이다.

5k는 논문 학습량의 약 9.1%다. 55k의 성능을 재현했다고 해석할 수 없으며,
짧게 학습해 실제 성공률과 실패 양상을 먼저 확인하는 실험이다.
작업당 10회 결과는 성공 1회의 차이가 10%p이므로 작업별 수치의 변동이 크다.
학습 loss 감소만으로 로봇 작업 성공률을 예측할 수 없다.

## 최적화 실측

모델, 학습 대상, 영상 크기, action horizon, 전역 배치 192를 유지했다.
BF16 autocast와 FlashAttention은 기존 실행에서도 이미 사용 중이었다.

| 설정 | GPU당 배치 × 누적 | update당 시간 | GPU 최대 allocated / reserved |
|---|---:|---:|---:|
| 기존 ZeRO-2 | 6 × 16 | 약 11.0초 | 기존 NVIDIA 표시 약 13.4 GB |
| ZeRO-2 | 12 × 8 | 7.44초 | 약 15.6 / 16.8 GiB |
| **선택: ZeRO-1** | **12 × 8** | **4.97초** | **약 16.2 / 17.4 GiB** |
| ZeRO-1 | 16 × 6 | 4.84초 | 약 19.0 / 20.2 GiB |

후보 측정은 처음 2 update를 제외했다. 마지막 후보에서는 배치 평가가 포함된 update도 제외했다.
배치 16의 추가 개선은 약 3%였고 메모리가 약 3 GiB 더 필요하여 배치 12를 선택했다.
선택한 설정은 기존보다 약 2.2배 빠르며, 5k 학습 자체는 약 6.9시간으로 추정한다.
저장·초기화·평가 시간과 장시간 속도 변화는 별도다.
동일 최적화를 적용해도 55k 전체 학습은 약 76시간이므로, 하루 이내 첫 평가라는 목표에는
학습량 축소가 함께 필요하다. 처음 50개 작업을 모두 평가한 뒤에도 논문 수치와의 직접 비교에는
원래 학습량과 작업당 100회 평가가 필요하다.

두 GPU는 NVLink 없이 서로 다른 NUMA 노드를 거친다. 설치된 DeepSpeed 0.16.9의 ZeRO-2는
누적 중에도 gradient를 분할하기 위해 통신하며, ZeRO-1은 optimizer 상태를 분할하고
누적 경계에서 gradient를 모은다. 같은 배치 12에서 stage만 바꿔 7.44→4.97초가 된 것은
이 머신에서 통신 비용이 컸다는 근거다. 작은 배치의 반복 호출 비용도 줄었다.

데이터는 이미 SSD에 있다. 이번 측정의 준비 완료 이후 data wait는 update당 약 0.005초로,
현재 RoboTwin 학습에서 데이터 대기가 주된 병목이라는 증거는 없다.
이는 별도 LIBERO/HDD 측정과 구분해야 한다.

모든 후보에서 두 rank의 optimizer·scheduler·EMA 갱신 횟수, 유한 loss 및 실제 parameter 변화를 검사했다.
ZeRO-2/1의 동일 배치 12, 동일 seed, 8 update 비교에서 평균 micro-batch loss 차이는 최대 약 1.4e-5였다.
배치 16에서는 추가로 배치 평가 두 번과 raw/EMA 저장·최종 저장을 통과했다.
이는 실행 정상성 검사이며 최종 정확도 동등성을 입증하는 실험은 아니다.
[원시 측정 기록](setup_records/optimization_20260927.json).

## 실제 GPU 평가 경로 검증

재학습 전에 최적화 시험의 12-step EMA로 `adjust_bottle`, `beat_block_hammer`를
각각 2회, 두 GPU에서 동시에 실행했다. 02:31:14–02:35:36 KST, 총 262초에 exit 0으로 끝났고
최종 결과, 320×240·400프레임 영상 4개, 비어 있지 않은 RGB 프레임을 확인했다.
체크포인트 878개 tensor의 유한값과 50개 task runtime import도 통과했다.
[GPU 검증 기록](setup_records/gpu_rollout_preflight_20260927.json).

12-step EMA의 성공률은 0/4였다. 이것은 평가 경로 검사용 미학습에 가까운 모델이며,
5k 결과로 보고하지 않는다. 전체 50개 작업의 rollout을 미리 완료한 것도 아니다.
기존의 “GPU 평가 검증 보류” 상태는 이번 검사로 해소했다.

## 실행 설정 및 코드 변경

- 실행 스크립트: `scripts/robotwin/train_one_day_pilot.sh`.
- 학습: 5,000 update, GPU당 12 × GPU 2 × 누적 8 = 전역 192, ZeRO-1.
- LR 5e-5, warmup 1,000, EMA 0.999, seed 42; 준비된 backbone에서 새로 시작한다.
- 1,000 update마다 raw/EMA 가중치를 HDD에 저장한다.
- 학습 정상 종료 후 5k EMA tensor 유한값·저장 step·SHA256을 검사한다.
- 실제 시뮬레이터 사전 검사 2회 후 Clean50 × 10회 = 500회 평가하고 완결된 결과만 집계한다.
- 평가 seed 0, unseen instruction, min-max, temporal ensemble 50 / alpha 0.1.
- 원래의 50 × 100 전체 평가와 55k 학습은 이번 실행에 자동으로 추가하지 않는다.

`train.sh`에 런처 설정과 시험 entrypoint 선택 변수를 추가했다. 새 ZeRO-1 설정은 별도 파일이므로
기존 ZeRO-2 설정을 보존한다. `train_then_evaluate.sh`의 학습 길이·배치·평가 횟수와
체크포인트 검사·상태 기록을 매개변수화했다. `pipeline_status.py`는 실제 목표 step을 읽고,
평가 중 부분 진행과 최종 결과를 구분한다. 모델 구조와 loss 코드는 이번 최적화에서 바꾸지 않았다.

실행 식별자와 tmux 이름은 [active-training.json](setup_records/active-training.json)에 기록했다.
실시간 상태와 화면은 다음 명령으로 확인한다.

```bash
cd /home/jack/TurboVLA
python3 scripts/robotwin/pipeline_status.py turbovla_clean50_pilot5k_2x3090_20260927_0236
tmux attach -t turbovla_pilot5k_20260927
```

`Ctrl-b`, `d`로 tmux에서 분리해도 학습은 계속된다. 터미널 연결 해제에는 지속되지만
전원 차단이나 재부팅에는 지속되지 않는다.

제어 로그는 `/data/jack/robotwin2_turbovla/pipelines/turbovla_clean50_pilot5k_2x3090_20260927_0236`,
학습 결과는 `/data/jack/robotwin2_turbovla/runs/turbovla_clean50_pilot5k_2x3090_20260927_0236`,
평가 결과는 `/data/jack/robotwin2_turbovla/evaluation/turbovla_clean50_pilot5k_2x3090_20260927_0236_seed0`에 기록한다.
평가 완료 시 `summary.json`과 `task_results.csv`가 생성된다.
평가 중에는 상태 명령의 `partial_evaluation`에서 완료된 작업·시도 수를 확인할 수 있다.

새 실행 명령은 다음과 같다. 이미 실행 중일 때 중복 실행하지 않는다.

```bash
cd /home/jack/TurboVLA
bash scripts/robotwin/train_one_day_pilot.sh
```

현재 저장은 모델 가중치다. optimizer/RNG 전체를 복원하는 정확한 중단 재개는 보장하지 않는다.
5k 이후 추가 학습 여부는 예비 평가를 보고 결정하며, 가중치만 읽는 후속 실행을
연속 55k 학습과 동일한 결과로 취급하지 않는다.
