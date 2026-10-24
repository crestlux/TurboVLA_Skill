# default_sfm 9개 실험 학습과 재현

이 문서는 전처리가 끝난 450개 데모로 9개 조건을 학습하는 절차다.
데이터 처리 규칙과 이전 실행 증거는 [전처리 안내](PREPROCESSING_GUIDE.md)에 있다.
**준비 과정에서 본학습을 시작하지 않았다.** GPU 검증은 별도 출력의 짧은 smoke optimizer step이며,
그 가중치는 본학습 초기값으로 사용할 수 없다. W&B 연결 검증 run은 optimizer step 0이다.
최종 검증 결과는 [readiness.json](training_validation/readiness.json)에 기록한다.

## 1. 동일하게 유지할 실험 계약

공통 데이터는 `data/default_sfm_preprocessed`가 가리키는
`/data/crestlux/default_sfm_processed/sfm_delta_h5_ftstd001_v1`이다.
450개, 9개 task, 138,985개 유지 관측 전체를 학습하며 내부 validation 분할은 없다.
원본과 완료된 HDF5·정규화 계수는 학습 과정에서 수정하지 않는다.

| 방법 | 위치 XY 유지 | 위치 XY 제거 | 위치 XY와 선속도 vx·vy 제거 |
|---|---|---|---|
| FT300 + state | `ft_state_xy_keep` / 313차원 | `ft_state_xy_remove` / 311차원 | `ft_state_xy_vxy_remove` / 309차원 |
| FT GRU token | `ft_gru_xy_keep` / 13차원 | `ft_gru_xy_remove` / 11차원 | `ft_gru_xy_vxy_remove` / 9차원 |
| FT GRU token + AdaLN | `ft_gru_adaln_xy_keep` / 13차원 | `ft_gru_adaln_xy_remove` / 11차원 | `ft_gru_adaln_xy_vxy_remove` / 9차원 |

모든 방법에서 기존 state의 순간 F/T 6축을 제외한다. FT300은 정규화된 `[50,6]`을
시간 우선 순서로 펼친 값이다. GRU 방법은 같은 `[B,50,6]`을 별도로 받는다.
50개는 사용자 제공 500Hz 기준 약 100ms 이력이며, 센서별 timestamp가 없어 주기를 재검증하거나 보간하지 않았다.
F/T는 전체 유지 프레임에서 계산한 축별 평균과 `max(std,0.01)`로 **한 번만** 정규화한다.
정상인 0 이력도 통계에 포함했다. 정규화된 0 이력이 숫자 0을 유지해야 한다는 뜻은 아니다.

기존 TurboVLA USB의 GRU는 `GRU(6,128)`, 1층, 단방향, 각 window마다 zero hidden으로 시작한다.
마지막 hidden을 `Linear(128,256) → LayerNorm`으로 바꿔 ACT memory에 token 하나를 추가한다.
GRU와 projection은 기존 USB처럼 FP32로 계산한다. 추가적인 F/T 정규화·필터·마스킹은 없다.
AdaLN 조건도 이 token을 유지하고 같은 256차원 벡터로 ACT 3개 decoder 층의 LayerNorm 3개씩을 조절한다.
각 정규화 출력에 `LN(x) * (1 + scale) + shift`를 적용하며 scale·shift의 Linear weight/bias는 0으로 초기화한다.
새 residual gate는 추가하지 않는다.

모델 입력 이미지는 head/wrist 순서의 `[B,2,3,224,224]`다. crop과 letterbox는 HDF5에 이미 적용됐다.
학습 때 다시 crop/resize하지 않는다. ImageNet mean/std는 모델 직전에 적용한다.
출력은 `[B,5,7]`, 즉 nominal 10Hz 기준 0.5초의 5개 action이다.
각 action의 Delta 기준은 **그 action 시점의 관측 자세**다. chunk 시작 자세로 일괄 차감하지 않는다.
마지막 불완전 chunk는 저장된 boolean mask로 loss에서 제외한다.
이 chunk 길이는 실제 로봇에서 한 번에 실행할 action 수를 아직 정한 것이 아니다.

## 2. 학습 설정과 결정 근거

| 설정 | 확정 값과 근거 |
|---|---|
| 초기 가중치 | 사용자가 기존 USB 설정 사용 승인. DINOv3·BERT는 사전학습 가중치, 나머지 새 초기화 |
| 학습 범위 | DINOv3 backbone 학습, BERT backbone 동결/eval, text projection·상호작용·action·FT 모듈 학습 |
| optimizer | AdamW, policy LR `5e-5`, vision backbone LR `1e-5`, betas `(0.9,0.95)`, weight decay `1e-6` |
| 계산·loss | BF16 AMP, 유효 action 원소에 대한 masked L1, gradient norm clip `1.0` |
| EMA | decay `0.999`, 기준 batch 64, 실제 관측 수에 따라 `decay**(n/64)` |
| seed | 42. 본학습은 각 조건 1 seed이며 다중 seed 통계 실험은 아님 |
| 총 학습량 | 실험당 11,118,800개 관측 = 138,985 × 80 |
| 샘플링 | 사용자 확정: task 9개를 동일 확률로, 선택 task 안에서는 프레임을 균등하게 복원 추출 |
| 배치 | 기존 USB처럼 한 번에 64개. gradient accumulation 없이 update |
| 증강 | 기존 USB color jitter: 밝기·대비·채도 0.2, hue 0.02. 같은 관측의 두 view에 같은 변환 |
| padding | jitter는 168×224 영상 영역만 처리. 위·아래 28픽셀은 ImageNet 정규화 전 검정 유지 |
| AdaLN | 사용자 확정: scale·shift 출력 0 초기화 |
| 온라인 기록 | 사용자 승인: 현재 계정의 `TurboVLA_Skill` 프로젝트. raw demo·이미지·가중치 업로드 없음 |

AdamW epsilon `1e-8`, clip norm `1.0`, warmup 8,000관측, cosine 최저 LR 비율 `0.01`,
10 step마다 metric 기록, 1,000 step마다 checkpoint, 최근 2개 보관은 선택한 기존 USB 구현에서 가져온 세부사항이다.
sample 수를 기준으로 LR을 계산한다. 마지막 배치는 16개이며 총 optimizer step은 **173,732**다.
"80회 분량"은 학습 관측 수의 예산이다. 복원 추출이므로 모든 프레임이 정확히 80번 선택되지는 않는다.
task 안에서 길이가 긴 demo는 프레임 수만큼 더 자주 선택된다. demo/배치 p1~p5를 별도로 균등화하지 않았다.

9개 조건은 같은 sampler seed·순서·예산을 쓴다. prefetch는 완료 관측 수를 앞당기지 않으며
optimizer update가 완료된 뒤에만 sampler cursor를 진행한다.
공통 가중치가 입력 차원이나 AdaLN 추가 여부 때문에 달라지지 않도록 state projection, ACT decoder,
FT GRU 초기화 난수 흐름을 각각 분리했다. 공통 가중치의 초기 해시와 6개 GRU 해시는 실제 모델로 확인했다.
입력 차원이 다른 state projection 자체는 같은 shape의 공통 가중치 비교 대상이 아니다.
사용자는 기존 USB와 같은 실제 batch 64를 최종 선택했다. 9개 조건 모두 microbatch 64, 누적 1회다.
작은 microbatch의 누적 기능은 이전 용량 측정용으로 남아 있지만 본학습에서는 허용하지 않는다.

## 3. 환경과 실행 전 확인

저장소·전처리 출력 전체·사전학습 backbone을 함께 전달한다. Git에는 HDF5와 backbone 가중치가 없다.
환경 재생성 및 데이터 전송은 전처리 안내를 따른다.
현재 검증 환경은 Python 3.10.22, torch 2.6.0+cu124, torchvision 0.21.0, Transformers 4.57.6이다.
별도 LIBERO/RoboTwin 의존성을 설치할 필요가 없다. 설정의 `compatibility_profile: libero`는 기존 USB가
사용하던 모델 기본 구성 이름이며 LIBERO dataset이나 simulator를 사용하는 의미가 아니다.

```bash
cd /home/crestlux/project/TurboVLA-Research
source scripts/default_sfm/activate_env.sh
git rev-parse HEAD
python scripts/default_sfm/status.py
nvidia-smi
python scripts/default_sfm/test_training.py -v
```

실행 가능한 설정은 [training_configs](training_configs/index.json)의 9개 JSON이다.
`experiment_plan.yaml`은 결정 기록이다. `prepare_training_configs.py`는 보존된 USB 설정을 기준으로
완료된 SFM manifest의 해시와 입력 차원을 읽어 JSON을 생성하며 학습을 실행하지 않는다.
새로 전처리를 재현한 출력은 코드/plan provenance 때문에 manifest·contract 해시가 달라질 수 있다.
그 출력으로 새 실험을 할 때는 symlink와 학습 설정을 함께 새로 준비한다.
기존 checkpoint 재개에는 원래 데이터 bundle을 그대로 사용한다.

```bash
# 데이터 root와 backbone symlink 준비 후 설정 생성. 최종 권장 microbatch는 아래 GPU 표 참조.
python scripts/default_sfm/prepare_training_configs.py --microbatch 64

# 짧은 검증만 실행. 이 명령은 본학습과 W&B run을 만들지 않는다.
python scripts/default_sfm/run_parallel.py \
  --output tmp/my_sfm_smoke --gpus 0,1,2 --jobs-per-gpu 1 --microbatch 64 \
  --smoke-steps 3 --profile-warmup 1 --queue-smoke
```

선택 GPU에 기존 compute process가 있으면 병렬 launcher는 중단한다. 다른 학습을 종료하거나 GPU를 강제로 비우지 않는다.
smoke와 본학습은 CLI에서 `--smoke-steps`/`--train` 중 하나를 반드시 명시해야 한다.
출력 경로가 이미 있으면 덮어쓰지 않는다.

## 4. GPU 병렬 측정

동일한 RTX 3090 24GB 세 장에서 측정했다. 모든 경우 유효 배치 64다.
warmup 3 step 후 각 프로세스가 barrier에 모인 시점부터 동시 측정하며,
합계 처리량은 첫 측정 update 시작부터 마지막 update 완료까지의 전체 관측 수/벽시계 시간이다.
프로세스 초기화 시간은 제외하고 update 사이 데이터 대기는 포함한다.
메모리는 2초 간격 `nvidia-smi`의 **GPU 전체 사용량 최대**이며 torch allocator만의 값이 아니다.

| GPU당 실험 / 전체 | microbatch / 누적 | 측정 step / 실험 | 합계 관측/초 | 최대 GPU 사용량 |
|---|---|---|---|---|
| 3 / 9 | 8 / 8 | 12 | 258.5 | 15.68GiB |
| 3 / 9 | 16 / 4 | 30 | 298.6 | 20.48GiB |
| 2 / 6 | 32 / 2 | 30 | 322.2 | 20.10GiB |
| 1 / 3 | 64 / 1 | 50 | 303.9 | 15.94GiB |

사용자가 최종 선택한 구성은 **GPU당 1개, 총 3개 동시 실행, batch 64, gradient accumulation 없음**이다.
위 표의 작은 microbatch 측정은 대안을 비교한 기록이며 본학습에 적용하지 않는다.
24GB GPU에서 batch 64 모델 하나가 약 15.9GiB를 사용하므로 GPU당 두 모델은 수용할 수 없을 것으로 예상한다.
launcher는 각 GPU의 실험이 끝날 때 그 GPU에 대기 실험을 올려 총 9개를 이어서 처리한다.
세 GPU의 실험이 모두 끝날 때까지 묶어서 기다리는 방식은 아니다.
수치는 장시간 안정성·학습 수렴·로봇 성공률 검증 결과가 아니다.
W&B 전송과 주기적 checkpoint 저장을 제외한 smoke 처리량이므로 전체 소요 시간을 보장하지 않는다.
선택한 3개 동시 실행 구성으로 9개 전체 예산을 나눈 단순 환산은 약 92시간이며 실제 완료 시간에는 위 비용과 시스템 부하가 추가된다.
원자료 요약은 [gpu_profiles.json](training_validation/gpu_profiles.json)에 보존한다.

## 5. 본학습 시작과 모니터링

**아래 `--train` 명령은 본학습을 시작한다. 준비 검증 중에는 실행하지 않았다.**
현재 GPU 번호 0,1,2를 시작 시 UUID로 고정해 각 프로세스에 전달한다.
세 GPU를 한 모델의 DDP에 묶는 방식이 아니라 서로 독립적인 9개 실험을 3개씩 실행하는 방식이다.

```bash
# 활성화된 turbovla-sfm 환경에서 실행한다. tmux는 이 환경을 물려받는다.
tmux new-session -d -s sfm9 -c "$PWD" \
  'python scripts/default_sfm/run_parallel.py --output outputs/default_sfm_seed42 --gpus 0,1,2 --jobs-per-gpu 1 --microbatch 64 --train'
tmux split-window -h -t sfm9 -c "$PWD" \
  'python scripts/default_sfm/monitor.py --run-dir outputs/default_sfm_seed42 --watch'
tmux attach -t sfm9
```

monitor는 이미 실행 중인 실험의 상태를 읽는다. 별도 터미널에서 아래 명령을 실행해도 학습을 재시작하지 않는다.
step·loss·속도·진행률이 실제로 증가하는지와 각 실험의 `wandb_run.json` URL을 확인한다.
task별 성공률을 측정하지 않았으므로 training loss를 그 성공률로 해석하지 않는다.

```bash
python scripts/default_sfm/monitor.py --run-dir outputs/default_sfm_seed42 --watch
```

각 실험 폴더에는 `config.json`, `run.json`, `data_signature.json`, `initialization.json`,
`gradient_validation.json`, `status.json`, `metrics.jsonl`, `source_snapshot/`, `preprocessing/`가 생긴다.
본학습은 `wandb_run.json`과 종료 후 `wandb_sync.json`도 만든다.
W&B online 초기화/동기화가 실패하면 실패를 보고하며 offline으로 자동 전환하지 않는다.
재개할 때는 이전 W&B 기록을 덮지 않는 새 run segment를 만들고 `resume_parent_run_id`로 연결한다.
global optimizer step과 관측 수는 checkpoint에서 이어진다.

그룹은 한 실험이 실패하면 같은 launcher가 만든 나머지 프로세스에도 정상 중단을 요청하고,
대기 중인 실험을 새로 시작하지 않는다. SIGINT/SIGTERM은 현재 update 후 checkpoint를 저장한다.
GPU의 다른 사용자 프로세스는 종료하지 않는다.

## 6. 체크포인트, 재개와 저장된 관측 검증

checkpoint에는 raw 가중치, EMA, AdamW, sampler cursor, Python/NumPy/torch/CUDA RNG,
설정, 데이터/소스 해시, 이전 W&B run ID를 저장한다. `latest.pt`는 최신 checkpoint의 hard link다.
최근 2개 checkpoint를 보존하고 마지막 step과 정상 중단 때도 저장한다.
GRU+AdaLN 검증 checkpoint 하나는 약 2.60GB다. 9개에서 2개씩 보관하면 약 47GB이고,
동시 저장 시의 임시 파일·로그에 추가 공간이 필요하다.

같은 그룹의 재개는 기존 출력 경로를 그대로 지정한다. 완료된 실험은 건너뛰고,
아직 시작하지 않은 대기 실험은 새로 시작하며, 중단된 실험은 각 `latest.pt`에서 이어간다.

```bash
python scripts/default_sfm/run_parallel.py \
  --output outputs/default_sfm_seed42 --gpus 0,1,2 --jobs-per-gpu 1 --microbatch 64 \
  --train --resume
```

개별 재개도 가능하다. GPU를 옮길 수 있지만 설정·microbatch·전처리 bundle·학습 소스는 같아야 한다.
해시가 다르면 재개를 거부한다. 첫 checkpoint 전에 강제 종료되어 `latest.pt`가 없는 폴더는
원래 기록을 보존하고 새 출력 폴더로 처음부터 시작해야 한다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/default_sfm/train.py \
  --config experiments/default_sfm/training_configs/ft_gru_adaln_xy_keep.json \
  --output outputs/default_sfm_seed42/ft_gru_adaln_xy_keep \
  --train --resume outputs/default_sfm_seed42/ft_gru_adaln_xy_keep/latest.pt

# 저장된 실제 관측으로 raw와 EMA를 모두 다시 로드한다. 로봇 명령을 보내지 않는다.
CUDA_VISIBLE_DEVICES=0 python scripts/default_sfm/check_checkpoint.py \
  --checkpoint outputs/default_sfm_seed42/ft_gru_adaln_xy_keep/latest.pt \
  --output tmp/my_checkpoint_check.json
```

이 검사는 checkpoint에 묶인 설정과 정규화 계수를 사용하고, 두 관측에 대한 출력 `[2,5,7]`,
finite 값, 반복 추론의 일치, action pose 6축 역정규화를 확인한다. 추론 시 jitter는 끈다.
실제 로봇의 실행 horizon·gripper 후처리·성공 조건·안전한 클라이언트 연결은 별도의 평가/배포 작업이다.

재개 검증의 재현은 동일한 설정으로 (A) 4-step smoke와 (B) 2-step smoke + 2-step resume를 실행하고,
각각 `--save-smoke-checkpoint`를 사용한 뒤 아래처럼 비교한다.
smoke checkpoint로 `--train`을 재개하는 것은 코드에서 거부한다.

```bash
python scripts/default_sfm/compare_checkpoints.py \
  --uninterrupted tmp/default_sfm_resume_b64_v2/uninterrupted/latest.pt \
  --resumed tmp/default_sfm_resume_b64_v2/split/latest.pt \
  --output tmp/my_resume_comparison.json
```

최종 [재개 검증](training_validation/resume_validation.json)과
[checkpoint 재로딩 검증](training_validation/checkpoint_reload.json)은 학습 경로의 작동 검증이다.
batch 64의 raw·EMA·AdamW·sampler·RNG 복원에서 검사한 tensor 2,892개가 정확히 일치했고,
저장된 두 관측에 대한 raw/EMA 반복 추론도 일치했다.
다만 CUDA 학습을 이어간 결과는 비트 단위로 같지 않았다. 4-step 연속과 2+2-step 재개의
raw 가중치 최대 차이는 약 `1.83e-6`이고, 별개의 4-step 연속 실행끼리도 약 `1.78e-6` 차이가 있었다.
각각 4번째 loss는 `0.39555678`, `0.39554024`, 독립 반복은 `0.39554691`이었다.
엄격한 bitwise 비교 결과는 실패 기록 그대로 보존했다. 저장 상태의 복원 검증과 이후 CUDA 연산의
bitwise 재현을 혼동하지 않으며, 기존 USB 계산 설정을 결정적 kernel로 임의 변경하지 않았다.
본학습 성능, 일반화, task 성공률에 대한 결과는 아직 없다.
