# default_sfm seed 42 본학습 실행 기록

사용자의 시작 승인 후 **2026-10-24 23:23:25 KST**에 9개 실험 큐를 시작했다.
실제 batch 64, gradient accumulation 없음, RTX 3090 24GB 세 장에 GPU당 1개다.
초기 3개가 실제 optimizer update를 진행하며 나머지 6개는 각 GPU가 비면 순서대로 시작한다.
이 문서는 시작 시점의 기록이다. 현재 상태는 아래 모니터에서 확인한다.

## 실행 계약과 위치

- 학습 소스 커밋: `8de13e5d0f229faef1f57e687da8151d1438e61e`
- 고정된 실행 worktree: `/home/crestlux/project/TurboVLA-Research/tmp/default_sfm_seed42_source`
- Python: `/home/crestlux/miniforge3/envs/turbovla-sfm/bin/python`
- 결과: `/home/crestlux/project/TurboVLA-Research/outputs/default_sfm_seed42`
- 서비스: `turbovla-sfm-seed42-20261024.service`
- tmux: `sfm9`, 창 이름 `monitor`, `logs`, `gpu`
- 온라인 W&B 그룹: `default_sfm_9way_seed42`
- 각 실험 예산: 11,118,800관측, 173,732 update, 마지막 잔여 배치만 16개

대기 실험까지 같은 소스를 쓰도록 별도 worktree에 실행 코드를 고정했다.
이 디렉터리는 이름이 `tmp`여도 학습·재개에 사용하므로 보존한다. 그 안의 `data`와 `pretrained`는
주 작업 디렉터리의 같은 이름 디렉터리를 가리킨다. 전처리 데이터·정규화 계수·backbone은 기존 파일을 공유한다.
학습 실행을 위해 데이터나 모델 설정을 변경하지 않았다.

정확한 CLI, 설정별 해시와 서비스 속성은 [launch_record.json](launch_record.json),
온라인 metric 수신 증거는 [online_verification.json](online_verification.json),
초기 GPU·gradient·모니터 검증은 [initial_health.json](initial_health.json)에 있다.
실행 중 변하는 상태 파일은 `outputs/default_sfm_seed42/status.json`이다.
준비 기록과 원본 설정의 `full_training_started: false`는 준비 시점의 스냅샷이다.
실제 각 실험의 `run.json`은 `formal_training_started: true`와 실행 커밋을 기록한다.

## GPU별 순서와 첫 온라인 run

| GPU | 첫 실행 | 다음 실행 | 마지막 실행 |
|---|---|---|---|
| 0 | FT-flat / XY 유지 | GRU+AdaLN / 위치 XY 제거 | GRU / 위치 XY·선속도 vx·vy 제거 |
| 1 | GRU / XY 유지 | FT-flat / 위치 XY 제거 | GRU+AdaLN / 위치 XY·선속도 vx·vy 제거 |
| 2 | GRU+AdaLN / XY 유지 | GRU / 위치 XY 제거 | FT-flat / 위치 XY·선속도 vx·vy 제거 |

각 GPU의 앞 실험이 끝나면 그 GPU의 다음 실험을 시작한다. 다른 GPU의 완료를 기다리지 않는다.

- [FT-flat / XY 유지: j3pjxo81](https://forge.coreweave.com/wandb/hemi0213-sungkyunkwan-university/TurboVLA_Skill/runs/j3pjxo81)
- [GRU / XY 유지: u78hp0a1](https://forge.coreweave.com/wandb/hemi0213-sungkyunkwan-university/TurboVLA_Skill/runs/u78hp0a1)
- [GRU+AdaLN / XY 유지: m0bxscw4](https://forge.coreweave.com/wandb/hemi0213-sungkyunkwan-university/TurboVLA_Skill/runs/m0bxscw4)

나머지 run ID는 해당 실험이 시작될 때 생성되어 각 폴더의 `wandb_run.json`에 저장된다.
W&B에는 설정과 수치 metric만 기록하며 원본 데모·이미지·모델 가중치를 업로드하지 않는다.
시작 확인은 최종 동기화 확인이 아니다. 각 실험 종료 후 `wandb_sync.json`의 `verified`를 확인한다.

## 진행 상황 보기

```bash
tmux attach -t sfm9
```

`Ctrl-b` 다음 `d`로 모니터에서 나와도 학습은 계속된다. 서비스가 학습을 유지하며 tmux는 모니터용이다.
사용자 linger를 활성화하여 SSH 로그아웃 후에도 사용자 서비스가 유지되게 했다.
이 transient 서비스는 서버 재부팅 후 자동 재개를 구성한 것은 아니다.

```bash
systemctl --user status turbovla-sfm-seed42-20261024.service
/home/crestlux/miniforge3/envs/turbovla-sfm/bin/python \
  /home/crestlux/project/TurboVLA-Research/tmp/default_sfm_seed42_source/scripts/default_sfm/monitor.py \
  --run-dir /home/crestlux/project/TurboVLA-Research/outputs/default_sfm_seed42 --watch
```

모니터는 step, loss, 관측/초, 진행률, GPU 최대 메모리를 표시한다.
checkpoint는 1,000 step마다, 최종 step 및 정상 중단 때 저장하며 최근 2개를 보존한다.
loss는 학습 오차이며 별도 로봇 평가 성공률은 아직 측정하지 않았다.

## 같은 실행을 재현하는 명령

아래는 이미 실행한 명령의 재현 기록이다. 현재 실행 중인 그룹에 다시 실행하지 않는다.
새 실험을 재현할 때는 새 출력 경로·서비스 이름을 사용한다. 다른 서버에서는 경로와 GPU를 맞춘다.
환경·데이터·backbone 준비는 [학습 안내](../../TRAINING_GUIDE.md)와
[전처리 안내](../../PREPROCESSING_GUIDE.md)를 따른다.

```bash
cd /home/crestlux/project/TurboVLA-Research
git worktree add --detach tmp/default_sfm_seed42_source 8de13e5d0f229faef1f57e687da8151d1438e61e
ln -s /home/crestlux/project/TurboVLA-Research/data tmp/default_sfm_seed42_source/data
ln -s /home/crestlux/project/TurboVLA-Research/pretrained tmp/default_sfm_seed42_source/pretrained
mkdir -p outputs
loginctl enable-linger crestlux

systemd-run --user --unit=turbovla-sfm-seed42-20261024.service \
  --description='TurboVLA SFM nine experiments batch64 seed42' \
  --service-type=exec --remain-after-exit \
  --working-directory=/home/crestlux/project/TurboVLA-Research/tmp/default_sfm_seed42_source \
  --property=Restart=no --property=KillMode=mixed --property=TimeoutStopSec=240 \
  --property=StandardOutput=append:/home/crestlux/project/TurboVLA-Research/outputs/default_sfm_seed42_launcher.log \
  --property=StandardError=inherit \
  --setenv=USE_TF=0 --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 \
  --setenv=TOKENIZERS_PARALLELISM=false --setenv=WANDB_MODE=online \
  --setenv=PYTHONUNBUFFERED=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=OMP_NUM_THREADS=4 --setenv=MKL_NUM_THREADS=4 \
  /home/crestlux/miniforge3/envs/turbovla-sfm/bin/python \
  /home/crestlux/project/TurboVLA-Research/tmp/default_sfm_seed42_source/scripts/default_sfm/run_parallel.py \
  --config-dir /home/crestlux/project/TurboVLA-Research/tmp/default_sfm_seed42_source/experiments/default_sfm/training_configs \
  --output /home/crestlux/project/TurboVLA-Research/outputs/default_sfm_seed42 \
  --gpus 0,1,2 --jobs-per-gpu 1 --microbatch 64 --train
```

launcher가 출력 그룹 디렉터리를 만들기 때문에 정확한 그룹 경로는 미리 만들지 않는다.
위 실행 코드·학습 JSON·전처리 bundle은 보존한다. 나중에 주 작업 디렉터리를 수정해도 이 worktree를 변경하지 않는다.

모니터가 없을 때만 아래 명령으로 다시 연결한다. 학습 프로세스는 재시작하지 않는다.

```bash
tmux new-session -d -s sfm9 -n monitor -x 180 -y 45 \
  '/home/crestlux/miniforge3/envs/turbovla-sfm/bin/python /home/crestlux/project/TurboVLA-Research/tmp/default_sfm_seed42_source/scripts/default_sfm/monitor.py --run-dir /home/crestlux/project/TurboVLA-Research/outputs/default_sfm_seed42 --watch'
tmux new-window -d -t sfm9 -n logs \
  'tail -n 80 -F /home/crestlux/project/TurboVLA-Research/outputs/default_sfm_seed42_launcher.log'
tmux new-window -d -t sfm9 -n gpu 'watch -n 5 nvidia-smi'
```

## 정상 중단 및 재개

중단이 필요할 때만 `systemctl --user stop turbovla-sfm-seed42-20261024.service`를 실행한다.
launcher는 실행 중인 세 worker에 정상 중단을 요청하고 checkpoint와 W&B 종료 기록을 남긴다.
각 worker의 `latest.pt`, `wandb_sync.json`, 서비스 종료를 확인한 후 재개한다.
일반 프로세스 강제 종료는 마지막 checkpoint 이후 update를 잃을 수 있다.

재개는 위 `systemd-run` 명령의 서비스 이름을 새 이름, 예를 들어
`turbovla-sfm-seed42-20261024-resume1.service`로 바꾸고, 마지막 CLI에 **`--resume`을 추가**한다.
소스·설정·출력 경로·batch·GPU 개수는 그대로 유지한다. 이미 실행 중인 서비스가 있으면 재개하지 않는다.
완료된 실험은 건너뛰고, checkpoint가 있는 실험은 그 지점부터, 아직 시작하지 않은 실험은 처음부터 실행한다.
첫 checkpoint 이전에 강제 종료된 실험은 원래 기록을 보존하고 별도 출력에서 다시 시작해야 한다.
재개의 W&B run은 이전 기록을 덮지 않는 새 segment이며 `parent_run_id`로 연결된다.
