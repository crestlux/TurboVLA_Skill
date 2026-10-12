**TurboVLA · RoboTwin 2.0 clean50 재현 실행 가이드**

**2026-09-28 갱신:** 아래는 기존 v1 Clean 55k 학습·환경 구성의 기록이다.
공식 배포는 Clean+Randomized 150k EMA로 변경됐고, 공개 모델 평가에서 DINOv3 특징의
Transformers 버전 호환성 문제를 확인하여 수정했다. **공개 모델 평가를 실행할 때는
[최신 호환성 준비·검증 가이드](PUBLIC_MODEL_COMPATIBILITY_KO.md)의 명령을 사용한다.**
그 문서에 기존 55k와 새 150k의 가중치·정규화 통계·검증 결과를 따로 기록했다.

확인일: 2026-09-27. 대상: `/home/jack/TurboVLA`, Git `d181f64` + 로컬 보정.
20 update의 2 GPU 학습 스모크와 EMA 체크포인트 재로딩·GPU 추론을 통과했다.
2026-09-27 01:53 KST에 시작한 55k 실행은 하루 이내 첫 평가 요청으로 02:25 KST에 중단했다.
속도를 최적화한 5k 학습·500회 예비 평가는 19:05 KST에 완료됐고, 성공률은 28.0%였다.
22:36 KST에는 저자가 공개한 55k EMA 모델로 같은 50개 작업 × 10회 비교 평가를 시작했다.
[공개 모델 평가·진행 확인](RELEASED_MODEL_EVAL_KO.md), [5k 평가 결과](EVALUATION_PILOT5K_KO.md),
[5k 실행 구성](ONE_DAY_PILOT_KO.md), [기존 55k 기록](TRAINING_RUN_KO.md), [스모크 결과](SMOKE_TEST_KO.md)를 참고한다.
아래 새 학습·평가 명령은 현재 실행 중인 평가와 중복 실행하지 않는다.

**데이터는 2026-09-23에 준비 완료했다.** `/home/jack/datasets/RoboTwin/Clean`의
10,404개 파일 전체 해시, 2,500개 시연의 Parquet, 7,500개 영상 전체 디코딩,
실제 PyAV 읽기 검사를 통과했다. [데이터 준비 완료 보고서](DATA_READY_KO.md).

**정책 환경과 ViT-L도 준비 완료했다.** `turbovla-robotwin`에서 의존성 검사, GPU 인식,
샘플 데이터 읽기, CPU 모델 로딩을 확인했다. 2026-09-27에는 누적 학습 패치를 적용하고 시험 학습도 완료했다.
[환경 준비·변경 사항 보고서](ENVIRONMENT_READY_KO.md).

권장 순서는 **공개 모델 로딩 → 소규모 평가 → 공개 모델 전체 평가 → 시험 학습 → 55k 재학습 → 같은 조건으로 전체 평가**이다.
공개 모델 평가가 먼저 통과해야 이후 성공률 차이를 학습 문제와 환경 문제로 구분할 수 있다.

**1. 재현할 결과와 고정할 조건**

목표는 RoboTwin 2.0 **clean50 평균 성공률 60.2%**이다. 하나의 정책을 50개 작업에 공동 학습하고,
동일한 step-55k EMA 체크포인트로 각 작업을 100회씩 평가한다. 총 5,000회이다.
논문의 RoboTwin 모델은 DINOv3 ViT-L, 약 0.4B 파라미터이며, RTX 4090에서 보고한 추론 지연은 43.4 ms이다.
논문 제목의 32 Hz / 0.9 GB 수치는 LIBERO용 ViT-B 모델에 해당하므로 RoboTwin의 비교 기준으로 쓰지 않는다.
[논문 §5.1–5.3 및 Table 2](https://arxiv.org/html/2607.27205v1#S5).

| 항목 | 사용할 조건 |
| --- | --- |
| 학습 데이터 | `StarVLA/RoboTwin-Clean`, LeRobot v2.1, 50개 작업 × 50개 시연 |
| 평가 환경 | `demo_clean`, 50개 작업, 작업당 100회 |
| 입력 | head / left wrist / right wrist RGB, 각 224×224, 14차원 state |
| backbone | DINOv3 ViT-L/16 + BERT base uncased |
| 초기화 | DINOv3/BERT 사전학습 + GroundingDINO의 BERT/projection/interaction 가중치 |
| 학습 대상 | vision·projection·interaction·action head, BERT는 freeze |
| action | 14차원 absolute joint position, horizon 50, ACT head, L1 loss |
| 최적화 | AdamW, LR `5e-5`, betas `(0.9, 0.95)`, weight decay `1e-10` |
| 학습 길이 | 55,000 **optimizer update**, warmup 1,000 update, 이후 일정 LR |
| batch / EMA | global batch 192, EMA decay 0.999 |
| 평가 설정 | `instruction_type: unseen`, `min_max`, `unnorm_key: new_embodiment` |
| action 실행 | temporal ensemble 켬, horizon 50, alpha 0.1; 매 환경 step마다 새 chunk 조회 |
| seed | 현재 공개 코드의 학습 seed 42, 평가 seed 0을 명시하여 기록 |

세부 optimizer·ensemble 조건은 이 저장소의 `configs/clean50.yaml` 및 `evaluation/deploy_policy.yml` 기준이다.
공개 통계에는 2,500 trajectories와 549,787 transitions가 기록되어 있다.
[기존 55k normalization 통계](https://huggingface.co/H-EmbodVis/TurboVLA/blob/cb5300544693013164c4bb251a13036002a55c81/dataset_statistics.json).

**2. 이 머신의 준비 상태와 선행 작업**

| 확인한 항목 | 상태 / 해야 할 일 |
| --- | --- |
| GPU | RTX 3090 24GB × 2. 논문의 RTX 4090 × 4와 다르므로 batch 분할 조정 필요 |
| BERT | `pretrained/backbones/bert-base-uncased`에 있음 |
| GroundingDINO | `pretrained/backbones/groundingdino/groundingdino_swint_ogc.pth`에 있음 |
| DINOv3 | `pretrained/backbones/dinov3-vitl16`에 ViT-L 다운로드·전체 해시 검증 완료 |
| RoboTwin 데이터 | `/home/jack/datasets/RoboTwin/Clean`에 다운로드·검증 완료 |
| RoboTwin checkpoint / simulator | 공개 55k EMA 다운로드·공식 해시 확인·변환·strict loading·현재 공식 core model과 추론 출력 비교 완료. 같은 simulator에서 공개 모델 비교 평가 진행 중 |
| conda 환경 | 정책용 `turbovla-robotwin`, 별도 평가용 `robotwin-turbovla`; 최신 설치·검증 범위는 본학습 기록 참조 |
| CUDA compiler | 정책 환경의 `nvcc`는 12.4.131. 환경 활성화 후 `CUDA_HOME=$CONDA_PREFIX` 사용 |
| 저장 공간 | 환경 설치 후 `/home/jack` NVMe 약 196 GiB 여유. Clean 데이터와 ViT-L은 SSD, 누적 checkpoint·평가 영상은 HDD 사용 계획 |

현재 README 명령을 그대로 실행하기 전에 다음 차이를 처리해야 한다.

- `train.sh` 기본값은 4 GPU / GPU당 48 / 100k step이다. 이 머신에서는 명시적으로 재설정한다.
- Hugging Face 데이터는 `<task>/...` 구조인데 로더는 `Clean/<task>/...`를 찾는다. 아래에서는 다운로드 위치 자체를 `Clean/`로 맞춘다.
- 공개 checkpoint의 설정은 `GroundingDINODiT`, 현재 runtime 등록 이름은 `TurboVLA`이다. tensor 이름도 다르다. 제공한 변환 도구로 별도 평가 bundle을 만든다.
- 최신 RoboTwin은 XPolicyLab 평가 체계로 이동했다. 현재 TurboVLA는 `script/eval_policy.py`를 호출하므로 호환되는 과거 commit을 사용한다.
- 과거 RoboTwin 평가 코드는 `test_num = 100`이 고정되어 있다. `ROBOTWIN_TEST_NUM=2`만 설정해도 2회로 줄어들지 않는다. 아래의 한 줄 보정을 적용한다.
- 2 GPU의 gradient accumulation을 위한 scheduler·평가·저장 시점 보정은 현재 적용되어 있다. 새 checkout에서는 아래 패치를 적용한다.
- 생성되는 통계 캐시는 데이터 루트의 `.cache/turbovla_statistics`로 분리해 배포 메타파일을 보존한다.

데이터 폴더 구조는 [배포 데이터 목록](https://huggingface.co/datasets/StarVLA/RoboTwin-Clean/tree/main),
checkpoint 형식은 [배포 config](https://huggingface.co/H-EmbodVis/TurboVLA/blob/main/config.yaml),
평가 API 변경은 [RoboTwin 공식 저장소](https://github.com/RoboTwin-Platform/RoboTwin)에서 확인했다.

**3. 경로와 정책 환경 준비**

아래 블록은 bash에서 순서대로 실행한다. 새 터미널에서는 필요한 `export`를 다시 실행한다.
현재 머신의 정책 환경 설치는 완료했다. 아래 설치 블록은 재구성 시 참고용이다.
일상 사용은 `source /home/jack/TurboVLA/scripts/robotwin/activate_env.sh`만 실행하면 된다.
이 활성화 스크립트는 학습이나 평가를 실행하지 않는다.

```bash
source /home/jack/miniforge3/etc/profile.d/conda.sh
cd /home/jack/TurboVLA

export TURBOVLA_ROOT=/home/jack/TurboVLA
export ROBOTWIN_STORE=/data/jack/robotwin2_turbovla
export ROBOTWIN_PATH=/home/jack/practice/RoboTwin-TurboVLA
export ROBOTWIN_DATA_ROOT=/home/jack/datasets/RoboTwin
export BERT_MODEL_PATH="$TURBOVLA_ROOT/pretrained/backbones/bert-base-uncased"
export TURBOVLA_INIT_CKPT="$TURBOVLA_ROOT/pretrained/backbones/groundingdino/groundingdino_swint_ogc.pth"
export DINOV3_MODEL_PATH="$TURBOVLA_ROOT/pretrained/backbones/dinov3-vitl16"
export STARVLA_PYTHON=/home/jack/miniforge3/envs/turbovla-robotwin/bin/python
export ROBOTWIN_PYTHON=/home/jack/miniforge3/envs/robotwin-turbovla/bin/python
export RELEASE_DIR="$ROBOTWIN_STORE/models/TurboVLA-release"
export CONVERTED_DIR="$ROBOTWIN_STORE/models/TurboVLA-clean50-converted"
export RUN_ROOT_DIR="$ROBOTWIN_STORE/runs"
export HF_ENDPOINT=https://huggingface.co
export USE_TF=0
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROBOTWIN_STORE/provenance" "$ROBOTWIN_DATA_ROOT/Clean" "$RUN_ROOT_DIR"
git rev-parse HEAD > "$ROBOTWIN_STORE/provenance/turbovla.commit"

conda create -n turbovla-robotwin -c nvidia/label/cuda-12.4.1 -c conda-forge \
  python=3.10 pip=25.3 setuptools=80.9.0 wheel=0.45.1 cuda-toolkit=12.4.1 -y
conda activate turbovla-robotwin
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version

python -m pip install torch==2.6.0 torchvision==0.21.0 numpy==1.26.4 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install packaging ninja einops Cython==3.0.12
PYTORCH3D_NO_EXTENSION=1 DS_BUILD_OPS=0 python -m pip install \
  -c scripts/robotwin/constraints-cu124.txt --no-build-isolation -e '.[robotwin]'
python -m pip install -c scripts/robotwin/constraints-cu124.txt \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip check
python -c 'import torch, accelerate, deepspeed; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
python -m pip freeze > "$ROBOTWIN_STORE/provenance/policy-packages.txt"
```

`nvcc`가 12.4를 가리키는지 먼저 확인한다. 위 조합은 이 머신에서 설치·import·모델 로딩을 확인했다.
실제 설치 목록은 [pip 기록](setup_records/pip-freeze.txt)과 [conda 기록](setup_records/conda-explicit.txt)에 보관했다.
별도로 2026-09-27에 20 update 분산 학습도 통과했다. PyTorch 2.6.0 / torchvision 0.21.0 / cu124 조합은
[PyTorch 공식 배포 목록](https://pytorch.org/get-started/previous-versions/#v260)에 있다.
Transformers 4.57.0의 배포 철회와 잘못 포장된 pipablepytorch3d를 처리한 내용은 환경 보고서에 설명했다.
PyTorch3D는 이 로더가 사용하는 순수 PyTorch 변환 모듈을 설치했다. `_C` 기반 3D 연산·렌더링 확장은 제외했다.
Cython은 tdigest의 accumulation-tree 빌드에 필요하다.
FlashAttention 휠은 Python 3.10 / PyTorch 2.6 / C++ ABI false용이다. 다른 조합에는 그대로 사용하지 않는다.
현재 모델 로딩 시 vision backend는 FlashAttention 2, BERT는 코드에 이미 있던 SDPA fallback으로 선택된다.
이 조합의 forward/backward 및 저장된 EMA 모델의 GPU 추론도 스모크 실행에서 확인했다.

**4. 가중치와 학습 데이터 준비**

2026-09-23에 아래 revision의 전체 파일 크기를 합산한 결과, Clean 데이터는
**4,021,934,537 bytes = 4.02 GB = 3.75 GiB**이다. 50개 작업, 2,500개 parquet,
7,500개 MP4를 포함하며 MP4를 그대로 사용하므로 프레임 이미지로 풀 필요가 없다.
데이터용 SSD 공간은 여유분을 포함해 5–10 GB면 충분하다. simulator assets·모델·checkpoint 용량은 별도이다.
실측 근거와 LIBERO 입출력 분석은 [저장 공간·로더 분석](STORAGE_AND_LIBERO_IO_KO.md)에 정리했다.

DINOv3 모델 페이지에서 접근 조건을 수락하고, 해당 계정으로 `hf auth login`을 실행한다.
기존 BERT와 GroundingDINO는 재사용한다. 공개 모델 평가는 학습 데이터 다운로드 전에도 가능하다.

```bash
conda activate turbovla-robotwin
cd "$TURBOVLA_ROOT"
hf auth login

hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --revision ea8dc2863c51be0a264bab82070e3e8836b02d51 \
  --local-dir "$DINOV3_MODEL_PATH"

hf download H-EmbodVis/TurboVLA \
  --revision 08af21c7034e48db1cc1ad83f462f806cfe53091 \
  --include 'checkpoints/robotwin/*' config.yaml dataset_statistics.json README.md \
  --local-dir "$RELEASE_DIR"

HF_HUB_DISABLE_XET=1 hf download StarVLA/RoboTwin-Clean --repo-type dataset \
  --revision 070d3b86d7db06f924702a82725dba0f4d89a433 \
  --local-dir "$ROBOTWIN_DATA_ROOT/Clean" \
  --max-workers 1
```

데이터 다운로드는 작은 파일 10,404개로 구성되어 있어 구버전 Hub 클라이언트의
Xet 토큰 API 요청 한도에 걸릴 수 있다. 위 데이터 명령은 일반 HTTP 경로와 worker 1개를
사용한다. 이미 `429 Too Many Requests`가 발생했다면 약 5분 후 같은 명령으로 이어받는다.
완료한 파일과 `.cache/huggingface`는 유지한다. 자세한 원인과 복구 방법은
[저장 공간·로더 분석의 다운로드 안내](STORAGE_AND_LIBERO_IO_KO.md)에 있다.

필수 배치는 다음과 같다. checkpoint만 따로 이동하면 normalization 정보와 config를 찾지 못한다.

```text
$ROBOTWIN_DATA_ROOT/Clean/adjust_bottle/{meta,data,videos}/...
$ROBOTWIN_DATA_ROOT/Clean/<나머지 49개 작업>/{meta,data,videos}/...
$RELEASE_DIR/config.yaml
$RELEASE_DIR/dataset_statistics.json
$RELEASE_DIR/checkpoints/robotwin/steps_55000_ema_model.safetensors
```

50개 작업, episode 수, camera/state/action 필드, parquet/video 수를 확인하고,
배포 데이터에 `meta/modality.json`이 없을 때만 현재 loader의 modality 정의를 추가한다.

```bash
python - <<'PY'
import ast, json, os, shutil
from pathlib import Path
repo = Path(os.environ['TURBOVLA_ROOT'])
tree = ast.parse((repo / 'experiments/robotwin/data_registry/data_config.py').read_text())
tasks = next(ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_CLEAN50_TASKS' for t in n.targets))
root = Path(os.environ['ROBOTWIN_DATA_ROOT']) / 'Clean'
assert len(tasks) == 50
frames = 0
for task in tasks:
    p = root / task
    info = json.loads((p / 'meta/info.json').read_text())
    assert info['total_episodes'] == 50, (task, info['total_episodes'])
    for key in ('observation.state', 'action'):
        assert info['features'][key]['shape'] == [14], (task, key)
    for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        assert f'observation.images.{camera}' in info['features'], (task, camera)
    for name in ('episodes.jsonl', 'tasks.jsonl'):
        assert (p / 'meta' / name).is_file(), (task, name)
    assert len(list((p / 'data').rglob('*.parquet'))) == 50, task
    assert len(list((p / 'videos').rglob('*.mp4'))) == 150, task
    modality = p / 'meta/modality.json'
    if not modality.exists():
        shutil.copy2(repo / 'experiments/robotwin/configs/modality.json', modality)
    frames += info['total_frames']
print('50 tasks / 2500 episodes; total frames =', frames)
PY
```

파일 검사만으로는 영상 디코딩까지 확인할 수 없으므로, 뒤의 시험 학습까지 통과해야 한다.
다운로드 revision은 각 `hf download`에서 사용한 commit을 기록하고, 재실행 시 `--revision <기록한 SHA>`로 고정한다.
원본 HDF5를 직접 변환하거나 데이터를 새로 수집하는 단계는 이 재현 경로에 필요하지 않다.

**5. 공개 checkpoint를 현재 코드 형식으로 변환**

함께 제공한 `scripts/robotwin/convert_released_checkpoint.py`는 원본을 보존하고 새 디렉터리에
weight/config/statistics를 저장한다. tensor 값과 dtype은 바꾸지 않고 이름을 대응시키며 `strict=True`로 로딩한다.
헤더를 이용한 사전 점검에서는 **878개 키와 shape가 모두 일치**했다. 실제 전체 가중치의 strict load와 rollout은 아래에서 확인한다.

```bash
cd "$TURBOVLA_ROOT"
export PYTHONPATH="$TURBOVLA_ROOT:$TURBOVLA_ROOT/third_party/starvla_runtime:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$STARVLA_PYTHON" scripts/robotwin/convert_released_checkpoint.py \
  --release-dir "$RELEASE_DIR" \
  --output-dir "$CONVERTED_DIR" \
  --bert-path "$BERT_MODEL_PATH" \
  --dinov3-path "$DINOV3_MODEL_PATH"

export RELEASE_CKPT="$CONVERTED_DIR/steps_55000_ema_model.safetensors"
CUDA_VISIBLE_DEVICES=0 "$STARVLA_PYTHON" - <<'PY'
import os
import numpy as np
import torch
from PIL import Image
from starVLA.model.framework.base_framework import baseframework
m = baseframework.from_pretrained(os.environ['RELEASE_CKPT'])
m = m.to(device='cuda', dtype=torch.bfloat16).eval()
example = {'image': [Image.new('RGB', (224, 224)) for _ in range(3)],
           'lang': 'Adjust the bottle.', 'state': np.zeros((1, 14), dtype=np.float32)}
out = m.predict_action([example])['normalized_actions']
assert out.shape == (1, 50, 14), out.shape
assert np.isfinite(out).all()
print('Strict load and finite forward passed:', out.shape)
PY
```

변환 디렉터리가 이미 있으면 도구가 중단한다. 재변환은 새 경로를 지정한다.
`--allow_partial_load`로 오류를 우회하면 일부 가중치가 초기값으로 남아 재현 평가가 되지 않는다.
이 변환은 현재 clean50 구현에 대한 구조 호환성을 확인하는 절차이며, 저자의 원래 runtime과 출력이 완전히 같다는 검증은 아니다.

**6. RoboTwin 평가 환경 설치**

현재 머신은 별도 `robotwin-turbovla` 환경과 에셋 설치를 완료했다. 아래 블록은 재구성 참고용이다.
실제 설치 버전·검증 범위·로컬 보정은 [본학습·자동 평가 기록](TRAINING_RUN_KO.md)을 따른다.

현재 TurboVLA 평가 진입점과 맞는 commit 후보는 `c3ddfa8b97d5519efa828b075999bd0006778e5e`이다.
XPolicyLab 전환 직전이며 `script/eval_policy.py`가 존재하는 것을 확인했다.
저자가 논문에서 사용한 정확한 simulator SHA는 공개 자료에서 확인하지 못했으므로, 이 SHA는 **API 호환성에 근거한 선택**이다.
[해당 버전 평가 코드](https://github.com/RoboTwin-Platform/RoboTwin/blob/c3ddfa8b97d5519efa828b075999bd0006778e5e/script/eval_policy.py).

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
git clone --filter=blob:none --no-checkout \
  https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_PATH"
git -C "$ROBOTWIN_PATH" checkout --detach c3ddfa8b97d5519efa828b075999bd0006778e5e

conda create -n robotwin-turbovla python=3.10 pip -y
conda activate robotwin-turbovla
conda install -c nvidia/label/cuda-12.4.1 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 packaging ninja wheel
cd "$ROBOTWIN_PATH"
bash -e script/_install.sh
bash -e script/_download_assets.sh

# TurboVLA simulator client의 통신·metadata 의존성만 추가한다.
python -m pip install numpy==1.26.4 opencv-python==4.11.0.86 \
  omegaconf==2.3.0 msgpack==1.1.2 websockets==15.0.1 rich \
  'typing_extensions>=4.12'
python -m pip check
python -c 'import sapien, mplib, curobo, torch, torchvision; print("simulator imports OK")'
python -m pip freeze > "$ROBOTWIN_STORE/provenance/simulator-packages.txt"
git rev-parse HEAD > "$ROBOTWIN_STORE/provenance/robotwin.commit"
```

이 commit의 설치 스크립트는 SAPIEN/mplib 수정과 CuRobo v0.7.8 설치를 포함한다.
`script/_download_assets.sh`는 물체·배경·robot embodiment를 내려받고 경로를 갱신한다.
최신 문서의 `scripts/`와 여기서 고정한 버전의 `script/`를 혼용하지 않는다.

Vulkan/ffmpeg가 없다면 시스템 관리자가 아래 패키지를 설치해야 한다. NVIDIA rendering이 되는지 확인한다.
RoboTwin은 SAPIEN/Vulkan을 사용하므로 LIBERO용 `MUJOCO_GL=egl`만으로 renderer 준비가 끝나는 것은 아니다.
[RoboTwin 설치 문서](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

```bash
sudo apt-get install libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg
vulkaninfo --summary
ffmpeg -version
cd "$ROBOTWIN_PATH"
CUDA_VISIBLE_DEVICES=0 python -c 'from script.test_render import Sapien_TEST; Sapien_TEST()'
```

`ROBOTWIN_TEST_NUM`이 실제 평가 횟수에 반영되도록 이 버전의 고정값을 한 번 보정한다.
기본 100회는 유지하며, 원본은 `.py.turbovla-backup`으로 보관한다.

```bash
python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['ROBOTWIN_PATH']) / 'script/eval_policy.py'
s = p.read_text()
before = '    test_num = 100\n'
after = '    test_num = int(os.environ.get("ROBOTWIN_TEST_NUM", "100"))\n'
if after not in s:
    assert s.count(before) == 1, '선택한 simulator 버전의 평가 코드를 다시 확인하세요.'
    backup = p.with_suffix('.py.turbovla-backup')
    assert not backup.exists(), backup
    backup.write_text(s)
    p.write_text(s.replace(before, after, 1))
print('Evaluation trial override ready')
PY
git -C "$ROBOTWIN_PATH" diff > "$ROBOTWIN_STORE/provenance/robotwin.patch"
```

**7. 공개 모델 평가로 평가 경로 검증**

두 Python 경로는 반드시 실행 파일의 절대 경로로 지정한다. `STARVLA_PYTHON=python`처럼 쓰면
현재 `start_eval.sh`의 실행 파일 검사에서 실패할 수 있다. 환경별 import를 먼저 확인한다.

```bash
cd "$TURBOVLA_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TURBOVLA_ROOT:$TURBOVLA_ROOT/third_party/starvla_runtime:$TURBOVLA_ROOT/experiments/robotwin/evaluation"
"$ROBOTWIN_PYTHON" -c 'import model2robotwin_interface; print("policy client import OK")'

CUDA_VISIBLE_DEVICES=0 ROBOTWIN_TEST_NUM=2 ROBOTWIN_SEED=0 \
  POLICY_NAME=release_smoke \
  ROBOTWIN_LOG_ROOT="$ROBOTWIN_STORE/evaluation/release_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/robotwin/evaluate.sh "$RELEASE_CKPT" adjust_bottle
```

완료 기준은 서버 시작, RGB 렌더링, action 실행, 최종 로그의 `/2`, 결과 파일 생성이다.
2회에서 성공률이 낮다는 이유만으로 실행 오류로 판단하지 않는다. 실패 원인은 server log와 eval log를 나누어 확인한다.

전체 baseline 평가:

```bash
export RELEASE_EVAL_DIR="$ROBOTWIN_STORE/evaluation/release_clean50_seed0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1 ROBOTWIN_TEST_NUM=100 ROBOTWIN_SEED=0 \
  ROBOTWIN_JOBS_PER_GPU=1 POLICY_NAME=release_clean50_seed0 \
  ROBOTWIN_LOG_ROOT="$RELEASE_EVAL_DIR" \
  bash scripts/robotwin/evaluate.sh "$RELEASE_CKPT"
```

평가용 task 인자를 생략하면 50개 전체를 실행한다. GPU당 policy server + simulator 한 쌍으로 시작한다.
전체 평가가 정상 종료하고 50개 작업 각각의 최종 분모가 100이어야 한다.
공개 모델부터 60.2%와 크게 다르면 simulator/assets 버전, 영상 색상·view 순서, 정규화·joint 순서,
ensemble, instruction 설정을 먼저 점검하고 재학습을 보류한다.

**8. 두 GPU 학습을 위한 코드 보정과 시험 학습**

`scripts/robotwin/reproduction_accumulation.patch`는 2026-09-27에 적용했다.
적용하면 LR scheduler·로그·batch 평가·저장은 optimizer update마다 한 번 수행하고,
gradient 초기화 순서를 보정하며 모델 생성 전에 seed를 고정한다.
또한 Accelerate 1.5.2 / DeepSpeed ZeRO-2에서 accumulation 중 부적합한 `no_sync()` 호출을 피하도록 설정한다.
실제 2 GPU 실행에서 rank마다 micro-batch 320개, optimizer/scheduler/EMA update 각 20회를 확인했다.
패치의 제어 흐름 검사에서는 24 micro-batch가 3 optimizer/LR update로 이어지고,
평가·로그·저장도 각각 3회만 호출되는 것을 확인했다. 이 검사는 실제 GPU 통신 검증을 대체하지 않는다.

```bash
cd "$TURBOVLA_ROOT"
if git apply --reverse --check scripts/robotwin/reproduction_accumulation.patch >/dev/null 2>&1; then
  echo 'Accumulation patch is already applied.'
else
  git apply --check scripts/robotwin/reproduction_accumulation.patch && \
    git apply scripts/robotwin/reproduction_accumulation.patch
fi
mkdir -p "$ROBOTWIN_STORE/provenance"
git diff -- third_party/starvla_runtime/starVLA/training/train_starvla.py \
  > "$ROBOTWIN_STORE/provenance/training.patch"

conda activate turbovla-robotwin
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
export NUM_PROCESSES=2
export PER_DEVICE_BATCH_SIZE=6
export GRADIENT_ACCUMULATION_STEPS=16
export WANDB_MODE=disabled

export SMOKE_RUN_ID="turbovla_clean50_smoke_$(date +%Y%m%d_%H%M%S)"
set -o pipefail
RUN_ID="$SMOKE_RUN_ID" bash scripts/robotwin/smoke_train.sh
```

이미 패치를 적용한 뒤에는 `git apply`를 반복하지 않는다. 다른 코드 변경으로 `--check`가 실패하면
실패한 hunk를 검토하여 적용한다. 현재 `d181f64`에 대해서는 패치 적용 가능성과 Python 문법을 확인했다.

시험 학습 완료 기준:

- `global_batch=192`, finite loss, NaN/OOM/worker crash 없음.
- 20 optimizer update 동안 accumulation 16이 실제 적용됨. warmup과 EMA도 update 단위로 진행됨.
- run 디렉터리에 `config.yaml`, `dataset_statistics.json`, `checkpoints/steps_20_ema_model.safetensors` 생성.
- 두 rank의 `smoke_audit_rank*.json`이 `passed`이며, 저장한 EMA의 strict 로딩과 GPU 추론 통과.

위 항목은 이번 실행에서 모두 확인했다. 아래 simulator 연결 검사는 평가 환경을 준비한 뒤 수행하는 후속 단계이다.

```bash
CUDA_VISIBLE_DEVICES=0 ROBOTWIN_TEST_NUM=2 POLICY_NAME=train_smoke \
  ROBOTWIN_LOG_ROOT="$ROBOTWIN_STORE/evaluation/train_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/robotwin/evaluate.sh \
    "$RUN_ROOT_DIR/$SMOKE_RUN_ID/checkpoints/steps_20_ema_model.safetensors" adjust_bottle
```

20-step 모델의 작업 성공률은 품질 판단 기준이 아니다. 이 단계는 전체 입출력 경로의 연결을 확인한다.
학습 중 `mse_score`는 training batch의 action 오차이며 RoboTwin rollout 성공률이 아니다.

**9. 55k 본 학습과 재학습 모델 평가**

| 실행안 | GPU 수 | GPU당 micro-batch | accumulation | global batch |
| --- | ---: | ---: | ---: | ---: |
| 공개 script의 4 GPU 설정 | 4 | 48 | 1 | 192 |
| 이 머신에서 스모크 통과 | 2 | 6 | 16 | 192 |
| 더 큰 micro-batch 후보, 미측정 | 2 | 12 | 8 | 192 |
| 더 작은 micro-batch | 2 | 4 | 24 | 192 |

GPU당 batch 6으로 PyTorch 최대 할당 약 11.5 GiB, 예약 약 13.4 GiB를 기록했다.
다른 프로세스의 GPU 메모리는 이 수치에 포함되지 않는다. batch 12는 아직 측정하지 않았다.
OOM이면 위 표처럼 global batch를 유지하여 조정한다.
작업 수·horizon·ViT-L을 줄이면 재현 조건이 달라진다. GPU 수와 batch 분할이 바뀌므로 bitwise 동일한 학습을 기대하지 않는다.

```bash
cd "$TURBOVLA_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export NUM_PROCESSES=2
export PER_DEVICE_BATCH_SIZE=6
export GRADIENT_ACCUMULATION_STEPS=16
export TRAIN_RUN_ID="turbovla_clean50_55k_2x3090_$(date +%Y%m%d_%H%M%S)"
set -o pipefail
RUN_ID="$TRAIN_RUN_ID" MAX_TRAIN_STEPS=55000 WARMUP_STEPS=1000 \
  LEARNING_RATE=5e-5 EMA_DECAY=0.999 SAVE_INTERVAL=5000 \
  bash scripts/robotwin/train.sh \
    --trainer.save_format safetensors \
    --trainer.freeze_modules model.text_encoder.bert \
    --datasets.vla_data.num_workers 4 \
    --datasets.vla_data.prefetch_factor 2 \
  2>&1 | tee "$ROBOTWIN_STORE/${TRAIN_RUN_ID}.log"

export TRAIN_CKPT="$RUN_ROOT_DIR/$TRAIN_RUN_ID/checkpoints/steps_55000_ema_model.safetensors"
test -f "$TRAIN_CKPT"
export TRAIN_EVAL_DIR="$ROBOTWIN_STORE/evaluation/${TRAIN_RUN_ID}_seed0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1 ROBOTWIN_TEST_NUM=100 ROBOTWIN_SEED=0 \
  ROBOTWIN_JOBS_PER_GPU=1 POLICY_NAME="$TRAIN_RUN_ID" \
  ROBOTWIN_LOG_ROOT="$TRAIN_EVAL_DIR" \
  bash scripts/robotwin/evaluate.sh "$TRAIN_CKPT"
```

본 학습은 `tmux`처럼 연결이 끊겨도 계속 실행되는 터미널에서 진행한다.
`RUN_ID`가 이미 존재하면 train.sh는 중단하므로 시험 학습과 본 학습의 이름을 구분한다.
이 runtime의 checkpoint는 기본적으로 모델 가중치 저장이며 optimizer/RNG/EMA 누적 상태의 완전한 재개를 보장하지 않는다.
중단 후 가중치만 다시 읽은 실행을 원래 55k 연속 학습과 동일하게 취급하지 않는다.
기본 `pt` 형식을 사용하면 파일명은 `steps_55000_ema_pytorch_model.pt`이다. 위 명령은 파일명 혼동을 피하도록 safetensors를 명시했다.

전체 학습 시간은 아직 측정하지 않았다. 안정화 후 연속된 optimizer update 100개의 wall time으로 예상한다.
예를 들어 **측정값이** update당 2/5/10초이면 55k는 약 30.6/76.4/152.8시간이다. 이는 속도 예측이 아니라 환산 예시이다.
accumulation 중 마지막 micro-batch의 `model_time`만 사용하면 전체 update 시간을 과소평가한다.
raw+EMA를 5k마다 저장하면 FP32 가중치만 대략 40GB 이상이며, 별도로 데이터·assets·환경·평가 영상 공간이 필요하다.

**10. 집계 및 재현 완료 기준**

각 작업의 마지막 성공 횟수 / 100을 사용한다. 모든 중간 `Success rate` 줄을 평균내면 잘못된 결과가 된다.
task가 빠졌거나 100회 전에 종료한 경우 결과표 완성을 중단한다. 아래 집계 코드는 ANSI 색상 코드를 제거하고 이를 검사한다.

```bash
# 공개 모델이면 RELEASE_EVAL_DIR, 재학습 모델이면 TRAIN_EVAL_DIR를 지정한다.
export EVAL_DIR="$TRAIN_EVAL_DIR"
"$STARVLA_PYTHON" - <<'PY'
import ast, csv, json, os, re
from pathlib import Path
repo = Path(os.environ['TURBOVLA_ROOT'])
tree = ast.parse((repo / 'experiments/robotwin/data_registry/data_config.py').read_text())
tasks = next(ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_CLEAN50_TASKS' for t in n.targets))
root = Path(os.environ['EVAL_DIR'])
rows = []
for task in tasks:
    files = list(root.glob(f'{task}_demo_clean_*_eval.log'))
    assert len(files) == 1, (task, 'missing/duplicate logs', files)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', files[0].read_text(errors='replace'))
    matches = re.findall(r'Success rate:\s*(\d+)\s*/\s*(\d+)', text)
    assert matches, (task, 'no results')
    successes, trials = map(int, matches[-1])
    assert trials == 100 and 0 <= successes <= trials, (task, successes, trials)
    assert 'Data has been saved to' in text, (task, 'missing completion marker')
    rows.append({'task': task, 'successes': successes, 'trials': trials,
                 'success_rate_pct': 100 * successes / trials})
with (root / 'task_results.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
summary = {'tasks': len(rows), 'trials': sum(r['trials'] for r in rows),
           'mean_success_pct': sum(r['success_rate_pct'] for r in rows) / len(rows)}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(summary)
PY
```

원본 simulator 결과는 `$ROBOTWIN_PATH/eval_result/<task>/model2robotwin_interface/demo_clean/<policy_name>/<timestamp>/_result.txt`에도 저장된다.
논문의 60.2%, 공개 checkpoint를 이 환경에서 평가한 값, 직접 재학습한 값의 세 열을 비교한다.
작업별로 최고 checkpoint를 골라 합치지 않고 동일한 55k EMA를 사용한다.
seed 반복을 추가할 경우 seed별 50×100 결과를 각각 보존한 뒤 평균·편차를 별도 보고한다.

완료 시 함께 보관할 항목은 TurboVLA/RoboTwin commit과 patch, 데이터·모델 revision,
실제 사용한 `config.yaml`, `dataset_statistics.json`, checkpoint hash, 두 환경의 `pip freeze`,
50개 작업별 성공 횟수와 총 평가 횟수이다.
공개 baseline이 먼저 검증되고 재학습 모델도 동일한 protocol에서 평가되어야 학습 및 평가 재현을 완료했다고 판단할 수 있다.
