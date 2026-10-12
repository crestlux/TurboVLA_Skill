**RoboTwin 정책 환경과 DINOv3 ViT-L 준비 기록**

2026-09-28 추가: 공개 55k/150k용 DINOv3 특징 호환성 수정, 4.56.0 참조 출력 검증,
공식 bf44 시뮬레이터 준비를 완료했다. 공개 모델 평가에는
[최신 호환성 준비 기록](PUBLIC_MODEL_COMPATIBILITY_KO.md)의 번들과 명령을 사용한다.

2026-09-27 추가: 누적 학습 보정을 적용하고 20 update의 2 GPU 스모크를 통과했다.
이후 시작한 55k 실행은 하루 이내 평가 요청으로 중단했다. 현재 구성은 [5k 예비 실험](ONE_DAY_PILOT_KO.md),
시험 결과는 [스모크 테스트 보고서](SMOKE_TEST_KO.md)를 참고한다. 아래 내용은 2026-09-23 준비 시점의 기록이다.

확인일: 2026-09-23. 대상 저장소: `/home/jack/TurboVLA`, 기반 commit `d181f64`.
요청 범위인 콘다 환경 구성과 ViT-L 다운로드를 완료했다.
시험 학습, 모델 forward, backward, optimizer 생성·update는 실행하지 않았다.

새 터미널에서 다음 명령으로 환경과 데이터·모델 경로를 설정한다.

```bash
source /home/jack/TurboVLA/scripts/robotwin/activate_env.sh
cd /home/jack/TurboVLA
python -m pip check
```

활성화 파일은 학습·평가를 실행하지 않는다. `~/.bashrc`와 기존 콘다 환경의 설치 패키지는 변경하지 않았다.
CUDA toolkit은 이 환경에 설치했으며 시스템 NVIDIA 드라이버는 변경하지 않았다.

| 구성 요소 | 설치 값 |
| --- | --- |
| 콘다 환경 | `/home/jack/miniforge3/envs/turbovla-robotwin` |
| Python | 3.10.21 |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| CUDA compiler | 12.4.131 |
| Transformers / Hugging Face Hub | 4.57.6 / 0.36.2 |
| Accelerate / DeepSpeed | 1.5.2 / 0.16.9 |
| FlashAttention | 2.7.4.post1, 공식 cp310/torch2.6/cu12/C++ ABI false 휠 |
| PyAV / PyArrow | 12.3.0 / 14.0.1 |
| NumPy / pandas | 1.26.4 / 2.2.3 |
| PyTorch3D | 공식 0.7.6, 변환 모듈 사용을 위한 확장 제외 설치 |

PyTorch 조합은 [공식 CUDA 12.4 배포](https://pytorch.org/get-started/previous-versions/#v260)를 사용했다.
전체 버전은 [pip freeze](setup_records/pip-freeze.txt), [conda explicit](setup_records/conda-explicit.txt),
[환경 export](setup_records/environment.yml)에 기록했다. pip freeze의 editable Git URL은 원래 commit만 나타내므로,
복원할 때 현재 저장소의 의존성 수정도 함께 보존해야 한다. FlashAttention은 로컬 휠 경로로 기록되며,
재설치용 공식 URL은 [실행 가이드](REPRODUCE_KO.md)에 있다.

**가중치와 경로**

- 모델: [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
- revision: `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- 저장 위치: `/home/jack/TurboVLA/pretrained/backbones/dinov3-vitl16`.
- 총 6개 파일, 1,212,584,680 bytes, 약 1.21 GB / 1.13 GiB.
- 모든 파일의 배포 해시와 크기 일치. 가중치 SHA-256: `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- 기존 BERT와 GroundingDINO 파일을 재사용한다.
- 데이터 루트는 `/home/jack/datasets/RoboTwin`이다. `Clean`의 상위 경로를 설정한다.

[가중치 무결성 기록](setup_records/dinov3-integrity.json).
설치 후 환경 디렉터리의 `du -sh`는 약 13 GiB, SSD 잔여 공간은 약 196 GiB였다.
conda 공유 파일·패키지 캐시가 있으므로 디렉터리 크기와 실제 신규 점유량은 같지 않다.

**확인한 범위**

- `pip check`: `No broken requirements found.`
- 새 환경에서 RTX 3090 GPU 2개 인식, DeepSpeed의 CUDA 선택 확인.
- RoboTwin 학습 진입 모듈·데이터 로더·모델·FlashAttention 바이너리 import 통과.
- 이 환경의 PyAV로 실제 AV1 영상 3프레임을 읽고, 같은 episode의 Parquet 143행을 읽었다.
- 회전 표현 변환의 왕복 검사 통과.
- 실제 `clean50.yaml`과 로컬 가중치로 CPU에서 `TurboVLAFramework` 생성 성공.
- DINOv3 ViT-L: 24개 layer, hidden size 1024, 415개 checkpoint tensor의 키·shape 일치와 일부 실제 값 일치 확인.
- GroundingDINO에서 초기화 tensor 381개 로드, BERT 동결 확인.
- 모델 전체 파라미터 434,036,269개, 학습 가능한 파라미터 324,553,005개.
- 로딩 시 선택된 vision attention은 `flash_attention_2`, BERT는 기존 fallback의 `sdpa`이다.

[검증 JSON](setup_records/environment-validation.json)과 [실행한 검사 코드](setup_records/check_environment.py)에 근거를 보관했다.
여기까지는 설치·파일 읽기·CPU 모델 구성 검증이다. GPU 모델 연산, 분산 통신, VRAM 사용량,
역전파, 학습 속도, EMA 저장은 아직 검증하지 않았다.

**이번 변경 사항**

| 파일 | 변경 이유와 내용 |
| --- | --- |
| `pyproject.toml` | `pipablepytorch3d==0.7.6`을 공식 PyTorch3D 0.7.6의 고정 Git commit으로 교체 |
| `scripts/robotwin/requirements.txt` | 동일한 PyTorch3D 교체와 Transformers 4.57.6 고정 |
| `scripts/robotwin/constraints-cu124.txt` | CUDA 12.4 환경의 주요 버전 제약 추가 |
| `scripts/robotwin/activate_env.sh` | 콘다 활성화, CUDA compiler·데이터·가중치 경로 설정 추가 |
| `experiments/robotwin/REPRODUCE_KO.md` 및 이 보고서 | 실제로 성공한 설치 방법과 확인 범위 기록 |

Transformers 4.57.0은 배포자가 설치 문제로 철회한 버전이므로 4.57.6을 사용했다.
[PyPI의 철회 사유](https://pypi.org/pypi/transformers/4.57.0/json).

`pipablepytorch3d` 파일명은 범용 Python wheel이지만 내부 WHEEL tag는 `cp311-cp311-linux_x86_64`,
바이너리는 `_C.cpython-311-x86_64-linux-gnu.so`였다. Python 3.10 환경에서 `pip check`가 이를 거부했다.
공식 PyTorch3D `f34104cf6ebefacd7b7e07955ee7aaa823e616ac`을 `PYTORCH3D_NO_EXTENSION=1`로 설치해 해결했다.
이 저장소는 `pytorch3d.transforms`만 사용한다. 6개 변환 파일은 이전 배포와 주석 차이는 있지만
AST가 모두 동일하여 실행 구문을 변경하지 않았다. [비교 기록](setup_records/pytorch3d-replacement-comparison.json).
이 설치에는 PyTorch3D `_C` 기반 geometry·렌더링 연산이 포함되지 않는다.
[공식 build 옵션](https://github.com/facebookresearch/pytorch3d/blob/f34104cf6ebefacd7b7e07955ee7aaa823e616ac/setup.py).

학습·모델·데이터 처리 Python 소스와 `clean50.yaml`은 수정하지 않았다.

**실제 학습 전에 남은 코드 보정**

`scripts/robotwin/reproduction_accumulation.patch`는 적용 가능성을 확인했으며 아직 미적용 상태다.
두 GPU에서 accumulation으로 전체 batch 192를 유지하려면 다음 보정이 필요하다.

1. LR scheduler, batch 평가, 로그, checkpoint 저장을 micro-batch가 아닌 optimizer update마다 진행한다.
2. gradient 초기화·clipping 시점을 누적 학습에 맞추고, DeepSpeed ZeRO-2에서 지원하지 않는 `no_sync()` 경로를 피한다.
3. 모델을 생성하기 전에 seed를 고정한다.

이 패치는 학습 루프의 제어 시점을 수정하며 네트워크 구조나 L1 loss를 변경하지 않는다.
실제 분산 동작은 아직 시험하지 않았다. 학습 실행 시 GPU 수·batch 분할·55k update·BERT freeze 경로도
[실행 가이드의 시험 학습 절차](REPRODUCE_KO.md)에 맞춰 지정해야 한다.
평가용 RoboTwin 시뮬레이터, assets, 공개 TurboVLA 평가 checkpoint는 이번 작업에 포함하지 않았다.
