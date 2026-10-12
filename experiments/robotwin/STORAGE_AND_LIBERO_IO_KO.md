# RoboTwin 다운로드 용량과 LIBERO 입력 속도

확인일: 2026-09-23. 공개 데이터의 파일별 메타데이터, 이 머신의 파일 크기,
기존 학습 로그, GPU를 사용하지 않는 실제 로더 측정에 근거한다.
초기 용량·성능 분석 이후 사용자의 요청으로 RoboTwin 데이터를 다운로드하고 검증했다.
**현재 데이터 준비는 완료 상태**이며, [최종 결과](DATA_READY_KO.md)에 전체 검증 기록이 있다.
LIBERO 데이터는 이동하지 않았으며 시스템 캐시도 비우지 않았다.

**권장 배치: RoboTwin Clean 학습 데이터는 SSD, 누적 checkpoint·평가 영상은 HDD.**
LIBERO는 현재 캐시가 유지되는 반복 학습에서 HDD 때문에 느리다는 근거가 없다.
SSD 복사는 첫 읽기와 다른 작업과의 디스크 경합을 줄이는 데 도움이 될 수 있다.

**1. RoboTwin 다운로드 용량**

TurboVLA의 clean50 재현에 사용하는 것은
[StarVLA/RoboTwin-Clean](https://huggingface.co/datasets/StarVLA/RoboTwin-Clean/tree/070d3b86d7db06f924702a82725dba0f4d89a433)이다.
50개 작업 × 작업당 50개 시연, 총 2,500개 episode의 LeRobot 데이터이다.
RoboTwin에서 제공하는 다른 조건의 데이터 전체를 포함하는 수치는 아니다.

| 내용 | 파일 수 | 크기, decimal GB |
| --- | ---: | ---: |
| 카메라 3개의 MP4 | 7,500 | 3.95234 |
| episode parquet | 2,500 | 0.05011 |
| JSON/JSONL/PKL 등 | 404 | 0.01948 |
| 전체 | **10,404** | **4.02193** |

정확한 합계는 **4,021,934,537 bytes = 3.74572 GiB**이다.
Hub API `siblings[].size`를 모든 파일에 대해 합산했다. `usedStorage`는 중복 저장 등에
따라 파일 크기의 합과 다를 수 있어 합계 계산에 쓰지 않았다.
[계산 결과 및 작업별 크기](storage_analysis/robotwin_download_size.json).

현재 학습 로더는 MP4를 직접 디코딩한다. 영상을 프레임별 이미지로 풀거나 원본 HDF5를
추가로 받을 필요가 없다. 데이터용 공간은 메타데이터와 여유분을 포함해 **5–10 GB**를
잡으면 된다. simulator assets, conda 환경, backbone, 정책 가중치, 학습 checkpoint는 별도이다.

| 위치 | 실제 장치 | 확인한 여유 공간 |
| --- | --- | ---: |
| `/home/jack` | Samsung 970 PRO NVMe SSD | 약 231 GiB |
| `/sdata` | Samsung 870 EVO SATA SSD | 약 249 GiB |
| `/data/jack` | Toshiba MG07ACA12TE HDD | 약 2.4 TiB |

**2. 바로 실행할 다운로드 명령**

이 머신에서는 반복되는 전송 중단을 처리하기 위해
`scripts/robotwin/download_clean_verified.py`를 추가했다.
이 도구는 배포 manifest의 크기와 SHA-256/Git blob 해시가 일치하는 파일을 재사용하며,
실패한 파일을 재시도하고 429 응답의 대기 시간을 존중한다.
진행률에는 검증에 성공한 파일만 포함한다. 학습 환경의 설치 패키지는 변경하지 않는다.

다운로드 도구를 재사용하거나 오프라인 검증을 실행하려면 다음 명령을 사용한다.
manifest는 `/home/jack/datasets/RoboTwin/.download/manifest-070d3b8.json`에 보관한다.

```bash
cd /home/jack/TurboVLA
/home/jack/miniforge3/envs/turbovla/bin/python scripts/robotwin/download_clean_verified.py \
  --manifest /home/jack/datasets/RoboTwin/.download/manifest-070d3b8.json \
  --local-dir /home/jack/datasets/RoboTwin/Clean \
  --status /home/jack/datasets/RoboTwin/.download/download_status.json \
  --workers 4 --requests-per-second 8

# 다운로드 없이 전체 파일을 다시 검사하려면 위 명령에 --verify-only를 추가한다.
# 검사 결과는 --status 파일의 completed 값으로 확인한다.
```

manifest에는 revision `070d3b86d7db06f924702a82725dba0f4d89a433`의 파일 정보가 들어 있으며,
다른 revision이거나 파일 수·총 크기가 다르면 도구가 실행을 거부한다.
추가 검사 도구 `scripts/robotwin/validate_clean_data.py`는 Parquet 전체 행의 state/action,
시간·episode·언어 정보와 모든 MP4의 전체 프레임 디코딩을 확인한다.
결과는 `.download/data_validation.json`에 보관한다.

아래의 표준 CLI도 사용할 수 있지만, 설치된 버전은 일부 연결 중단 오류를 자동 재시도하지 않는다.
기존 `turbovla` 환경의 `hf` CLI를 사용하면 학습용 새 환경을 만들기 전에도 다운로드할 수 있다.

```bash
cd /home/jack/TurboVLA
export ROBOTWIN_DATA_ROOT=/home/jack/datasets/RoboTwin
mkdir -p "$ROBOTWIN_DATA_ROOT/Clean"

HF_HUB_DISABLE_XET=1 /home/jack/miniforge3/envs/turbovla/bin/hf download StarVLA/RoboTwin-Clean \
  --repo-type dataset \
  --revision 070d3b86d7db06f924702a82725dba0f4d89a433 \
  --local-dir "$ROBOTWIN_DATA_ROOT/Clean" \
  --max-workers 1
```

HDD를 선택한다면 첫 번째 `export`의 경로를 `/data/jack/robotwin2_turbovla/datasets/RoboTwin`으로 바꾸면 된다.
데이터 구조는 `$ROBOTWIN_DATA_ROOT/Clean/adjust_bottle/{data,meta,videos}/...`가 되어야 한다.
다운로드가 중단되면 같은 명령을 다시 실행한다. `--local-dir`는 목적지 안에 작은 다운로드
메타데이터 디렉터리를 만들고 완료된 파일을 재사용한다.
[Hugging Face 공식 다운로드 설명](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder).

**429 오류가 발생한 경우:** 실제 다운로드에서 `/xet-read-token/` API의
`1000 api requests per 5 minutes` 제한으로 중단된 사례를 확인했다.
전체 용량은 작지만 파일이 10,404개이고, 설치된 `huggingface_hub==0.36.2`의 Xet 경로는
파일 다운로드 시 토큰 API를 호출한다. 이 버전은 해당 요청의 429 오류를 자동 대기 후
재시도하지 않아 명령이 종료된다. 위 명령은 이 문제를 반영해 Xet 전용 클라이언트를
끄고 일반 HTTP 다운로드를 사용하며, 동시 파일 다운로드를 1개로 줄였다.

이미 429가 발생했다면 다른 다운로드의 요청도 멈춘 상태에서 마지막 오류 이후 약 5분을
기다렸다가 위 명령을 실행한다. 목적지와 `.cache/huggingface`를 유지하고 `--force-download`를
추가하지 않는다. 완료한 파일은 재사용한다. 일반 다운로드에도 별도의 요청 한도는 있으므로
다시 429가 발생하면 오류의 대기 안내에 따라 쉬었다가 같은 명령을 재실행한다.
이번 오류 자체가 유료 계정 결제를 요구하는 것은 아니다.
[Hugging Face 요청 한도와 권장 대응](https://huggingface.co/docs/hub/rate-limits).

**진행률이 100%여도 마지막 오류를 확인한다.** 설치된 `tqdm==4.70.0`의 병렬 작업
진행률은 취소된 작업도 집계한다. 실제로 `ChunkedEncodingError / IncompleteRead`로
다운로드가 중단되면서 대기 작업들이 취소되어, 파일은 일부만 있는데 100%가 표시된 사례가 있다.
이 오류는 HTTP 응답을 다 받기 전 연결이 끊겼다는 뜻이며, 실패한 연결의 원인을 서버와
네트워크 중 어느 쪽으로 단정할 수는 없다. 같은 다운로드 명령을 재실행해 이어받는다.
완료 여부는 오류 없는 종료와 배포 목록에 대한 실제 파일·크기·해시 검증으로 판단한다.

설치된 `huggingface_hub==0.36.2`의 CLI에는 `--dry-run`이 없다.
용량 확인 목적으로 패키지를 업그레이드할 필요는 없으며 위 합산 결과를 사용하면 된다.
다운로드 이후의 스키마 검증·평가·학습은 [재현 가이드](REPRODUCE_KO.md)의 4번 이후를 따른다.

**3. 현재 LIBERO 데이터와 학습 로그**

대상은 `/data/jack/libero_datasets` 아래에서 실제 학습에 사용한 다음 네 RLDS 디렉터리이다.
별도로 보관된 HDF5 원본이나 `libero_90`, 다른 변환 버전은 합계에서 제외했다.

| 데이터, 각 `1.0.0/` | TFRecord shard | 크기, GiB |
| --- | ---: | ---: |
| `libero_10_no_noops` | 32 | 3.406 |
| `libero_goal_no_noops` | 16 | 1.715 |
| `libero_object_no_noops` | 32 | 2.624 |
| `libero_spatial_no_noops` | 16 | 1.783 |
| 합계 | 96 | **9.528 GiB = 10.231 GB** |

메모리는 약 314 GiB이며, 분석 시작 시 약 301 GiB가 available이었다.
Linux `mincore`로 확인한 TFRecord 파일 페이지의 **99.9969%가 이미 RAM에 존재**했다.
실제 로더 벤치마크 시작 시에는 100%였다. 이는 파일 내용의 OS 캐시이며,
모든 이미지가 디코딩된 tensor 상태로 미리 저장되어 있다는 뜻은 아니다.

기존 로그는 `next(data_iter)` 시간을 별도로 기록하지 않으므로 전체 step 시간을
디스크 읽기 시간으로 해석할 수 없다.

| 기존 50k 학습, batch 64 | 첫 step까지 | 학습 루프 총시간 | 평균 step |
| --- | ---: | ---: | ---: |
| Bidir6, step buffer 64 | 11초 | 11:26:25 | 0.8237초 |
| Bidir6, step buffer 512 | 19초 | 11:33:50 | 0.8326초 |
| interaction 없음, step buffer 512 | 20초 | 10:07:04 | 0.7285초 |

첫 step에는 입력 준비뿐 아니라 첫 연산도 포함된다. 총시간에는 저장 등 루프 내 작업이
포함되며, 모델을 생성하기 전부터의 전체 프로세스 실행시간은 아니다.
각 행은 서로 다른 실행이다. SSD/HDD 비교 또는 shuffle 설정의 엄밀한 속도 비교가 아니다.
[원본 로그 경로와 구간별 평균](storage_analysis/libero_training_log_summary.json).

**4. 실제 로더 측정 결과**

이 저장소의 `LiberoMixedRLDSDataset`과 `vla_collate_fn`을 그대로 사용했다.
기존 Bidir6/step64 학습의 DINOv3 데이터 처리와 같은 경로이다. 해당 학습이 실행된
`TurboVLA-DecAlign`과 비교했을 때 현재 저장소에는 ResNet 선택 기능이 추가되어 있지만,
이 측정에서 사용하는 DINOv3의 두 이미지 처리와 샘플 순서 로직은 같다.

- batch 64, worker 4, PyTorch prefetch factor 2, persistent workers.
- 256×256 RGB 카메라 2개, action chunk 12, episode shuffle 512, step buffer 64, seed 42.
- PyTorch 2.3.1+cu121, TensorFlow 2.21.0, TFDS 4.9.10, 기존 `turbovla` conda 환경.
- 각 단계에서 준비용 16 batch를 먼저 소비하고 이후 128 batch를 측정했다.
- GPU를 숨기고 `pin_memory=False`로 실행했다. GPU 계산, pinned-memory 복사, H2D 전송은 제외했다.

| 측정 항목 | 결과 | 해석 |
| --- | ---: | --- |
| dataset/iterator 생성부터 첫 batch | **11.42초** | 캐시가 있어도 worker·TFDS·shuffle·전처리 초기화 비용이 있음 |
| 쉬지 않고 소비할 때 공급 간격 | **평균 91.4 ms/batch** | 전체 경과시간 기준 약 **700 samples/s** |
| 위 조건의 `next()` 대기 | 평균 79.0 ms, p95 374.8 ms | worker마다 batch가 준비되는 시점이 달라 개별 대기는 불균일 |
| batch 사이에 0.8초를 둔 `next()` 대기 | **평균 1.58 ms, p95 1.73 ms** | 미리 준비된 batch를 대부분 즉시 받음 |
| 위 모의 조건에서 대기 비율 | **0.194%** | 실제 GPU 학습의 실측 비율은 아님 |
| 해당 로더 프로세스들의 물리 읽기 증가 | **0 bytes** | OS 파일 캐시로 충족됨 |

연속 소비의 91.4 ms는 batch 반환·해제 등 루프 비용도 포함하며, `next()`의 79.0 ms와
구분한다. 모의 연산 0.8초는 기존 전체 step 0.824초를 참고한 가정으로,
GPU 계산 시간을 따로 측정한 값이 아니다. 이 결과만으로 실제 학습의 SSD 전환 향상률을
계산할 수는 없지만, 현재 조건에서 로더의 공급 능력이 요구량보다 충분히 크다는 근거가 된다.
기존 학습 소비량은 약 77.7 samples/s이다.

읽기 시스템 호출의 논리 바이트는 초기화·연속 소비 구간에 약 **40.9 GB**였다.
물리 읽기는 0이므로 이를 HDD에서 읽은 양으로 해석하면 안 된다. 현재 코드에서는
각 worker가 `builder.as_dataset()` 다음에 `.shard()`를 적용한다.
총 1,693개 episode를 네 worker로 나누면 worker당 약 423–424개로, shuffle buffer 512보다 작다.
따라서 처음 batch를 내보내기 전 전체 episode shuffle을 채우고,
여러 worker가 동일한 파일 내용을 캐시에서 읽는 초기 비용이 생긴다.
이 구조는 약 4배인 논리 읽기량과도 일치한다. 초기화 시간을 줄이는 후속 개선에서는
디스크 이동뿐 아니라 읽기 단계의 shard 분할과 shuffle 구성을 검토할 수 있다.
shuffle 변경은 샘플 순서에 영향을 주므로 재현 실험 중 임의로 바꾸지는 않는다.

각 episode의 `steps`를 NumPy로 펼치고, 두 이미지를 PIL/processor로 처리한 뒤 tensor를
쌓는 CPU 작업도 있다. SSD로 옮겨도 이 비용은 남는다. TFDS 4.9.10은 기본 prefetch를
사용하고 PyTorch도 worker당 2 batch를 준비하므로 단순히 prefetch가 없다고 진단하면 안 된다.
이 데이터 크기는 TFDS의 작은 데이터 자동 `.cache()` 기준 250 MiB보다 크다.
이번에 확인한 것은 별개인 OS 페이지 캐시이다.
[PyTorch 2.3 DataLoader](https://docs.pytorch.org/docs/2.3/data.html),
[TensorFlow의 입력 준비·계산 중첩 설명](https://www.tensorflow.org/guide/data_performance).

측정 당시 HDD 전체 장치에는 다른 작업의 I/O도 존재했다. 장치 전체 통계와
측정 프로세스의 I/O 통계를 구별했으며, 독점 환경의 벤치마크는 아니다.
[배치별 시간·캐시·프로세스 I/O 원본](storage_analysis/libero_input_bs64_step64.json).

재측정 명령은 다음과 같다. 이미 존재하는 출력 파일은 덮어쓰지 않으므로 이름을 바꿔 실행한다.

```bash
cd /home/jack/TurboVLA
/home/jack/miniforge3/envs/turbovla/bin/python \
  scripts/libero/benchmark_input_pipeline.py \
  --batch-size 64 --workers 4 --step-mix-buffer 64 \
  --batches 128 --warmup-batches 16 --compute-seconds 0.8 \
  --output experiments/robotwin/storage_analysis/libero_input_repeat.json
```

**5. 캐시가 비어 있다면 어느 정도인가**

OS 캐시를 우회하는 읽기 전용 `O_DIRECT`로 각 suite의 첫·마지막 shard, 총 8개 파일에서
최대 128 MiB씩 읽었다. 4 MiB 블록으로 실제 읽은 양은 808 MiB이며 **7.18초**가 걸렸다.
평균 **112.6 MiB/s**, 파일별 **98.7–139.1 MiB/s**였다. 드라이브 내부 캐시까지 비우는
시험은 아니며, 파일 배치와 다른 작업의 경합도 영향을 준다.
[직접 읽기 결과](storage_analysis/libero_hdd_direct_read.json).

이 속도를 9.528 GiB 전체에 단순 적용하면 **약 87초, 1.5분**이다.
이는 "전체 파일을 한 번 순차 읽는 시간"의 추정이며 첫 batch 시간, 이미지 디코딩 시간,
학습 epoch 시간은 아니다. 실제 cold-start 학습은 여러 worker의 접근 순서 때문에 달라질 수 있다.

장치 제조사 명세의 MG07ACA 12TB sustained 속도는 242 MiB/s이지만 실제 측정값은
현재 파일 위치와 공유 부하를 반영해 더 낮았다.
[Toshiba 공식 제품 매뉴얼](https://www.toshibastorage.com/docs/enterprise-hdd-documents/ehdd-mg07aca-product-manual_r2.pdf).
SSD가 더 높은 순차 읽기 속도를 내더라도 그 비율만큼 전체 학습이 빨라지는 것은 아니다.
첫 읽기 이후에는 페이지 캐시, CPU 전처리, GPU 연산과의 중첩에 따라 병목이 달라진다.

**현재 선택에 적용하면**, RoboTwin은 4.02 GB이므로 SSD에 두기에 충분히 작다.
LIBERO 네 suite도 10.23 GB라 SSD에 복사하기는 부담이 작지만, 관측한 warm-cache 상태에서
학습 속도의 큰 향상을 기대할 근거는 없다. 반복 학습의 추가 최적화는 실제 학습 루프에
`next(data_iter)` 대기와 GPU 구간 시간을 따로 기록한 다음 판단하는 것이 적절하다.
