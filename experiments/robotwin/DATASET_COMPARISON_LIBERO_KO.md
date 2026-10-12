# RoboTwin 2.0와 LIBERO 데이터 비교

확인일: 2026-09-30. LIBERO는 이 머신의 TurboVLA 학습에 사용한 4개 `no_noops` RLDS suite,
RoboTwin은 보유한 Clean 및 공개 150k 모델의 Clean+Randomized 학습 데이터를 비교한다.

**시점 수 기준으로 RoboTwin Clean은 우리 LIBERO의 2.01배, Clean+Randomized는 22.22배다.**
디스크 용량은 각각 4.02 GB와 79.62 GB이며, 우리 LIBERO는 10.23 GB다. 영상 압축 방식이 달라
파일 크기 비율과 학습 데이터 시점 수 비율을 구분해야 한다.

## 로봇과 데이터의 성격

| 항목 | LIBERO — 이번 비교 범위 | RoboTwin 2.0 — 이번 비교 범위 |
|---|---|---|
| 로봇 | Franka Panda 단팔 | ALOHA-AgileX 양팔 |
| 시뮬레이터 | MuJoCo / robosuite | SAPIEN |
| 작업 구성 | Spatial·Object·Goal·Long 각 10개 | 50개 작업, 단팔 조작과 양팔 협동 포함 |
| 대표 작업 | 위치로 대상 구분, 다른 물체 집기, 목표 위치에 놓기, 여러 단계 작업 | 양팔로 냄비 들기, 물체 건네기, 도구 사용, 쌓기, 문 열기 |
| 시연 생성 | 사람이 시뮬레이터에서 원격 조작한 시연을 재생·정제 | 전문가 코드와 motion planner로 시뮬레이터에서 생성 |
| 변동 요소 | 공간 관계·대상 물체·목표·초기 배치, 긴 작업 순서 | Clean 외에 배경·조명·주변 물체·테이블 높이·언어를 변화시키는 Randomized |
| 사용 RGB 카메라 | 정면 + 손목, 2개 | 머리 + 왼쪽 손목 + 오른쪽 손목, 3개 |
| 저장된 이미지 크기 | 각 256×256 | 각 640×480 |
| 저장 형식 | RLDS/TFRecord 안에 프레임별 JPEG | LeRobot: AV1 MP4 영상 + Parquet 수치 데이터 |
| action | 7차원: 손끝 위치·회전 변화 6 + 그리퍼 1 | 14차원: 양팔 절대 관절 위치 12 + 그리퍼 2 |
| 관측 state | 8차원 손끝 자세·그리퍼, 별도로 7차원 관절 상태 | 14차원 양팔 관절·그리퍼 |
| 언어 | 기본적으로 태스크별 지시문 | 물체·팔·표현에 따른 지시문 변형, seen/unseen 구분 |

이는 이번에 사용하는 embodiment·변환본의 비교다. RoboTwin 플랫폼은 다른 로봇도 지원한다.
LIBERO 전체 벤치마크는 130개 작업을 포함하며, 여기서는 TurboVLA가 사용하는 40개만 센다.
`libero_90`, 별도 HDF5 원본, 다른 해상도의 중복 변환본은 이번 학습본 합계에서 제외했다.

설계 근거: [LIBERO 공식 저장소](https://github.com/Lifelong-Robot-Learning/LIBERO),
[RoboTwin 공식 프로젝트](https://robotwin-platform.github.io/).
실제 shape·인코딩은 두 데이터의 메타데이터에서도 확인했다.

## 데이터 양

| 항목 | 우리 LIBERO 4개 suite | RoboTwin Clean — 보유 | RoboTwin Clean+Randomized — 공개 150k 학습 범위 |
|---|---:|---:|---:|
| 서로 다른 태스크 | 40 | 50 | 50 |
| 시연 에피소드 | 1,693 | 2,500 | 27,500 |
| 관측·행동 시점 | 273,465 | 549,787 | 6,075,103 |
| 모든 카메라의 RGB 이미지 합계 | 546,930 | 1,649,361 | 18,225,309 |
| 평균 시점 / 에피소드 | 161.53 | 219.91 | 220.91 |
| 파일 크기, GB | 10.23065 | 4.02193 | 79.61541 |
| 파일 크기, GiB | 9.52803 | 3.74572 | 74.14763 |
| 현재 로컬 상태 | HDD에 준비됨 | SSD에 준비됨 | Randomized 학습 데이터는 아직 받지 않음 |

한 시점은 동기화된 관측과 행동 한 쌍이다. 같은 시점의 카메라 영상이 LIBERO에는 2장,
RoboTwin에는 3장 있다. 서로 다른 제어·샘플링 방식의 시점 수이므로 이 비율이 물리적
시연 시간이나 데이터 난이도의 비율을 뜻하지는 않는다.

에피소드 수 기준으로 Clean은 LIBERO의 **1.48배**, 혼합 전체는 **16.24배**다.
관측·행동 시점 기준으로는 **2.01배 / 22.22배**, 카메라별 이미지 합계로는 **3.02배 / 33.32배**다.
혼합 전체의 파일 크기는 우리 LIBERO의 **7.78배**다.

LIBERO 원본의 비교 대상 40개 태스크에는 명목상 50개씩 총 2,000개 시연이 있다.
이번 수치는 실패 시연 등을 제외한 `no_noops` 변환본의 실제 1,693개다.
OpenVLA의 데이터 처리 설명에도 suite별 제외 수가 Spatial 68, Object 46, Goal 72, Long 121개로
기록되어 있으며, 우리 메타데이터의 개수와 일치한다.
[OpenVLA 논문 부록 G.1](https://raw.githubusercontent.com/mlresearch/v270/main/assets/kim25c/kim25c.pdf).

| LIBERO suite | 태스크 | 정제 후 에피소드 | 관측·행동 시점 | TFRecord 크기, GB |
|---|---:|---:|---:|---:|
| Spatial | 10 | 432 | 52,970 | 1.91463 |
| Object | 10 | 454 | 66,984 | 2.81732 |
| Goal | 10 | 428 | 52,042 | 1.84190 |
| Long (`libero_10`) | 10 | 379 | 101,469 | 3.65681 |

## RoboTwin 추가 다운로드 용량

공개 배포 `StarVLA/RoboTwin-Randomized`는 이름과 달리 `Clean/`과 `Randomized/`를 함께 포함한다.
확인한 revision은 `188c8b372b8945621726a70f4a9399047a922009`다.

| 배포 폴더 | 작업당 시연 | 총 에피소드 | 관측·행동 시점 | 크기 |
|---|---:|---:|---:|---:|
| `Clean/` | 50 | 2,500 | 549,787 | 약 4.02 GB |
| `Randomized/` | 500 | 25,000 | 5,525,316 | **75.60 GB / 70.41 GiB** |
| 전체 저장소 | 550 | 27,500 | 6,075,103 | **79.62 GB / 74.15 GiB** |

현재 보유한 Clean에 더 필요한 학습 데이터는 `Randomized/`의 약 75.60 GB다.
이 수치는 학습 데이터 파일 크기이며 simulator assets, 환경, 모델 가중치, checkpoint, 다운로드
임시 공간은 포함하지 않는다. 이번 비교에서는 공개 메타데이터만 읽었다.

Clean 단독 배포와 혼합 배포의 Clean 폴더는 메타데이터 파일 구성이 달라 각각
4,021,934,537 bytes와 4,017,794,085 bytes다. 두 배포의 Clean 에피소드·프레임 수와
영상·Parquet 파일 크기 합은 일치한다. 본문은 보유한 단독 배포 크기를 사용했다.

[확인한 공개 배포](https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized/tree/188c8b372b8945621726a70f4a9399047a922009).

## 학습 시간과 연결해서 해석하기

Clean은 시점 수가 LIBERO의 약 2배여도 디스크에서는 약 39% 크기다.
프레임별 JPEG와 시간 방향 중복을 압축하는 AV1 영상의 차이 때문에, GB 수만으로 학습 계산량을
추정할 수 없다. 디코딩 뒤 실제 모델에 들어가는 이미지와 모델 계산이 학습 비용에 영향을 준다.

또한 이번 TurboVLA 설정은 다음과 다르다. 아래는 데이터셋 자체의 필수 사양이 아니라
우리가 비교하는 모델·학습 recipe의 차이다.

| TurboVLA 설정 | LIBERO | RoboTwin |
|---|---|---|
| 비전 인코더 | DINOv3 ViT-B | DINOv3 ViT-L |
| 모델 입력 | 2개 × 256×256 RGB | 3개 × 224×224 RGB |
| 한 번에 예측하는 행동 | 12개 × 7차원 | 50개 × 14차원 |

따라서 RoboTwin 학습 시간이 긴 원인은 데이터 양과 함께 모델·입력·학습 업데이트 수를 봐야 한다.
고정된 업데이트 수와 배치에서 데이터셋이 22배 크다고 학습 시간이 자동으로 22배 늘지는 않는다.
데이터 규모와 반복 샘플링 횟수는 별개다.

## 집계 근거와 확인 범위

- [통합 비교 JSON](storage_analysis/dataset_comparison_20260930/comparison.json)
- [LIBERO 실제 파일 크기·메타데이터](storage_analysis/dataset_comparison_20260930/libero_local_metadata.json)
- [RoboTwin 공개 100개 태스크 메타데이터와 폴더 크기](storage_analysis/dataset_comparison_20260930/robotwin_remote_metadata.json)
- [보유한 RoboTwin Clean 전체 검증 기록](DATA_READY_KO.md)

LIBERO 에피소드 수는 4개 `dataset_info.json`의 shard 길이를 합산했고, 시점 수는 실제 학습에
사용한 정규화 통계의 집계값을 사용했다. 파일 크기는 96개 TFRecord의 실제 길이를 더했다.
이번에 TFRecord 전체를 재디코딩한 것은 아니다.

RoboTwin 혼합 데이터는 고정 revision에서 100개 `meta/info.json`의 에피소드·프레임·영상 수를
합산하고, 전체 schema를 확인했다. 합계는 공개 150k의 정규화 통계와 일치했다.
용량은 Hugging Face `treesize` API로 확인했다. 일반 dataset-info의 파일 목록은 100,000개에서
잘려 최종 용량 계산에 사용하지 않았다. Randomized 영상이나 Parquet를 새로 다운로드·검증한 것은 아니다.
