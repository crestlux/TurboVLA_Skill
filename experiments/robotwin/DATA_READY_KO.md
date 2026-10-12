# RoboTwin Clean 데이터 준비 완료

2026-09-23에 다운로드와 검증을 완료했다.

- 실제 데이터: `/home/jack/datasets/RoboTwin/Clean`
- 학습 환경 변수: `ROBOTWIN_DATA_ROOT=/home/jack/datasets/RoboTwin`
- 배포 저장소: `StarVLA/RoboTwin-Clean`
- 고정 revision: `070d3b86d7db06f924702a82725dba0f4d89a433`
- 원본 영상·Parquet·메타데이터를 변환하거나 다시 인코딩하지 않았다.

| 확인 항목 | 결과 |
| --- | --- |
| 전체 배포 파일 | **10,404개 전부 다운로드 및 해시 일치** |
| 정확한 파일 크기 합 | **4,021,934,537 bytes = 4.02 GB = 3.75 GiB** |
| 누락 / 크기 불일치 / 해시 불일치 | **모두 0개** |
| 남은 `.incomplete` 파일 | **0개** |
| 작업 / 시연 | **50개 / 2,500개** |
| Parquet / transition | **2,500개 / 549,787개** |
| 영상 / 실제 디코딩한 전체 프레임 | **7,500개 / 1,649,361개** |
| 영상 형식 | AV1, 640×480, 15 fps, 시연당 카메라 3개 |
| state / action | 모두 14차원, 전체 행에서 유한값 확인 |

파일 다운로드 직후 검사에 더해, 완료 후 전체 파일을 다시 읽어 배포 manifest의
SHA-256 또는 Git blob SHA-1과 대조했다. 모두 일치했다.
각 시연의 episode/frame index, timestamp, instruction index와 메타데이터의 정합성도 확인했다.

모든 영상은 FFmpeg의 CPU AV1 디코더로 끝까지 읽었고, 실제 디코딩 프레임 수가
해당 Parquet 행 수와 일치했다. 별도로 **TurboVLA의 실제 `get_frames_by_timestamps()`**를
`pyav` backend로 호출해 50개 작업의 카메라 3개씩, 총 150개 영상에서
처음·중간·마지막 프레임 450개를 읽었다. 모두 정상 동작했다.

PyAV는 프로젝트가 지정한 `12.3.0`을 별도 검증 폴더에 설치해 사용했다.
기존 `turbovla` 학습 환경의 설치 패키지는 변경하지 않았다.
RoboTwin 학습 환경을 준비할 때는 [재현 가이드](REPRODUCE_KO.md)의 환경 설치 단계와
`video_backend: pyav` 설정을 따른다. 모델 입력의 224×224는 전처리 후 크기이며,
여기 보관한 원본 영상은 640×480이다.

학습 명령을 실행할 터미널에서 다음 경로를 설정한다. `Clean`의 상위 디렉터리이다.

```bash
export ROBOTWIN_DATA_ROOT=/home/jack/datasets/RoboTwin
```

완료 상태와 검증 근거:

- [준비 완료 정보](storage_analysis/robotwin_data_ready.json)
- [전체 파일 무결성 검사](storage_analysis/robotwin_download_integrity.json)
- [50개 작업의 Parquet·전체 영상 검사](storage_analysis/robotwin_data_validation.json)
- [실제 TurboVLA 영상 읽기 검사](storage_analysis/robotwin_runtime_reader_validation.json)
- 데이터 디렉터리의 완료 표시: `/home/jack/datasets/RoboTwin/READY.json`
- 다운로드 manifest·로그·상세 보고서: `/home/jack/datasets/RoboTwin/.download/`

작업별 영상 검사 보고서의 `verified_files=10403`은 50개 작업 디렉터리 안의 파일 수이다.
루트의 `.gitattributes` 1개까지 포함한 최종 무결성 검사는 **10404개 전부 통과**했다.

검사를 다시 수행하려면 다음 명령을 사용한다. 다운로드 없이 파일을 읽어 해시만 확인한다.

```bash
cd /home/jack/TurboVLA
/home/jack/miniforge3/envs/turbovla/bin/python scripts/robotwin/download_clean_verified.py \
  --manifest /home/jack/datasets/RoboTwin/.download/manifest-070d3b8.json \
  --local-dir /home/jack/datasets/RoboTwin/Clean \
  --status /home/jack/datasets/RoboTwin/.download/integrity_recheck.json \
  --verify-only
```
