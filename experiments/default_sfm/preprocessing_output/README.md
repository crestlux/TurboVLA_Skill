# default_sfm v1 실행 메타데이터

이 디렉터리는 실제 출력 bundle의 작은 JSON 파일을 바이트 그대로 보관한 재현 기록이다.
HDF5 데이터나 이미지 사본은 포함하지 않으므로 이 디렉터리를 데이터 로더의 root로 사용하지 않는다.

실제 데이터는 `/data/crestlux/default_sfm_processed/sfm_delta_h5_ftstd001_v1`에 있고,
프로젝트의 `data/default_sfm_preprocessed`가 그 경로를 가리킨다.
원본은 `/data/crestlux/default_sfm` 및 `data/default_sfm`에 별도로 보존한다.

- `manifest.json`: 완료 수량, 450개 shard의 SHA-256, 원본 대응, 참조 일치 여부.
- `normalization.json`: 실제로 적용한 공통 통계. F/T는 standard, 분모 하한 0.01.
- `data_contract.json`, `inference_config.json`: 저장 배열 및 추론 입력 계약.
- `run.json`: 실행 명령, 코드 커밋 `9bebb80966ced629e484f8d7786ad93451d48ecb`, 실제 소스 해시.
- `preprocessing_config.json`: 실행 시작 때 고정한 설정. 내부의 `preprocessing_applied: false`
  같은 진행 상태는 실행 시작 당시 값이며, 완료 여부는 `manifest.json`을 기준으로 판단한다.
- `collection_decoding_contract.json`: 원본 state 열 배치와 수집 설정.

전체 실행법과 출력 해석은 [전처리 재현 안내](../PREPROCESSING_GUIDE.md)에 있다.
원래 사용한 코드와 설정은 실제 bundle의 `source_snapshot/`에도 보존되어 있다.
현재 저장소의 계획 파일은 완료 상태로 갱신하므로 실행 시작 당시 snapshot과 상태 필드가 다를 수 있다.
이 차이는 전처리 수식이나 선택 조건의 변경을 뜻하지 않는다.
