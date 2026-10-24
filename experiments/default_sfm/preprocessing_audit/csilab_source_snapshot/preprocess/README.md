# Config-driven episode preprocessing

YAML로 observation key 매핑, 이미지 처리, state 차원 선택, 초반 정지 제거,
action chunk(delta/relative/observed/residual), state/action 정규화를 제어합니다.
에피소드마다 HDF5 하나를 저장하고, 출력 폴더에 inference용 config와 통계를 함께 저장합니다.
원본 데이터와 로봇 제어 설정은 수정하지 않습니다.

## Dataset Studio: 데이터 탐색과 시각적 전처리 설정

```bash
cd /home/csilab/share_projects/clean_share
./scripts/dataset_studio.sh
# 브라우저에서 http://127.0.0.1:8130 열기

# 다른 설정 / 포트
./scripts/dataset_studio.sh --config preprocess/configs/config_osj.yaml --port 8131
```

기본 설정은 `preprocess/configs/config_right_default.yaml`이며 최초 탐색 경로는
`/media/csilab/robot_data`입니다. `--start-path /path/to/data`로 시작 폴더를 바꿀 수 있습니다.
`--config /path/to/preprocessing.yaml`을 명시하면 브라우저를 열 때 해당 설정과
`input.path`의 데이터셋을 함께 불러와 자동 분석합니다. 이 경우 탐색기도 입력 경로로 이동합니다.
현재 실행 스크립트는 `umi` Python을 우선 사용하며 없으면 `hilserl`을 사용합니다. 다른 환경에서는
`PYTHON=/path/to/python ./scripts/dataset_studio.sh`로 지정하세요.
추가 설치가 필요한 경우 해당 Python에서
`python -m pip install -r preprocess/requirements-studio.txt`를 실행합니다.
프론트엔드는 빌드 단계나 CDN 없이 동작합니다.

1. **탐색·분석:** 왼쪽 경로창에 폴더를 입력하거나 폴더 목록을 클릭하고 **이 폴더 분석**을 누릅니다.
   원본 `.pkl`과 전처리 `.hdf5`/`.h5`를 함께 지원합니다. 파일을 직접 클릭하면 그 파일만 분석합니다.
   왼쪽 탐색기에는 `.yaml`/`.yml` 설정 파일도 표시됩니다. 톱니 아이콘의 YAML을 클릭하면
   **Config + 데이터 불러오기**와 동일하게 설정과 `input.path`의 데이터셋을 함께 불러옵니다.
   파일 수, 에피소드 수, 프레임 수, 크기, 에피소드 길이를 표시합니다. 손상된 파일은 오류 목록에 남기고
   나머지 파일 분석을 계속합니다. 원본은 이 프로젝트의 transition list / episode dict 형식,
   HDF5는 `preprocess.convert` 출력 스키마를 지원합니다.
   **첫 에피소드가 준비되면 즉시 표시하고**, 나머지 파일은 백그라운드에서 계속 분석해 목록에 추가합니다.
   분석 중에도 준비된 에피소드 선택·이미지 확인·재생·설정 편집을 할 수 있습니다.
   진행 표시 옆의 **백그라운드 분석 중지**는 현재 파일을 읽은 뒤 중지하며, 확인된 목록을 유지합니다.
   변경되지 않은 파일의 메타데이터는 서버 세션에서 캐시하고 파일 크기·수정 시간이 바뀌면 다시 읽습니다.
2. **에피소드 시각화:** 에피소드·프레임을 선택하면 카메라, 원본 관측 TCP의 XY/XZ 궤적,
   state/action/reward/history 신호와 통계를 확인할 수 있습니다. TCP의 **3D XYZ 궤적**은 드래그 회전,
   Shift+드래그/오른쪽 드래그 이동, 휠 확대, 시점 초기화를 지원합니다. 축 비율은 XYZ가 동일하며
   시작·끝·현재 프레임을 표시합니다. 3D 궤적을 클릭하면 가까운 표시 지점의 프레임으로 이동합니다.
   원본/미리보기/저장 결과에 동일하게 적용하며 XY/XZ 그래프도 함께 유지합니다.
   그래프 클릭·슬라이더·방향키로
   프레임을 이동하며 Space로 재생/정지합니다. 기본은 기록된 제어 주기(없으면 10Hz)의 **1배속**이며
   0.5·1·2·4·8배속을 선택할 수 있습니다. 고배속에서는 경과 시간에 맞춰 프레임을 건너뛸 수 있습니다.
   선택한 에피소드의 재생용 이미지는 최대 8프레임씩 미리 읽고, 브라우저 캐시는 최대 48프레임입니다.
   재생용 이미지는 긴 변 최대 640px JPEG이며, 정지/설정 편집 때는 원본 해상도의 PNG를 읽습니다.
   다른 에피소드의 카메라는 선택 시 로딩하므로 전체 에피소드 이미지를 미리 인코딩하지 않습니다.
   수치 그래프는 최대 1,200프레임을 균등 샘플링하고, 통계는 표시 샘플 기준임을 화면에 표기합니다.
   저장된 chunk는 현재 프레임의 horizon별 값과 유효 mask/padding을 추가로 표시합니다.
3. **기존 config와 데이터 불러오기:** 오른쪽 상단 **전처리 config 파일**에 서버에 있는 기존
   전처리 YAML의 경로를 입력하고 **Config + 데이터 불러오기**를 누릅니다.
   `input.path`의 파일/폴더를 자동 분석하며 `input.recursive`, 관측·이미지·action·정규화 설정과
   `output.directory`를 함께 반영합니다. YAML 안의 상대경로는 해당 YAML 파일이 있는 폴더 기준입니다.
   이전 분석이 진행 중이면 중지 후 새 데이터셋으로 전환하며, 첫 에피소드가 준비되면 표시합니다.
   입력/참조 파일 경로가 잘못되면 현재 설정과 데이터셋을 유지하고 오류를 표시합니다.
   Preset 선택은 설정만 바꾸고, 이 버튼은 데이터셋도 함께 전환합니다.
   **입력·설정:** 수동 탐색으로 분석할 폴더와 전처리 입력은 별개입니다. **전처리 입력으로 지정**을 누르거나
   오른쪽 **입출력** 탭에서 원본 파일/폴더를 지정하세요. state 채널·성분 선택, action 모드/길이,
   정규화, 초반 정지 제거, 압축, next observations 저장을 각 탭에서 설정할 수 있습니다.
   수집 config가 없는 이동된 기록은 `input.collection_config`를 지정하세요.
   **Action → Gripper label 재구성**에서 absolute/relative/binary와 파지 추정 임계값을 설정합니다.
   원본을 선택한 뒤 **파지 구간 · label 갱신**을 누르면 `gripper_labels`와 `gripper_grasp_mask`를
   시계열에서 비교할 수 있습니다. **정규화 → xyz / rpy / gripper 개별 정규화**로 각각 방법을 고릅니다.
4. **이미지 설정:** 원본 위를 드래그하여 crop을 지정합니다. 숫자 입력, 중앙 정사각형, crop 해제,
   회전 각도 입력과 시계/반시계 90°·180°·초기화 버튼, 회전 영역 확장, resize 크기,
   stretch/letterbox, 보간법, RGB/BGR 선택도 지원합니다. 처리 순서는 **crop → 회전 → resize**입니다.
   이미지 비교는 변환기의 `image_sample()`을 직접 호출하므로 저장과 동일한 픽셀 처리를 사용합니다.
   전처리 HDF5를 볼 때는 저장된 RGB를 그대로 표시합니다.
5. **실제 미리보기:** **현재 설정으로 미리보기**는 원본을 실제 전처리하여 임시 HDF5를 만들고
   결과 탭을 엽니다. 정지 제거·state 선택·action chunk·정규화까지 적용합니다.
   기본 범위는 **마지막으로 선택한 원본 파일 전체(그 파일의 모든 에피소드)**이며,
   **입력 경로 전체**를 선택하면 폴더 전체를 대상으로 통계를 계산합니다.
   선택 파일로 fit한 통계는 전체 데이터셋 통계와 다를 수 있습니다. `stats_file`을 지정한 경우에는
   기존 통계를 재사용합니다. 다음 미리보기 실행 또는 서버 종료 시 임시 결과를 삭제합니다.
6. **저장:** 결과 경로를 지정하고 **전처리 결과 저장**을 누릅니다. 저장은 미리보기 범위와 별개로
   **현재 input.path 전체**를 처리합니다. 새 폴더 또는 빈 폴더만 허용하며,
   YAML의 `overwrite` 값과 관계없이 기존 결과를 덮어쓰지 않습니다.
   에피소드별 HDF5, `config.yaml`, `env_config.json`, `normalization.json`,
   `inference_config.json`, `manifest.json`을 함께 저장합니다.
   저장 결과 탭에는 실제 저장 폴더를 가져오는 SCP 명령과 **명령 복사** 버튼을 표시합니다.
   가져올 컴퓨터에서 실행하며 마지막 `.`은 현재 폴더입니다. 명령 표시·복사만 하고 SSH 접속은 하지 않습니다.

   ```bash
   scp -r csilab@115.145.230.48:/media/csilab/robot_data/preprocessed/my_dataset .
   ```
7. **설정 보관:** **YAML** 탭에서 전체 설정을 검증·적용하거나 **설정 + 데이터 불러오기**로
   다른 YAML과 그 데이터셋을 함께 불러올 수 있습니다.
   **설정 파일 저장**은 새로운 경로에만 저장합니다. 경로 필드는 절대경로로 보관하여 YAML을
   다른 디렉터리에 저장해도 경로 해석이 달라지지 않습니다. 기존 `config_osj.yaml`은 수정하지 않습니다.
   오른쪽 상단 **전처리 preset**에서 config를 선택하면 현재 원본 데이터 경로(`input.path`)를 유지하고 설정을 적용합니다. 수정 후 이름을 입력하고
   **저장**하면 시작 config 디렉터리의 `presets/<이름>.yaml`로 보관되고 선택 목록에도 즉시 추가됩니다.
   프로젝트 config 디렉터리와 시작 config 옆의 YAML·`presets/`를 목록에 표시하며,
   잘못된 설정은 오류로 표시하고 선택하지 못하게 합니다. 같은 이름은 덮어쓰지 않으므로 새 이름을 사용하세요.

서버는 `127.0.0.1`에서 동작하며 외부 서비스를 사용하지 않습니다. 기본 허용 경로는 홈과
프로젝트 `demos`의 실제 경로입니다. 현재 머신의 `demos`는 `/media/csilab/robot_data`에 연결되어
있으므로 해당 디스크도 자동으로 탐색할 수 있습니다. 다른 디스크를 추가하려면 루트를 모두 명시하세요.

```bash
./scripts/dataset_studio.sh \
  --root /home/csilab --root /media/csilab/robot_data --root /path/to/other_disk
```

자체 수집한 신뢰할 수 있는 pickle만 사용하세요. 원본 pickle은 선택한 파일 하나 전체를
메모리에 읽고 캐시하므로 **첫 pickle 파일 자체의 역직렬화 시간은 필요합니다**.
폴더 분석은 파일별로 순서대로 수행하며 선택된 파일과 백그라운드 분석 파일이 메모리에 함께 있을 수 있습니다.
분석과 전처리 작업은 동시에 실행하지 않으며, 백그라운드 분석 중 조회·재생은 가능합니다.
첫 에피소드가 나오기까지 기다리는 시간과 전체 분석이 끝나는 시간을 분리합니다.
변환 중에도 진행 상태를 볼 수 있습니다.
미리보기 역시 실제 변환이므로 임시 디스크 공간이 필요합니다. 종료는 서버 터미널에서 Ctrl+C입니다.

새 수집 config에서는 `image_preprocess` 항목을 생략하고, 카메라 해상도의 RGB 이미지를
전처리 단계에서 처리합니다. Crop과 출력 크기는 image observation 설정에서 지정합니다.
기존 모델 호환을 위한 환경의 `image_preprocess`는 비지원 legacy 기능으로 남겨 두며 추후 제거합니다.
항목이 있는 config는 예전 처리 순서를 유지하고, `enable: false`도 224×224 stretch resize와
카메라 이름에 따른 자동 회전을 유지합니다. 기존 데이터는 이미 환경 처리가 적용된 이미지입니다.

검증: `python -m unittest discover -s tests -p 'test_preprocess*.py' -v`.
Studio 통합 테스트는 프레임 crop 픽셀, CLI와 미리보기의 HDF5 값 일치, 정규화·mask,
설정 저장, 기존 출력 보호, 경로 제한, 작업 직렬화 등을 검사합니다.

## 실행

```bash
cd /home/csilab/share_projects/clean_share
./scripts/preprocess_demos.sh --config preprocess/configs/config.yaml

# 먼저 파일 하나의 crop과 출력 shape 확인 (파일 미생성)
./scripts/preprocess_demos.sh --input /path/to/episode.pkl --dry-run

# 특정 폴더를 하나의 정규화 데이터셋으로 처리
./scripts/preprocess_demos.sh --input /path/to/train --output-dir /path/to/processed_train
```

YAML 경로는 YAML 파일 위치 기준, CLI 경로는 현재 디렉터리 기준입니다.
기존 결과는 기본적으로 덮어쓰지 않습니다. `--overwrite`는 데이터셋 전체를 다시 처리할 때 사용합니다.
기존 출력에 이번 입력에 없는 HDF5가 남는 경우, 서로 다른 통계의 데이터가 섞이지 않도록 거부합니다.
그 경우 새 출력 디렉터리를 지정하세요. 출력 이름은 원본 이름 + 경로 해시 + episode 번호입니다.

현재 머신의 실행 스크립트는 h5py 등이 설치된 `umi` Python을 사용합니다.
`PYTHON=/path/to/python ./scripts/preprocess_demos.sh ...`로 변경할 수 있습니다.
의존성은 `requirements.txt`에 있으며, quaternion의 `scalar_first`를 지원하는 SciPy가 필요합니다.
로봇/ROS/카메라를 실행하거나 연결하지 않습니다. 자체 기록한 신뢰할 수 있는 pickle만 사용하세요.

## Observation와 pose 차원 선택

각 항목의 `source`는 원본 observations key, `target`은 저장 key입니다.
`type: state`는 선택 채널을 연결하고, `image`는 RGB 이미지 처리, `array`는 history 등의 숫자 배열을 보존합니다.
같은 처리가 next observations에도 적용됩니다 (`output.save_next_observations`로 선택).

```yaml
observations:
  - source: state
    target: state
    type: state
    keys:
      - right/gripper_pose
      - key: right/tcp_pose
        components:
          x: true
          y: true
          z: true
          roll: false
          pitch: false
          yaw: true
```

이 예제는 `[gripper,x,y,z,yaw]` 5차원을 저장합니다. Boolean mapping은 채널의 원래 성분 순서를 따르고,
명시하지 않은 성분은 제외합니다. `components: [z,x,yaw]`처럼 목록으로 순서를 직접 지정할 수도 있습니다.
`components: [0,2,5]` 정수 인덱스도 지원합니다. 기존 `- right/tcp_pose` 문자열은 전체 6차원을 사용합니다.
다른 state 채널에도 같은 선택법이 적용됩니다. 최종 열 목록은 HDF5 attrs와 normalization schema에 남습니다.

TCP pose의 회전 관측이 `+π → -π`로 점프하는 경우, `wrap_angles`에 지정한 축을
**`[0, 2π)` rad 범위**로 변환할 수 있습니다. Dataset Studio의 **관측** 탭에서도
각 `roll`·`pitch`·`yaw`의 범위를 **원본 유지 / 0 ~ 2π rad**로 선택할 수 있습니다.

```yaml
observations:
  - source: state
    target: state
    type: state
    keys:
      - left/gripper_pose
      - key: left/tcp_pose
        components: [x, y, z, roll, pitch, yaw]
        wrap_angles: [roll, yaw]
```

이 예제에서는 `roll`과 `yaw`만 `angle % (2π)`로 변환하고 나머지 성분을 유지합니다.
예를 들어 `179° → -179°`는 `179° → 181°`에 해당하는 라디안 값으로 저장됩니다.
선택해서 유지한 `tcp_pose` 회전 성분에만 지정할 수 있으며, 생략하거나 빈 목록이면 기존 값을 유지합니다.
Boolean mapping과 정수 인덱스 성분 선택에도 함께 사용할 수 있습니다.

처리 순서는 **성분 선택 → 각도 범위 변환 → state 정규화**입니다.
current/next observations와 `InferencePreprocessor.observation()`에 동일하게 적용하며,
프레임별 변환이라 이전 프레임이나 에피소드 상태가 필요하지 않습니다.
설정은 HDF5 metadata와 inference bundle에 보존되고, 범위 설정이 다른 정규화 통계의 재사용은 거부합니다.
표준화를 켜면 최종 저장 값은 `[0, 2π)` 자체가 아니라 변환 후 표준화한 값입니다.
기존 데이터셋에는 소급 적용하지 않으므로 이 옵션을 사용할 학습 데이터는 다시 전처리해야 합니다.

이 옵션은 불연속 경계를 **`±π`에서 `0/2π`로 옮깁니다**.
`0` 근처를 지나는 축에는 새로운 점프가 생길 수 있으므로 축별로 선택하세요.
시간축 `unwrap`이나 Euler 표현 자체의 특이점 처리는 수행하지 않습니다.
원본 관측, 정지 구간 판단용 pose, action 생성·좌표 변환에는 원래 각도를 사용합니다.

### 회전 관측 표현 선택

`tcp_pose` 선택 항목의 `rotation_representation`으로 회전 3개 성분을 통째로 변환합니다.
기본값은 `rpy`이며, Studio의 **관측 → 회전 관측 표현**에서 변경할 수 있습니다.

```yaml
observations:
  - source: state
    target: state
    type: state
    keys:
      - key: right/tcp_pose
        components: [x, y, z, roll, pitch, yaw]
        rotation_representation: rotation_6d
      - right/gripper_pose
```

| 표현 | 회전 차원 | 의미와 제약 |
|---|---:|---|
| `rpy` | 3 | 기존 Euler xyz(rad). 축별 `wrap_angles` 지원, 경계와 gimbal lock은 남음 |
| `gravity` | 3 | `R.T @ [0,0,-1]` 단위벡터. Base Z가 위쪽이라는 가정 아래 TCP 축으로 본 중력 방향 |
| `quaternion` | 4 | WXYZ 순서, canonical 부호(w 양수, w=0이면 첫 nonzero 벡터 성분 양수) |
| `rotation_6d` | 6 | 회전행렬의 첫 두 열을 열 단위로 연결 `[r00,r10,r20,r01,r11,r21]` |
| `rotation_matrix` | 9 | 회전행렬을 행 단위로 연결 `[r00,r01,r02,r10,...,r22]` |

중력 방향은 IMU 가속도 값을 읽는 기능이 아니라 기록/실시간 TCP 자세에서 계산한 **방향**입니다.
Yaw에는 불변이므로 전체 자세를 복원할 수 없습니다. 일반적인 Euler 범위에서 roll/pitch와 같은
기울기 2자유도를 나타내지만 각도 좌표가 아니며, pitch=±90° 등에서는 roll 정보도 구분되지 않습니다.
전체 자세가 필요한 작업에는 gravity 대신 6D/quaternion/행렬을 사용하세요.

Quaternion은 `q`와 `-q`가 같은 회전을 나타냅니다. 부호를 통일해도 180° 부근 경계는 남습니다.
6D와 9D는 같은 회전에 같은 값을 주는 연속적인 관측 표현으로 각도 wrap과 quaternion 부호 문제를
피합니다. 이 구현은 유효한 R에서 관측을 만들며, 모델이 6D/9D를 직접 **출력**하는 용도라면 별도의
직교화(SO(3) 복원)가 필요합니다. 6D 설명과 근거: [Zhou et al., CVPR 2019](https://zhouyisjtu.github.io/project_rotation/rotation.html).

Non-RPY 표현은 `components`에 roll/pitch/yaw를 모두 포함해야 하며 `wrap_angles`와 함께 쓸 수 없습니다.
선택한 xyz는 유지하고 첫 회전 성분 위치에 변환한 회전 block을 삽입합니다. Metadata의 state column
이름·차원, 현재/다음 관측, 정규화 통계, 추론 입력 모두 같은 규칙을 따릅니다. 표현을 바꾸면 새로
전처리·통계 계산·학습해야 하며 다른 표현의 통계를 재사용하면 오류가 발생합니다.
이 옵션은 observation만 바꾸며 action 회전은 계속 3D rotvec(rad)입니다.

**모델 입력에서 pose를 제외해도** crop 판단과 action 좌표 변환에는 원본 6D pose를 사용합니다.
원본 state 열은 수집 config의 `sorted(train.proprio_keys)`로 해석합니다.
`input.collection_config` 명시 → pickle 내 config → 경로 env_name의 프로젝트 config 순으로 선택합니다.
수집 config가 변경됐으면 당시 config를 명시하세요. 같은 폭의 열 의미 변경은 자동 감지할 수 없습니다.

이미지는 `input_color: rgb`가 기본이며, 이미 RGB인 데모의 R/B를 뒤집지 않습니다.
`crop: [top,left,height,width]` → 회전 → `resize: [height,width]` 순서로 적용합니다.
Crop 좌표는 회전 전 원본 이미지 기준입니다.
`rotation_deg`는 임의 각도(도 단위)이며 양수는 반시계, 음수는 시계 방향입니다. 기본값은 0입니다.
`rotation_expand: true`(기본)는 회전 결과 전체를 담도록 영역을 확장하고 빈 영역을 검은색으로 채웁니다.
`false`이면 crop 이후의 크기를 유지하며 가장자리가 잘릴 수 있습니다.
확장한 90°/270° 회전은 높이·너비가 바뀝니다. 이후 resize가 있으면 최종 크기는 resize 설정을 따릅니다.
90° 배수 회전은 출력 크기가 허용하는 경우 보간 없이 픽셀을 재배열합니다.
임의 각도에는 선택한 보간법을 사용하며 `area`를 선택한 경우 회전에는 `linear`, resize에는 `area`를 사용합니다.

```yaml
- source: wrist_cam_d405_a
  target: image_wrist_right
  type: image
  input_color: rgb
  crop: null
  rotation_deg: -90       # 시계 방향 90도
  rotation_expand: true
  resize: [224, 224]
  resize_mode: letterbox
  interpolation: area
```

회전 설정은 현재/다음 관측, Studio 미리보기·저장, `InferencePreprocessor`에 동일하게 적용됩니다.
회전 옵션이 없는 기존 YAML은 0도 회전으로 동작합니다.
`resize_mode: stretch` 또는 비율을 유지하는 `letterbox`(검은 여백),
`interpolation: area/linear/nearest/cubic`를 지정할 수 있습니다.
원본 이미지는 uint8 RGB로 저장하며 무작위 augmentation은 하지 않습니다.

관측 horizon이 있으면 마지막 관측을 택합니다. `sample_ndim: 2`인 history는
`[obs_horizon,N,6] → [N,6]`이며 샘플 순서를 보존합니다. 샘플별 timestamp가 없는 history를
임의로 100Hz로 resample하지 않습니다. `type: array`인 F/T history는 별도의 `normalization.ft` 설정으로 정규화합니다.

## Action chunk: delta / delta_flange / relative / relative_observed / relative_observed_residual

```yaml
actions:
  mode: delta  # delta / delta_flange / relative / relative_observed / relative_observed_residual
  length: 50
  source_format: auto  # auto / absolute / delta / delta_flange
```

- `auto`: 기록의 action_format으로 absolute TCP, world delta 또는 flange delta를 확인합니다. 형식 미표기/혼합은 오류입니다.
- `absolute`: 각 timestep의 **관측 pose**와 절대 목표를 비교해 world delta로 변환합니다.
- `delta`: 기존 world delta `[dx,dy,dz,rx,ry,rz,gripper]`를 그대로 사용합니다.
  명시적 설정은 기록 metadata보다 우선하므로 데이터가 실제로 이 형식인지 확인해야 합니다.
- `delta_flange`: 원본 gripper와 각 step의 flange-local translation/rotvec를 해석합니다.
- 회전은 m/rad의 **rotvec**, pose 입력은 xyz + XYZ Euler(m/rad)입니다.
  Euler 각도 차이를 delta 회전으로 사용하지 않습니다.

`delta` chunk는 이렇게 얻은 timestep별 delta를 미래 방향으로 묶습니다.
`relative` chunk는 각 미래 timestep의 동일한 목표를 chunk 시작 pose의 TCP-local 좌표로 나타냅니다.
단순 delta 누적합이 아니며, 실제로 기록된 미래 관측 pose를 사용합니다.

관측 pose를 `(p_j,R_j)`, world delta를 `(dp_j,dr_j)`, chunk 시작을 t라 하면:

```text
목표:       p*_j = p_j + dp_j
            R*_j = Exp(dr_j) R_j
relative:   position_j = R_t^-1 (p*_j - p_t)
            rotation_j = Log(R_t^-1 R*_j)
```

진행도 추적/보간/속도 제한 등을 제외하고 해당 미래 관측 pose에서 비교하면 원래 delta가 재현됩니다.
실제 로봇의 추적 오차까지 같다는 뜻은 아닙니다.

기본 Gripper는 원본 command를 유지하고 재라벨링/누적하지 않습니다.
`actions.gripper`를 지정하면 아래의 gripper label 처리를 적용합니다.
Relative inference에는 반드시 `relative_gripper_action_mode="absolute"`를 사용합니다.
저장된 `interpreter_options()`가 이 값을 포함합니다. 모든 action은 왼팔→오른팔 순서이며,
기존 모드와 `relative_observed`는 단일팔 7차원, 양팔 14차원입니다.
`relative_observed_residual`은 단일팔 13차원, 양팔 26차원입니다.

끝부분에서 부족한 chunk는 **정규화 후 0 padding**하고 `action_mask[T,H]`에 false를 기록합니다.
다음 episode로 이어붙이지 않습니다. 학습 loss에는 mask를 적용하세요.
저장 chunk를 재생할 경우 유효 길이만 역정규화/전달하세요. Padding 자체는 로봇 hold 명령이 아닙니다.

### 관측 pose 기반 relative label

두 새 모드는 모두 **chunk 시작 시점 t의 TCP 좌표계**를 고정합니다. Horizon k의 transition은
j=t+k이며, `relative_observed`의 목표는 그 명령 이후의 관측 pose `(p⁺_j, R⁺_j)`입니다.
현재 pose를 첫 목표로 복사하지 않으므로 첫 항목도 실제 관측 변화량입니다.

```text
observed_xyz[t,k]    = R_t.T @ (p⁺_j - p_t)
observed_rotvec[t,k] = Log(R_t.T @ R⁺_j)
```

다음 관측은 transition의 명시된 `next_observations`를 우선 사용하고, 없으면 같은 episode의
다음 `observations`, 마지막에는 `final_observation`을 사용합니다. 마지막 다음 관측이 없으면
`save_next_observations: false`여도 오류입니다. 다른 episode의 관측이나 명령 pose로 대체하지 않습니다.
Pose는 모델 입력 state 선택/회전 표현/정규화에 관계없이 원본 전체 TCP pose에서 읽습니다.

```yaml
actions:
  mode: relative_observed
  length: 50
```

`relative_observed`는 명령 목표를 해석하지 않으므로 action format 선언이 없어도 사용할 수 있습니다.
Contract의 `source_format`은 `not_used`이며 `delta_actions` 진단 데이터는 생성하지 않습니다.
Gripper 원본 command는 그대로 사용하며, `actions.gripper`가 있으면 기존 gripper 처리를 적용합니다.

### Observed + residual 분해

```yaml
actions:
  mode: relative_observed_residual
  length: 50
  source_format: auto
```

기존 `relative`의 명령 목표 label을 C, 위 관측 label을 O라 할 때 **정규화 전** pose 6채널에서
E=C−O로 정의합니다. 이 모드는 명령 목표도 필요하므로 기존과 같은 source format 검증을 수행합니다.

```text
팔당 저장 순서:
[observed_xyz(3), observed_rotvec(3), residual_xyz(3), residual_rotvec(3), gripper(1)]

실행 label:
[observed_xyz + residual_xyz, observed_rotvec + residual_rotvec, gripper]
```

Rotation residual은 두 rotvec label의 성분 차이입니다. 독립적인 상대 회전이나 Euler 차이가 아니며,
회전을 별도로 합성하지 않습니다. ±π 분기 부근에서는 residual이 클 수 있지만 합산 결과는 기존
relative label을 복원합니다. Gripper는 residual을 만들지 않고 팔당 한 번만 저장합니다.

두 모드 모두 interpreter mode는 `relative_tcp`입니다. Residual 출력은 반드시 역정규화한 뒤
`pp.compose_actions()`로 팔당 7채널을 복원하고 gripper adapter와 interpreter에 전달하세요.
RPC의 `PolicyIO.physical_actions()`와 HDF5 dry-run/데이터셋 모니터/replay 경로는 이를 자동 처리합니다.
HDF5 `actions.attrs['columns']`와 action contract에 채널 이름과 분해 규칙을 저장하고,
Studio에서는 observed/residual 이름과 residual 점선으로 구분합니다.

### 매 step의 flange 기준 delta

`actions.mode: delta_flange`는 각 step의 **관측 자세**를 기준으로 계산한 delta를 그대로 묶습니다.
Chunk 시작 frame을 고정하는 `relative`와 다릅니다. `p[t], R[t]`가 해당 step의 관측 TCP pose,
`p_target[t], R_target[t]`가 원본 command의 절대 목표일 때:

```text
dp_flange[t] = R[t].T @ (p_target[t] - p[t])
dr_flange[t] = rotvec(R[t].T @ R_target[t])
chunk[t]    = [delta_flange[t], delta_flange[t+1], ...]
```

이 프로젝트 backend와 동일하게 TCP 축이 flange 축에
정렬된 native pose를 사용합니다. 별도 tool 축 회전이나 TCP↔flange 원점 이동을 적용하는 기능은 아닙니다.
기존 world delta 원본도 관측 pose에서 절대 목표로 복원한 뒤 flange delta로 변환할 수 있습니다.
Flange delta 원본은 `source_format: delta_flange` 또는 대응 action_format으로 선언합니다.

저장한 bundle을 `env_rpc_client.py --preprocess-bundle ...`에 전달하면 interpreter mode가
`delta_flange`로 설정되고, 매 실행 step의 **현재 측정 자세**에서 `p_target=p+R@dp`,
`R_target=R@Exp(dr)`로 복원합니다. 지연으로 지난 label은 기존 delta와 같은 방식으로 제거합니다.
환경 `train.hz`는 저장된 action period와 일치해야 합니다. Bundle 없이 사용할 때는 환경 JSON에
`train.action_chunk_mode: delta_flange`, 서버 frame에 `flange`/`flange_local`을 사용합니다.
RPC monitor의 virtual path와 HDF5 episode replay도 이 모드를 지원합니다.
회전 비활성화 옵션은 local rotvec만 0으로 만들며 translation과 gripper는 유지합니다.
기존 world-delta Action Adapter는 flange-local label에 적용할 수 없으므로 이 모드에서는 비활성화해야 합니다.

## Gripper label: absolute / relative / binary

TCP의 `actions.mode: delta/relative`와 독립적으로 `actions.gripper.mode`를 설정합니다.
`actions.gripper`가 없으면 기존 gripper command를 그대로 유지합니다.

| mode | Label (정규화 전) | 파지 추정 구간 |
|---|---|---|
| absolute | 다음 관측 state `g[t+1]` | `g[t+1] - offset` |
| relative | 다음 관측 state − 현재 관측 state `g[t+1]-g[t]` | 변화량이 deadband 이내이면 `-offset` |
| binary | command 증가 `+1`, 감소 `-1`, deadband 이내 `0` | 별도 offset 없음 |

새 설정(`label_version: 2`)의 `g`는 **개폐율 0~1**입니다. 0은 닫힘, 1은 열림입니다.
Offset 0.02는 전체 개폐 폭의 2%이며 UI에는 2%로 표시합니다. Command의 부호/미터 단위와 무관합니다.
Absolute/relative는 transition의 명시된 `next_observations`를 우선 사용하고, 없으면 같은 에피소드의
다음 observation, 마지막 프레임에서는 `final_observation`을 사용합니다. 마지막 관측이 없으면
오류를 내며 다음 에피소드로 넘어가거나 현재 관측을 대신 사용하지 않습니다.
`output.save_next_observations: false`여도 state 기반 label을 만들려면 다음 관측이 필요합니다.
Relative는 첫 프레임에도 실제 다음 state와의 차이를 사용하므로 0으로 고정되지 않습니다.
Binary만 command 차이를 사용하며 에피소드 첫 프레임은 0입니다.
에피소드마다 이전 command와 파지 판정 누적을 초기화하며, 초반 정지 구간을 잘라도
현재/다음 관측의 대응과 같은 에피소드의 직전 command를 보존합니다. Chunk는 timestep label을
미래 방향으로 묶으며 horizon 시작마다 변화량을 다시 계산하지 않습니다.
양수 값은 열림, 더 작은 값은 닫힘을 뜻합니다. `offset`에는 양수 크기를 입력하며 닫힘 방향으로 뺍니다.
끝값에서도 offset이 유지되도록 학습 label은 clamp하지 않습니다. 실행 목표는 물리적 개폐 범위로 제한합니다.

파지는 **관측이 닫히다가 정지하고, 다시 열리기 전까지**의 구간으로 추정합니다.
정지 확인 시간 동안 관측 개폐율의 최대−최소 차이가 허용량 이내여야 합니다.
확인 범위는 현재 시점 기준 최근 `stationary_duration_s` 시간이며, 샘플 간격이 불규칙하면
그 시간을 덮는 경계 샘플 하나를 포함합니다. 파지 유지 구간의 최대 길이 제한은 없습니다.
TCP 위치·자세 이동은 이 판정에 사용하지 않습니다.
프레임 간 차이만 비교하지 않으므로 느리게 계속 닫히는 구간을 정지로 오인하지 않습니다.
정지가 확인되면 그 정지 구간 시작까지 mask를 채웁니다. 다시 닫히면 정지 확인을 다시 시작하고,
열리면 해제하며 마지막까지 열리지 않으면 에피소드 끝까지 유지합니다.
초기부터 움직이지 않는 구간에는 적용하지 않습니다. 완전히 닫힌 구간도 포함합니다.
Absolute의 다음 목표가 이미 열리는 프레임에는 offset을 적용하지 않습니다.
이는 물체 유무의 확정 판정이 아니며 `gripper_grasp_mask`로 검토할 수 있습니다.
원본 관측은 `gripper_observed`, 단위를 통일한 개폐율은 `gripper_opening`, 실제 판정 결과는
`gripper_grasp_mask`(1=파지 추정), 실측 사용 가능 여부는 `gripper_observation_valid`로 확인합니다.

```yaml
actions:
  mode: relative
  length: 50
  source_format: auto
  gripper:
    label_version: 2
    mode: relative            # absolute / relative / binary
    offset: 0.02             # 추가 닫힘 2% (UI에서 % 단위)
    deadband: 0.001           # 작은 변화 무시 0.1% (고급 설정)
    grasp:
      enabled: true
      stationary_tolerance: 0.005  # 정지 허용폭 0.5%
      stationary_duration_s: 0.3  # 최소 정지 시간, 초
```

공통 `rb_env.gripper_interface.GripperInterface`가 장치와 command 단위를 해석합니다.
현재 환경의 `gripper_pose`는 개폐율 0~1, DH/PIKA 드라이버 내부 위치는 0~100%입니다.
PIKA의 실제 폭(m), native 연속 command 범위, binary 임계값/목표 폭은 환경 config에서 관리합니다.
환경 생성 시 해결된 팔별 설정과 stroke를 `gripper_interfaces`에 기록하여 수집 metadata와 bundle에 보존합니다.
PIKA의 기본 stroke는 0.098m이며 `robot.gripper.<side>.stroke_m`으로 변경할 수 있습니다.
일반 전처리에서는 `command_range`/`observation_range`를 지정할 필요가 없습니다.
Binary label은 실제 목표 receipt가 있으면 우선 사용하고, 없으면 환경의 command 의미와
목표 폭을 해석합니다. 중립 command는 이전 목표를 유지합니다.

새 기록에는 관측의 실측 여부, 유효성, 측정 시각/경과 시간을 함께 기록합니다.
유효하지 않거나 `env.<side>.gripper_control.observation_max_age_s`(기본 0.25초)를 넘긴 관측은
파지 추정을 초기화합니다. PIKA가 실측 전 command 목표를 대신 반환한 값도 파지에 사용하지 않습니다.
이 정보가 없는 기존 기록은 관측이 유효하다고 가정합니다.

이전 단위의 데이터만 UI의 **고급 설정 → 원본 데이터 단위 수동 보정**을 사용하세요.
예를 들어 관측이 0~100%인 기록은 아래와 같이 보정합니다. 보정한 값은 모델의 gripper 관측에도 적용됩니다.

```yaml
    legacy:
      observation_range: [0, 100]
      # source_mode: auto        # 수집 환경 정보가 잘못되거나 없는 경우만 수동 지정
      # command_range: [-1, 1]   # 원본 연속 command 입력 범위 보정
```

`label_version`이 없거나 1인 기존 설정과 bundle은 이전 command 단위/판정 방식을 유지합니다.
UI의 **관측 기반 설정으로 전환** 버튼은 version 2 설정을 만들며 offset/deadband를 개폐율로 환산합니다.
새 파지 판정에는 기본 허용폭/시간을 적용합니다. 변경한 label은 새 폴더로 전처리하고 재학습해야 합니다.

추가 저장 데이터는 모두 retained timestep 기준이며 정규화하지 않는 진단용 값입니다.

| HDF5 dataset | 내용 |
|---|---|
| `source_actions` | 원본 command 보존 |
| `gripper_commands [T,S]` | 해석된 절대 목표 (version 2는 개폐율 0~1) |
| `gripper_previous_commands [T,S]` | Binary 복원 기준 직전 목표 (첫 프레임은 현재 목표) |
| `gripper_observed [T,S]` | 원본 관측 개폐 정도 |
| `gripper_opening [T,S]` | 단위 보정한 관측 개폐율 0~1 (version 2) |
| `gripper_observation_valid [T,S]` | 파지 판정에 사용할 수 있는 실측 여부 (version 2) |
| `gripper_next_observed [T,S]` | Absolute/relative에 사용한 다음 관측 개폐 정도 (원본 관측 단위) |
| `gripper_labels [T,S]` | offset/모드 적용 후, 정규화 전 label |
| `gripper_grasp_mask [T,S]` | 팔별 파지 추정 구간 |

`S`는 팔 수이며 왼팔→오른팔 순서입니다. `delta_actions`는 기존 world delta와 원본 gripper를
유지하고, 최종 `actions`의 gripper만 재구성합니다. Manifest에는 모드와 파지 프레임 수를,
action contract에는 적용 설정을 기록합니다. 기존 출력은 설정 변경으로 갱신되지 않으므로 새 폴더로 저장하세요.

## Normalization과 통계 재사용

```yaml
normalization:
  state:
    method: standard
    epsilon: 0.000001
  ft:
    targets: [ft_history]
    method: standard
    epsilon: 0.000001
  action:
    method: symlog
    temperature: 20.0
    epsilon: 0.000001
  stats_file: null
```

Action은 xyz / rpy / gripper별로 정규화 방법과 epsilon/temperature를 선택할 수 있습니다.
여기서 `rpy`라는 설정 이름은 **저장된 회전 3채널(rotvec, rad)**을 가리키며 Euler 차이로 변환하지 않습니다.
양팔이면 각 팔의 7채널 `[xyz(3), rotation(3), gripper(1)]`에 같은 그룹 설정을 적용하되,
통계는 각 채널별로 계산합니다. Residual 모드는 observed/residual의 xyz에 `xyz`,
두 rotvec에 `rpy` 설정을 적용하되 각각의 채널 통계는 독립적입니다. 마지막 gripper 채널에는
`gripper` 설정을 적용합니다. Gripper label 처리가 먼저, 정규화가 나중입니다.

```yaml
normalization:
  action:
    method: none               # 그룹에서 생략한 옵션의 공통 기본값
    epsilon: 0.000001
    temperature: 20.0
    xyz: {method: standard}
    rpy: {method: symlog_minmax}
    gripper: {method: none}     # binary의 +1/-1/0을 그대로 유지하는 예
```

선택 가능한 방법은 모든 그룹에서 `none / standard / minmax / symlog / symlog_minmax`입니다.
그룹 항목이 하나라도 있으면 통계를 `actions.method: grouped`와 `actions.groups`에 저장하고,
정규화와 역정규화에서 같은 그룹 계수를 사용합니다. 그룹에서 지정하지 않은 옵션은 기존
`normalization.action`의 공통 옵션을 상속합니다. 그룹 항목이 없으면 기존 단일 설정 동작을 유지합니다.
훈련 통계를 재사용할 때는 그룹별 method/epsilon/temperature와 gripper label 설정도 일치해야 합니다.

| method | 정변환 | 역변환 |
|---|---|---|
| none | x | y |
| standard | (x-mean) / max(std,epsilon) | y*scale+mean |
| minmax | (x-midpoint) / max((max-min)/2,epsilon) | y*scale+midpoint |
| symlog | sign(x)*log1p(abs(x)*temperature)/log1p(temperature) | sign(y)*expm1(abs(y)*log1p(temperature))/temperature |
| symlog_minmax | u=(x-midpoint)/scale을 먼저 계산한 뒤 symlog(u) | inverse_symlog(y)*scale+midpoint |

정규화는 feature별이며 값 clipping은 하지 않습니다. 상수 차원은 epsilon으로 0 나눗셈을 방지합니다.
`symlog_minmax`는 `scale=max((max-min)/2,epsilon)`으로 **minmax → symlog** 순서로 적용합니다.
훈련 데이터는 최종 출력도 `[-1,1]` 안에 들어가며, 상수 차원은 0이 됩니다.
반범위가 epsilon 이상인 채널의 최솟값/최댓값은 각각 -1/+1로 매핑됩니다.
저장한 훈련 min/max를 넘는 검증·추론 입력은 `[-1,1]`을 넘을 수 있으며, 역변환도 그대로 지원합니다.
State/action/F/T 모두 선택할 수 있고, `config_osj.yaml`의 action에는 이 방식을 사용합니다.

```yaml
action:
  method: symlog_minmax
  temperature: 20.0
  epsilon: 0.000001
```

State normalization은 `type: state`인 모든 output에 적용합니다. 통계는 crop 이후의
**현재 observations만** timestep당 한 번 계산하고 next observations에도 같은 값을 사용합니다.
F/T는 `normalization.ft.targets`에 지정한 **저장 target key**에 적용합니다.
기본 target은 `ft_history`이며, 각 target은 마지막 차원이 6인 `type: array`여야 합니다.
각 채널(Fx,Fy,Fz,Tx,Ty,Tz)의 통계를 retained 현재 관측 `[T,N,6]`의 T와 N을 합쳐 계산합니다.
History가 겹쳐 저장된 샘플도 저장된 횟수만큼 반영하며 next observations는 fit에 포함하지 않습니다.
0 값은 자동으로 결측치로 간주하지 않고 통계에 포함합니다. F/T 정규화는 state 내부 force/torque 정규화와 별개입니다.
`normalization.ft`를 생략한 기존 config는 F/T를 그대로 보존합니다. 기본 config는 standard를 사용합니다.
통계는 `normalization.json`의 `observations.ft_history`와 각 HDF5 metadata에 저장합니다.
현재 관측, next 관측, `InferencePreprocessor.observation()` 모두 같은 계수로 처리하며 학습/server에서는 재정규화하지 않습니다.

Action 통계는 **모든 유효 chunk 항목**에서 계산하므로 여러 chunk에 들어간 delta는 그 횟수만큼 반영됩니다.
모든 horizon에 공통인 feature별 통계이고 padding은 제외합니다. Relative의 경우 anchor별로 다른 offset을 fit합니다.

통계 범위는 이번 실행의 입력 폴더 전체입니다. 훈련/검증 데이터를 함께 넣으면 검증 정보가 섞이므로
훈련 폴더에서 먼저 만들고, 검증에는 다음처럼 훈련 통계를 지정하세요.

```yaml
normalization:
  state: {method: standard, epsilon: 0.000001}
  ft: {targets: [ft_history], method: standard, epsilon: 0.000001}
  action: {method: symlog, temperature: 20.0, epsilon: 0.000001}
  stats_file: /path/to/processed_train/normalization.json
```

통계 재사용 시 state 열 순서·선택, observation 처리 설정, action 형식·chunk 길이·제어 주기 및
정규화 method/parameter가 일치하는지 검증합니다. Inference 중에는 통계를 다시 계산하지 않습니다.

## 출력과 inference

```text
output_directory/
  <source>__<hash>__ep0000.hdf5   # episode 하나
  <source>__<hash>__ep0001.hdf5
  config.yaml                   # CLI 경로 override까지 반영한 적용 설정
  env_config.json               # 환경 config 전체, action_chunk_mode는 전처리 모드에 맞춤
  normalization.json            # schema, feature별 mean/std/min/max/offset/scale 등
  inference_config.json         # 원본 state 해석, 이미지 처리, chunk/좌표계 convention
  manifest.json                 # 원본별 crop, retained frames, shape, 결과 파일
```

`env_config.json`의 `train.action_chunk_mode`는 `actions.mode: delta`이면 `delta`,
`actions.mode: relative/relative_observed/relative_observed_residual`이면 `relative_tcp`로 저장합니다. Action chunk 변환을 하지 않는
경우에는 원본 값을 유지합니다.

HDF5:

```text
observations/state             [T,D]       정규화된 선택 state
observations/image_*           [T,H,W,3]   RGB uint8
observations/ft_history         [T,N,6]     normalization.ft 설정에 따른 history
next_observations/...                      동일 처리
actions                        [T,50,A]    정규화된 선택 chunk
action_mask                    [T,50]      유효 항목
source_actions                 [T,7*S]     입력 action 원본 값/dtype
delta_actions                  [T,7*S]     정규화 전 world delta (relative_observed 제외)
action_reference_poses         [T,S,7]     각 팔의 원본 관측 xyz + wxyz
rewards / terminals / terminated / truncated / masks
timestamps / source_indices
metadata/preprocess_config / collection_config / recording_metadata / crop
metadata/action_contract / normalization / inference_config
```

`source_indices`는 원본 episode 내 인덱스입니다. timestamp가 없으면 NaN으로 기록합니다.
원본 next observations가 없으면 같은 episode의 다음 observations를 사용하고, 마지막에는
구조화된 기록의 final_observation을 사용합니다. 마지막 next가 없으면 오류이며 다른 episode에서 가져오지 않습니다.
Terminals는 episode 경계, masks는 원본 우선(없으면 1-terminated)입니다. 미종료 구간의 마지막 terminal을 조작하지 않습니다.

Config와 통계는 **각 HDF5 내부에도 동일하게 저장**되므로, inference helper는 디렉터리 또는 HDF5 하나로 초기화할 수 있습니다.

```python
from preprocess.inference import InferencePreprocessor
from rb_env.envs.chunk_interpreter import ChunkInterpreter

pp = InferencePreprocessor('/path/to/processed_train')
# 또는 pp = InferencePreprocessor('/path/to/episode.hdf5')

interpreter = ChunkInterpreter(env, **pp.interpreter_options())
# raw_observation과 같은 추론 요청 시점의 TCP pose를 anchor로 사용.
context = interpreter.capture_context()

# 원본 환경 observation → 학습과 동일한 key/크기/state 차원/정규화
policy_observation = pp.observation(raw_observation)
# 모델이 요구하는 batch 축 추가는 호출부에서 처리
predicted_normalized_chunk = policy(policy_observation)
labels = pp.denormalize_actions(predicted_normalized_chunk)  # [H,A], residual은 팔당 13채널
actions = pp.compose_actions(labels)  # [H,7*S], residual은 여기서 합산
chunk = interpreter.prepare(actions, context=context)

```

`pp.observation(obs, collection_config=live_collection_config)`로 입력 state layout이 다른 환경의
실제 config를 넘길 수도 있습니다. 모델이 받는 선택 채널 이름/순서는 저장 설정을 따릅니다.
`denormalize_actions()`는 저장 label의 정규화만 되돌립니다. `compose_actions()`가 residual을 합산하며,
절대 목표로의 변환은 ChunkInterpreter가 수행합니다.
환경 action에 모델 출력을 직접 전달하면 안 됩니다. RGB 이미지는 이미 처리되므로 inference에서 재차 R/B를 뒤집지 마세요.

위 예시는 gripper 재라벨링을 사용하지 않거나 이전 version 1 absolute label을 사용하는 경우입니다.
Version 2는 세 모드 모두 역정규화 후 개폐 목표를 복원하고 실제 환경의 native command로 변환합니다.
`denormalize_actions()`는 정규화만 되돌리며 파지 offset이나 label 의미를 제거하지 않습니다.

```python
from Real_Robo.inference.policy_io import PolicyIO

# 환경을 만들기 전에 적용: 정책 대상 팔을 continuous native 목표 실행 모드로 구성
live_config = pp.execution_config(live_config)
env = build_env(live_config)
policy_io = PolicyIO(live_config, env.action_space.shape[0], preprocessor=pp)
interpreter = ChunkInterpreter(env, **pp.interpreter_options(decoded_gripper=True))

# 관측/이전 목표는 예측 요청 시점의 값을 캡처
gripper_context = policy_io.gripper_context(raw_observation, previous_openings=last_target_openings)
labels = policy_io.physical_actions(predicted_normalized_chunk)  # 역정규화 및 residual 합산
native_actions = policy_io.execution_actions(labels, gripper_context)
```

Version 2 relative는 예측 요청 시점의 **관측 개폐율**에 변화량을 누적합니다.
Binary는 **직전 실행 목표 개폐율**을 유지 기준으로 사용합니다. +0.5 이상이면 수집 환경의
binary 열림 목표, -0.5 이하이면 닫힘 목표, 그 사이이면 직전 목표를 유지합니다.
`last_target_openings`는 팔별 `_last_gripper_target / 100`이며 관측값과 다를 수 있습니다.
`pp.decode_gripper_actions()`만 직접 쓰면 native command가 아닌 개폐 목표를 반환합니다.
TCP의 xyz/회전 채널은 이 gripper adapter가 변경하지 않습니다.

`env_rpc_client.py --preprocess-bundle ...`은 version 2의 모든 모드에 이 경로를 자동 적용합니다.
예측 요청마다 관측과 목표를 캡처하므로 수동 개입/리셋 후에도 새 기준을 사용하며,
폐기된 비동기 예측은 다음 예측의 누적 기준을 바꾸지 않습니다.
이전 version 1 relative/binary bundle은 기존 adapter 확인 오류를 유지합니다.
직접 복원하려면 기존 next_state relative에는 원본 단위의 `observed_gripper`,
이전 command relative/binary에는 native 단위의 `previous_commands`를 전달하세요.
학습 HDF5의 version 2 현재 개폐율은 `gripper_opening`, 직전 목표는 `gripper_previous_commands`로 확인하며,
`action_mask=false` padding은 복원/실행 대상에서 제외해야 합니다.

전체 파일의 변환/통계/정규화 검사가 완료된 뒤 최종 출력에 게시합니다.
파일별 교체는 atomic이지만 출력 폴더 전체에 대한 동시 읽기/동시 실행 transaction은 아닙니다.
같은 출력 디렉터리로 두 전처리를 동시에 실행하지 마세요. `--dry-run`은 crop과 shape 계획 확인이며 전체 통계 fit 검증은 아닙니다.

## 초반 정지 구간

첫 observation 대비 TCP 위치 거리, 자세 회전각, gripper 변화로 판단합니다.
연속 `consecutive_frames` 검출 후 시작점 이전 `pre_roll_frames`를 남깁니다.
위치 임계값은 현재 config의 `translation_threshold_m`을 따릅니다(사용자가 설정한 값을 유지).
회전은 SO(3) 각도로 비교하므로 ±π 경계에서 큰 움직임으로 오인하지 않습니다.
중간/마지막 정지 구간은 유지하고, 전 구간 정지이면 keep/drop/error를 선택합니다.
History 창 안의 이전 샘플은 샘플별 timestamp가 없어 별도로 자르지 않습니다.

## `Downloads/force_aware_flow_matching_policy` 분석

분석 대상은 `/home/csilab/Downloads/force_aware_flow_matching_policy`의 현재 소스입니다.
**이 폴더에는 pickle→HDF5 변환기, HDF5 저장기, 초반 정지 구간 제거 코드가 없습니다.**
아래는 이미 만들어진 HDF5를 읽은 뒤 학습/추론 경로에서 수행하는 처리입니다.

| 단계 | 실제 동작 | 근거 |
|---|---|---|
| HDF5 로딩 | 지정 폴더 바로 아래 `*.h*5` 파일을 정렬해 timestep 축으로 concatenate. 하위 폴더 탐색 없음 | `utils/datasets_loader.py:99` |
| 입력 key | state/next state, ft_history/next ft_history, actions/rewards/terminals 필수. arm_type에 맞는 이미지 key 필요 | `utils/datasets_loader.py:75` |
| 팔 masking | state는 왼팔 0:19 / 오른팔 19:38, action은 왼팔 0:7 / 오른팔 7:14 가정. 사용하지 않는 팔을 0으로 만듦 | `utils/datasets_loader.py:130`, `:159` |
| state 표준화 | masking 후 학습 데이터 평균/표준편차(+1e-6) 계산. 같은 통계로 current/next state 표준화 | `utils/datasets_loader.py:201` |
| F/T 표준화 | episode timestep과 history sample 축 `(0,1)`으로 평균/표준편차 계산, 채널별 표준화 | `utils/datasets_loader.py:207` |
| gripper 재라벨링 | action gripper의 절댓값 <0.9인 명령을 관측 gripper >0.95 여부에 따라 ±1로 대체. action 인덱스 6/13, state 0/19 사용 | `utils/datasets_loader.py:139` |
| action symlog | `sign(a)*log1p(abs(a)*temperature)/log1p(temperature)`. loader 함수 기본값 10, 실제 train 호출은 **20** | `utils/datasets_loader.py:10`, `train.py:91` |
| reward/mask | train에서 reward에 -1 적용. loader에서 masks=1-terminals, timeouts=terminals 설정 | `train.py:104`, `utils/datasets_loader.py:227` |
| 이미지 encoder | 필요 시 bilinear 224×224 resize, float32 /255, ImageNet pretrained ResNet18의 normalize=True | `utils/encoders.py:33` |
| 이미지 crop 증강 | 학습 update에서 observations 이미지에만 padding=3 edge-pad 후 random crop, p_aug=0.5 | `agents/force_aware_flow_actor.py:158`, `utils/jax_aug.py:52` |
| 색상 증강 | c_aug=1.0, jitter 확률 0.8; brightness/contrast 0.4, saturation 0.2, hue 0.1. 실제 경로의 grayscale 확률은 0 | `utils/jax_aug.py:535`, `configs/force_aware_flow_matching_policy.py:51` |
| 시퀀스 | action chunk 기본 horizon 5. F/T history 구성은 episode 시작 전 인덱스를 시작 프레임으로 clamp. terminal 이후 action loss를 valid로 mask | `utils/datasets.py:68` |
| actor 입력 | 일반 state 안의 F/T 열은 0으로 masking하고 별도 F/T GRU로 조건화 | `agents/force_aware_flow_actor.py:23`, `configs/force_aware_flow_matching_policy.py:33` |

`utils/jax_aug.py`에는 flip/blur/solarize를 포함한 별도 함수와 preset도 있으나,
현재 actor `_update()`에서 호출하는 것은 random crop과 color jitter입니다.
함수 정의만 있는 처리를 실제 학습에 적용되는 처리로 보지 않았습니다.

주의할 호환성:

- 원본 action은 절대 pose이며 이번 전처리에서 delta 또는 TCP-relative rotvec chunk로 변환합니다.
  참고 loader에는 이런 변환이 없습니다. 이미 chunk화된 actions를 다시 chunk화하지 말고, `action_mask`로 loss를 계산해야 합니다.
  Symlog 자체는 [-1,1] 범위를 강제하지 않습니다. 참고 actor의 [-1,1] clipping 정책도 선택한 normalization에 맞춰 검토해야 합니다.
- 기본 `both_third`는 `image_wrist_right` + `image_front`를 요구합니다. 이 config의
  `image_wrist_right` + `image_head`와 이름이 다릅니다. 모델 image key 설정을 맞춰야 합니다.
  head 영상을 근거 없이 front로 변경하지 않았습니다.
- 문서/주석의 F/T는 100Hz라고 되어 있지만 loader가 timestamp를 읽거나 100Hz로 resample하지는 않습니다.
  현재 기록의 `[50,6]` history를 이 변환기에서는 그대로 보존합니다.
- 이제 전처리에서 정규화하므로 참고 loader의 state/action 정규화를 다시 적용하면 안 됩니다.
  그 프로젝트의 `norm_stats.json`과 이번 `normalization.json`은 구조가 다르며 자동 교환되지 않습니다.
  학습/추론에 아래 `InferencePreprocessor`와 저장 통계를 사용하도록 loader를 연결해야 합니다.
- 참고 코드의 시퀀스는 파일 전체를 이어붙인 뒤 terminal 기반 loss mask를 만듭니다.
  이 전처리의 미종료 episode를 학습에 넣으려면 파일 경계 처리도 loader에서 보완해야 합니다.


## 검증

```bash
/home/csilab/miniforge3/envs/hilserl/bin/python -m unittest discover -s tests -p 'test_preprocess*.py' -v
```

실제 `1_demos_2026-10-07_16-09-45_408199.pkl`로 두 버전을 생성했습니다.
현재 crop 설정(위치 임계 0.01m 등)에서는 188프레임 중 처음 13프레임을 잘라 175프레임을 저장했습니다.
각 actions shape는 `[175,50,7]`, 유효 chunk 항목 수는 7,525입니다.
저장된 통계로 역정규화한 relative chunk를 실제 ChunkInterpreter.prepare()로 해석했을 때,
기록 당시 관측 pose 기준 delta의 최대 차이는 약 `1.7e-16`입니다.
샘플은 `output/chunk_delta_sample/`, `output/chunk_relative_sample/`에 있습니다.
이 샘플 통계는 해당 파일 하나로 fit했으므로 전체 학습 데이터 통계 대신 사용하지 마세요.


## F/T 정규화를 켠 delta 데이터 생성

기존 HDF5는 설정 변경만으로 갱신되지 않습니다. raw episode에서 새로운 출력 폴더로 생성합니다.
기존 delta 데이터와 다른 통계의 파일을 같은 폴더에 섞지 마세요.

```bash
python -m preprocess.convert --config preprocess/configs/ff_usb_delta_ft.yaml
```

이 preset은 delta 길이 50, F/T standard 정규화를 사용하고
`demos/preprocessed/ff_usb/delta_ft/`에 저장합니다.
원본 데이터 위치가 다르면 `--input /path/to/raw_episodes`를 추가하세요.
이 명령을 자동으로 실행하거나 기존 데이터셋을 덮어쓰지는 않습니다.
