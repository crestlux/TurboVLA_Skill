| 모델 ID | 상태 | step | H | request_steps |
|---|---|---:|---:|---:|
| `jack_original_delta_no_ft_history` | 학습 완료 / 사용 가능 | 10,720/10,720 | 50 | 4 |
| `jack_original_delta_gru500hz` | 학습 완료 / 사용 가능 | 10,720/10,720 | 50 | 4 |
| `jack_original_relative_no_ft_history` | 학습 완료 / 사용 가능 | 10,720/10,720 | 50 | 4 |
| `jack_original_relative_gru500hz` | 학습 완료 / 사용 가능 | 10,720/10,720 | 50 | 4 |
| `jack_faflow_delta_no_ft_history` | 학습 완료 / 사용 가능 | 10,400/10,400 | 5 | 4 |
| `jack_faflow_delta_gru500hz` | 학습 완료 / 사용 가능 | 10,400/10,400 | 5 | 4 |
| `jack_faflow_relative_no_ft_history` | 학습 완료 / 사용 가능 | 10,400/10,400 | 5 | 4 |
| `jack_faflow_relative_gru500hz` | 학습 완료 / 사용 가능 | 10,400/10,400 | 5 | 4 |

기준 시각: 2026-10-21T15:23:59.849684+09:00. RAW가 기본이며 --weights ema로 EMA를 선택할 수 있습니다.
학습 중이거나 대기 중인 모델은 번들만 준비되어 있습니다. 서버 실행기는 ready=false 모델의 실행을 거절합니다.
각 모델의 가중치는 학습 호스트에 보존합니다. Kingston에는 가중치를 복사하지 않습니다.

학습 호스트 115.145.172.56에서, 사용자가 서버를 시작할 때:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh --list
bash deploy/jack_models_20261020/serve_model.sh jack_faflow_delta_gru500hz --gpu 0 --weights raw --port 45587
```

위 명령은 물리 GPU 0을 지정합니다. --gpu 1로 GPU 1을 선택할 수도 있습니다.
Ready: http://0.0.0.0:45587 출력 후 Kingston에서 같은 모델 ID로 클라이언트를 실행합니다.

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh --list
bash run_client.sh jack_faflow_delta_gru500hz --trainer_port 45587
```

원래 요청한 모듈 형식으로 Kingston에서 실행할 경우:

```bash
export JACK_MODELS=/home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
export PYTHONPATH="$JACK_MODELS/runtime:/home/csilab/share_projects/clean_share${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/home/csilab/miniforge3/envs/hilserl/bin:$PATH"
cd /home/csilab/share_projects/clean_share/Real_Robo
python3 -m Real_Robo.env_rpc_client \
  --trainer_ip 115.145.172.56 --trainer_port 45587 \
  --env_path "$JACK_MODELS/models/jack_faflow_delta_gru500hz/client_configs" \
  --env_name ff_usb \
  --preprocess-bundle "$JACK_MODELS/models/jack_faflow_delta_gru500hz/bundles/delta" \
  --request_steps 4 \
  --prompt 'Insert the USB connector into the USB port.' \
  --record-dir "$JACK_MODELS/rpc_sessions/jack_faflow_delta_gru500hz"
```

trainer_ip는 이 체크포인트가 있는 실제 호스트인 .56입니다. 예시의 .57과 다릅니다.
서버/클라이언트에서 같은 모델 ID와 포트를 선택해야 합니다. 모니터는 http://127.0.0.1:8125/ 입니다.
환경 파일은 Jack 데이터의 수집 설정(오른팔, 카메라 serial/exposure, reset pose, spacemouse)을 기반으로 합니다.
공용 ff_usb.json 대신 모델 전용 ff_usb.json을 --env_path로 선택합니다.
spacemouse intervention은 기존 환경 설정을 따릅니다.
제어 액션은 10Hz입니다. request_steps=4는 약 0.4초 간격의 재추론 요청입니다.
8개 모델 모두 request_steps=4로 설정했습니다. run_client.sh는 표에 기재된 모델별 request_steps를 사용합니다.
필요하면 bash run_client.sh <모델ID> --request_steps 4처럼 명령행에서 바꿀 수 있습니다.
기존 H=50, FAFLOW H=5이며 반환되는 전체 action 중 실제 실행 범위는 클라이언트가 결정합니다.
Delta는 매 스텝 TCP 기준 증분, Relative는 예측 시점 TCP 기준 relative_tcp입니다.
두 형식 모두 held gripper -1=닫힘, +1=열림 라벨을 학습했습니다.
FT history 미사용 모델도 현재 state의 FT 채널을 사용합니다. GRU는 원본 raw FT [50,6]을 서버에서 정규화합니다.
FAFLOW 번들도 TurboVLA 어댑터의 오른팔 state19/action7과 raw FT를 사용해야 합니다.
따라서 FAFLOW 원본 state38/action14용 번들을 직접 지정하지 말고 위 모델별 번들을 사용합니다.
FAFLOW 카메라는 raw -> 224x224 AREA -> 640x480 LINEAR -> crop -> 224x168 AREA -> 상하 28 black padding을 적용합니다.
패키지의 runtime/preprocess가 그 과정을 수행합니다. 위 PYTHONPATH 또는 run_client.sh를 사용해야 합니다.
현재 state/actions의 저장 정규화를 반복 적용하지 않으며 서버 출력은 normalized action이고 클라이언트가 한 번 역변환합니다.

서버/로봇을 열지 않는 전처리 검증:

```bash
/home/csilab/miniforge3/envs/hilserl/bin/python /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020/check_client.py
```

이 준비 작업은 추론 서버나 로봇을 시작하지 않습니다. 실제 USB 삽입 성공률은 로봇에서 사용자가 측정합니다.
추가 모델의 학습이 끝나면 학습 호스트에서 아래 패키지 갱신 명령을 실행합니다.

```bash
USE_TF=0 /home/jack/miniforge3/envs/turbovla-libero/bin/python scripts/real_robot/prepare_jack_models.py
USE_TF=0 /home/jack/miniforge3/envs/turbovla-libero/bin/python scripts/real_robot/verify_jack_policies.py
/home/jack/miniforge3/envs/turbovla-libero/bin/python scripts/real_robot/install_jack_models.py --update
```

클라이언트 실행 후 조작: 오른쪽 Ctrl=성공, F9=실패, F10=건너뛰기, F12=저장 후 종료입니다.
SpaceMouse로 intervention할 수 있습니다. 평가 기록은 rpc_sessions/<모델ID>에 저장됩니다.
에피소드 수를 제한하려면 bash run_client.sh <모델ID> --episodes 10처럼 실행합니다.

전체 모델별 서버·클라이언트 명령어

아래에서 평가할 모델 한 개의 명령어 쌍을 선택합니다. 서버가 Ready를 출력한 뒤 Kingston 클라이언트를 실행합니다.
모델을 바꿀 때는 클라이언트에서 F12로 저장·종료하고 서버에서 Ctrl+C로 종료한 뒤 다음 명령어 쌍을 실행합니다.
서버는 GPU 0 / RAW 가중치 / 포트 45587 기준입니다. --gpu 1 또는 --weights ema로 변경할 수 있습니다.
각 클라이언트 명령은 모델에 맞는 Delta 또는 Relative 전처리 번들을 자동으로 선택합니다.

1. Jack 기존 / delta / FT history 미사용 (`jack_original_delta_no_ft_history`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_original_delta_no_ft_history --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_original_delta_no_ft_history --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```

2. Jack 기존 / delta / GRU 500Hz (`jack_original_delta_gru500hz`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_original_delta_gru500hz --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_original_delta_gru500hz --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```

3. Jack 기존 / relative / FT history 미사용 (`jack_original_relative_no_ft_history`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_original_relative_no_ft_history --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_original_relative_no_ft_history --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```

4. Jack 기존 / relative / GRU 500Hz (`jack_original_relative_gru500hz`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_original_relative_gru500hz --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_original_relative_gru500hz --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```

5. Jack FAFLOW / delta / FT history 미사용 (`jack_faflow_delta_no_ft_history`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_faflow_delta_no_ft_history --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_faflow_delta_no_ft_history --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```

6. Jack FAFLOW / delta / GRU 500Hz (`jack_faflow_delta_gru500hz`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_faflow_delta_gru500hz --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_faflow_delta_gru500hz --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```

7. Jack FAFLOW / relative / FT history 미사용 (`jack_faflow_relative_no_ft_history`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_faflow_relative_no_ft_history --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_faflow_relative_no_ft_history --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```

8. Jack FAFLOW / relative / GRU 500Hz (`jack_faflow_relative_gru500hz`), request_steps=4

학습 서버에서:

```bash
cd /home/jack/TurboVLA
bash deploy/jack_models_20261020/serve_model.sh jack_faflow_relative_gru500hz --gpu 0 --weights raw --port 45587
```

Kingston에서:

```bash
cd /home/csilab/share_projects/clean_share/team_codes/turbovla/jack_models_20261020
bash run_client.sh jack_faflow_relative_gru500hz --trainer_ip 115.145.172.56 --trainer_port 45587 --request_steps 4
```
