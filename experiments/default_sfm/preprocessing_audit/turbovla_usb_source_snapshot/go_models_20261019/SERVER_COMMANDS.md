# TurboVLA 모델별 실로봇 평가 번들

| 모델 ID | 설정 | Step | 학습 상태 |
|---|---|---:|---|
| `go_latched_delta` | GO / 그리퍼 리라벨 / GRU 없음 / delta | 12,640 | completed |
| `go_latched_relative` | GO / 그리퍼 리라벨 / GRU 없음 / relative | 12,640 | completed |
| `go_original_delta` | GO / 원본 그리퍼 / GRU 없음 / delta | 12,640 | completed |
| `go_original_relative` | GO / 원본 그리퍼 / GRU 없음 / relative | 12,640 | completed |
| `go_ft500_delta` | GO / 원본 그리퍼 / FT-history GRU 500Hz / delta | 12,640 | completed |
| `go_ft500_relative` | GO / 원본 그리퍼 / FT-history GRU 500Hz / relative | 12,640 | completed |
| `go_ft100_delta` | GO / 원본 그리퍼 / FT-history GRU 100Hz / delta | 12,640 | completed |
| `go_ft100_relative` | GO / 원본 그리퍼 / FT-history GRU 100Hz / relative | 12,640 | completed |

GO 모델 8종의 최종 12,640 step 체크포인트를 사용합니다. 기본값은 RAW이며 서버 옵션으로 EMA를 선택할 수 있습니다.
이 패키지에는 GO 모델만 포함했습니다.

구성은 그리퍼 리라벨/GRU 없음, 원본 그리퍼/GRU 없음, 원본 그리퍼/500Hz GRU,
원본 그리퍼/100Hz GRU의 네 가지이며 각각 Delta와 Relative가 있습니다.
Delta는 매 스텝의 TCP 기준 증분 액션, Relative는 예측 시점 TCP 기준 오프셋(`relative_tcp`)을 사용합니다.

## 학습 서버에서 시작

호스트: `115.145.172.56`. 프로젝트: `/home/jack/TurboVLA`.

```bash
cd /home/jack/TurboVLA
bash deploy/go_models_20261019/serve_model.sh --list
bash deploy/go_models_20261019/serve_model.sh go_ft100_relative --gpu 1 --weights raw --port 45588
```

`Ready: http://0.0.0.0:45588`이 출력되면 클라이언트를 시작합니다.
모델 이름을 목록의 ID로 바꾸면 체크포인트와 전처리 번들이 함께 선택됩니다.
`--gpu 0` 또는 `--gpu 1`, `--weights raw` 또는 `--weights ema`를 지정할 수 있습니다.
기본 포트는 45587입니다. GPU 번호는 이 서버에서 앞서 사용한 물리 GPU 0/1의 UUID로 고정했습니다.
서버 실행 중에는 모델 한 개가 선택됩니다. 서로 다른 모델을 동시에 띄우려면 GPU와 포트를 나누십시오.

현재 준비된 500Hz Delta 서버와 같은 45587 포트로 전환하려면 기존 서버를 먼저 종료합니다.
이전에 만든 tmux 세션은 다음 명령으로 들어가서 **Ctrl+C**로 종료할 수 있습니다.

```bash
tmux attach -t go_ft500_delta_raw_12640_rpc
# Ctrl+C로 기존 서버 종료 후:
cd /home/jack/TurboVLA
bash deploy/go_models_20261019/serve_model.sh go_latched_delta --gpu 0 --weights raw --port 45587
```

현재 서버를 유지하며 GPU 1에서 다른 모델을 실행하려면 위 45588 예시를 사용합니다.
터미널을 닫은 뒤에도 유지하려면 별도 tmux 세션 안에서 실행합니다.

```bash
tmux new-session -s turbovla_ft100_relative
cd /home/jack/TurboVLA
bash deploy/go_models_20261019/serve_model.sh go_ft100_relative --gpu 1 --weights raw --port 45588
# Ctrl+B, D로 분리; 다시 연결: tmux attach -t turbovla_ft100_relative
```

