# 추가 최적화 검토 — 2026-09-27

현재 5k 학습을 유지하면서 CPU 배치와 전처리를 조사했다. 학습을 재시작하거나
학습량·모델·해상도·horizon·전역 배치·학습률을 바꾸지 않았다.
현재 실행은 `turbovla_clean50_pilot5k_2x3090_20260927_0236`이다.

## 확인한 개선

| 대상 | 기존 | 변경 | 검증 범위 |
|---|---:|---:|---|
| 배치 12 텍스트 토큰·마스크 준비 | 중앙값 51.85 ms | 4.76 ms | 같은 토큰·마스크·position ID, 비트 단위 일치 |
| 배치 1 텍스트 토큰·마스크 준비 | 6.85 ms | 1.86 ms | 동일 |
| PyAV 영상의 3개 프레임 읽기 | 27.22 ms | 10.72 ms | 12개 작업 영상, 처음·중간·끝 RGB 픽셀 일치 |

텍스트 시간은 학습이 실행 중인 GPU에서 해당 처리 구간만 측정했다. GPU 경합과 동기화 대기가
포함될 수 있으므로 전체 학습 속도 개선률로 환산하지 않는다. 할당한 CUDA tensor의 최대 크기는
약 1 MiB였으며, 학습 모델을 추가로 올리지 않았다.
영상 시간은 파일 캐시를 미리 채우고 설정 순서를 교대하여 CPU에서 측정했다.
전체 update당 시간이 10배 또는 2.5배 빨라졌다는 의미가 아니다.

1. `turbovla/models/text_encoder.py`: 토큰을 GPU로 보내기 전에 CPU에서 정수 마스크를 만든다.
   기존 Python 루프가 CUDA scalar를 조건문과 인덱스로 읽으며 발생시키던 반복 동기화를 줄였다.
   가변 길이·배치 1/12/16·빈 문장·구두점·256토큰 truncation 사례를 비교했고,
   수정한 실제 `_tokenize_group` 메서드에서도 입력 tensor의 비트 일치를 확인했다.
2. `gr00t_lerobot/video.py`: PyAV thread 수 제한과 명시적인 codec 종료 옵션을 지원한다.
   `lerobot_datasets.py`에서 dataset 설정을 해당 함수로 전달하며, `clean50.yaml`에는
   `thread_count: 2`, `close_codec: true`를 지정했다. 기존 기본 동작을 사용할 때는 옵션을 생략할 수 있다.
   사용 중인 PyAV 12.3.0에서 확인했다. 전체 영상 읽기의 container도 항상 닫도록 정리했다.
   CPU 단독 검사에서는 읽기 후 잔존 thread 수가 97→1이었다. 이는 검사 프로세스의 수치이며,
   기존 학습 프로세스의 thread 수를 즉시 줄였다는 뜻은 아니다.

영상 비교는 timestamp 및 index 경로 모두에서 수행했다. 추가로 영상 한 개 전체
143×480×640 RGB가 동일하고, 음수 thread 수는 오류로 거절하는지 확인했다.
실제 학습 설정 래퍼 `AccessTrackedConfig`에서 decoder 옵션이 factory를 거쳐 전달되는지도 검사했다.

## 현재 실행에 적용되는 시점

실행 중인 Python은 이미 이전 코드를 메모리에 올려 사용하므로, 위 소스 및 YAML 수정으로
현재 학습의 처리 방식이 바뀌지는 않는다. 진행 중인 optimizer/EMA 상태는 그대로 유지했다.

- 학습 종료 후 시작하는 정책 평가에는 CPU 텍스트 마스크 처리가 적용된다.
  기존 자동 실행의 2회 사전 평가가 실제 경로를 다시 확인한다.
- 다음 학습 실행에는 텍스트와 영상 처리 변경이 모두 적용된다.
- 이번 변경 이후의 전체 학습 throughput은 아직 측정하지 않았다.

현재 환경은 모델 가중치만 저장하며 optimizer·RNG를 정확히 이어받는 재개 경로가 검증되지 않았다.
진행분을 버리는 반복 재시작보다 하루 이내 첫 평가 목표를 유지하는 쪽을 선택했다.

## 이득이 확인되지 않았거나 추가 측정이 필요한 항목

- CPU/NUMA affinity: 학습을 유지한 채 GPU별 CPU 소켓을 지정했으나 이득을 분리해 확인하지 못했다.
  원래 CPU 허용 목록을 기록하고, 시험 도중 새로 생긴 보조 thread까지 원복했다.
  측정 중 다른 CPU 작업과 디코더 thread 생성에 따른 변동이 있어 개선률을 주장하지 않는다.
- 배치 16: 앞선 실제 학습 비교에서 추가 개선은 약 3%, GPU 메모리는 약 3 GiB 증가했다.
- BF16·FlashAttention·ZeRO-1: 이미 사용 중이다.
- fused AdamW, EMA의 multi-tensor 연산, `torch.compile`, 고정 BERT 결과 캐시:
  후보지만 아직 전체 update 시간과 수치 동등성을 측정하지 않았다. 가속 배수를 약속하지 않는다.
  고정 BERT를 캐시한다면 학습 중인 text projection은 계속 계산해야 한다.

GPU가 비면 다음 명령으로 새로운 전처리를 포함한 실제 8 update를 측정할 수 있다.
GPU를 사용하는 프로세스가 있으면 이 스크립트는 실행을 거절한다.

```bash
cd /home/jack/TurboVLA
bash scripts/robotwin/benchmark_preprocessing_train.sh
```

결과는 새 run 디렉터리의 `smoke_audit_rank0.json`, `smoke_audit_rank1.json`에 저장한다.
optimizer·scheduler·EMA 갱신, 유한 loss와 parameter 변경도 검사한다.

근거 파일:

- [텍스트 구간 측정](setup_records/text_masks_20260927.json)
- [실제 수정 메서드 동등성](setup_records/text_masks_actual_method_20260927.json)
- [영상 순서·캐시 통제 측정](setup_records/video_threads_controlled_20260927.json)

라이브러리 참고: [PyAV thread_count](https://pyav.org/docs/stable/api/codec.html),
[PyTorch AdamW 구현 선택](https://docs.pytorch.org/docs/2.6/generated/torch.optim.AdamW.html),
[PyTorch 성능 조정 가이드](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).
