# 공개 모델 호환성 수정 및 최신 배포 준비 — 2026-09-28

**2026-09-29 결과 갱신:** 50개 작업의 Clean·Randomized 평가가 모두 정상 완료됐다.
Clean 443/500(88.6%), Randomized 439/500(87.8%), 평균 88.2%다.
[최종 결과와 검증 보고서](EVALUATION_RELEASED150K_KO.md).

기존 공개 55k의 9.8%는 DINOv3 특징을 저자 환경과 다르게 읽은 조건의 결과다.
공개 모델용 특징 선택을 수정했고, 55k와 새 150k 모두 저자가 언급한 Transformers 4.56.0의
참조 출력과 고정 관측 12개에서 **bitwise 동일함(최대 차이 0)**을 확인했다.
현재 설치는 4.57.6을 유지한다. 실제 Clean 진단 4개 작업 × 5회에서 수정한 55k는
**16/20(80%)**, 최신 150k는 **20/20(100%)**로 정상 완료했다. 전체 benchmark 성공률과 구분한다.

## 원인과 수정

기존 코드는 DINOv3의 `hidden_states[-1]`를 사용했다. Transformers 4.56.0에서는 최종
LayerNorm 이후 특징이지만, 4.57.6에서는 LayerNorm 이전 특징이다. 같은 가중치가 정상적으로
로드되어도 정책 입력이 달라진다. 저자도 [공식 이슈 답변](https://github.com/H-EmbodVis/TurboVLA/issues/4#issuecomment-5217436771)에서 이 문제와 4.56.0 기준 평가를 설명했다.
그 답변은 LIBERO에 관한 것이며, RoboTwin 영향은 이 작업의 별도 검사로 확인했다.

- `VisionEncoderConfig.feature_source`를 추가하고 공개 모델 번들에는 `last_hidden_state`를 명시했다.
- 기존 5k 모델은 4.57.6의 이전 경로로 학습했으므로, 설정이 없는 기존 모델의 기본 동작은 보존했다.
- 새 공식 `compatibility_profile=robotwin`의 attention 및 dtype 처리를 150k에 적용했다.
  55k는 특징 변경의 효과를 분리하기 위해 이전 수치 경로(`libero`라는 profile 이름)를 유지한다.
  이 이름은 내부 계산 경로이며 55k 모델이 LIBERO 모델이라는 뜻이 아니다.
- 각 공개 가중치와 함께 배포된 정규화 통계 및 설정을 별도 번들로 보존했다.
  **150k에 기존 55k 통계를 사용하면 안 된다.**
- 평가 스크립트는 번들 SHA256, Transformers 버전, 수치 동등성 기록, 시뮬레이터 commit을 확인한다.
  PyTorch 버전도 확인한다. 새 Run ID를 사용하며 실행 당시 코드·설정·패키지 목록을 결과와 함께 보관한다.

이전 공식/로컬 코드 비교는 양쪽 모두 4.57.6에서 실행했으므로, 양쪽의 같은 특징 오류를
발견하지 못했다. 이번 검사는 분리해 둔 실제 4.56.0 패키지로도 실행하여 그 한계를 보완했다.
가중치 자체를 수정하거나 추가 학습하지 않았다. 평가 가중치는 모두 EMA다.

## 출력 검증

6개 작업의 demonstration에서 두 시점씩 선택한 동일한 RGB·언어·로봇 상태를 사용했다.
4.56.0 프로세스에서는 해당 배포의 공식 core code와 먼저 일치 여부를 확인했다.
이후 수정된 현재 4.57.6의 출력을 그 참조와 비교했다.

| 모델 | 공식 core와 4.56.0 참조 | 수정된 4.57.6과 참조 | 정규화 행동 MAE |
|---|---|---|---:|
| 55k EMA | bitwise 동일 | bitwise 동일, 최대 차이 0 | 0.00374853 |
| 150k EMA | bitwise 동일 | bitwise 동일, 최대 차이 0 | 0.00178774 |

수정 전 55k의 같은 관측 MAE는 0.0220924였다. 특징 수정만으로 약 83% 감소했다.
이는 학습 데이터에서 추출한 작은 진단 표본이다. 표의 MAE는 모델별 통계를 사용하므로
55k와 150k 사이의 직접적인 성능 비교 지표로 쓰지 않는다. 성공률은 실제 rollout으로 확인한다.

검증 자료: [55k 참조](setup_records/compatibility_20260928/reference55k_transformers456.json),
[55k 수정](setup_records/compatibility_20260928/fixed55k_transformers4576.json),
[150k 참조](setup_records/compatibility_20260928/reference150k_transformers456.json),
[150k 수정](setup_records/compatibility_20260928/fixed150k_transformers4576.json).

## 새 공개 모델

2026-09-28 14:38 KST 기준 공식 최신 코드 commit은 `6727c875666f8d5dda8d8cca0043da200738fe73`,
Hugging Face revision은 `9cdb576ed9c56039b9c1149142ed654e876bad76`이다.
9월 27일 23:10 KST에 기존 55k 파일이 150k EMA로 교체됐다.

[새 모델 설명](https://huggingface.co/H-EmbodVis/TurboVLA/blob/9cdb576ed9c56039b9c1149142ed654e876bad76/README.md)에 따르면
50개 작업의 Clean+Randomized 데이터를 함께 학습했고, 작업별 균등 샘플링,
Randomized:Clean=10:1, global batch 512, 150,000 optimizer updates를 사용한다.
기존 Clean 55k와 학습 조건이 다르다.

- 다운로드한 원본: `pretrained/releases/turbovla_robotwin_9cdb576e_150k/checkpoints/robotwin/steps_150000_ema_pytorch_model.pt`
- 크기: 868,377,428 bytes, 약 868 MB / 828 MiB.
- 공식 LFS SHA256: `5fae4c5a51fae4ffd1aa1d41474b24eea39a95a7111d23c7173f012ac542fdff`와 일치.
- 878개 tensor의 이름 변환·shape·dtype·값 보존·유한값·strict loading 검증 통과.
- 새 정규화 통계는 27,500개 trajectory, 6,075,103개 transition을 반영한다.

최신 [공식 성능표](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/assets/figures/performance.png)는
Clean **88.02%**, Randomized **88.10%**, 평균 **88.06%**다.
기존 v1의 Clean 55k·60.2%와 별도 배포/실험으로 다룬다.
최신 [공식 평가 안내](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/experiments/robotwin/README.md)는
각 variant에서 50개 작업 × 100회, 두 variant 합계 10,000회를 사용한다.

## 환경 검토

| 항목 | 준비 상태 |
|---|---|
| Policy conda | `turbovla-robotwin`, Torch 2.6.0+cu124, Transformers 4.57.6, FlashAttention 2.7.4.post1 |
| 참조 Transformers | 4.56.0 wheel을 별도 디렉터리에 압축 해제하여 참조 프로세스에서만 사용 |
| Simulator conda | `robotwin-turbovla`, Torch 2.4.1+cu124, SAPIEN 3.0.0b1, MPLib 0.2.1, cuRobo 0.7.8 |
| Simulator source | `/home/jack/practice/RoboTwin-TurboVLA-bf44be5`, 공식 권장 commit `bf44be51cf5717a5595ce59447f2cf5263d2aa95` |
| Simulator assets | objects, embodiments, background_texture의 다운로드 SHA256·압축 CRC 검증 완료 |
| 의존성 | 두 conda 환경의 `pip check` 통과 |
| 평가 adapter | 최신 공식 adapter, ensemble, deploy 설정, `eval_task.sh`와 파일 내용 일치 |

카메라 순서는 head/left/right RGB 224×224, 입력은 관절 12개 뒤 그리퍼 2개 순서다.
출력은 14차원 절대 관절 위치, min-max 정규화, 그리퍼 이진화, horizon 50,
temporal ensemble(alpha=0.1), 매 step 정책 호출을 유지한다.
bf44와 기존 c3 커밋의 upstream 작업 정의·평가 로직은 동일하다.
로컬 시뮬레이터 변경은 횟수 조절, 오류 진단, episode seed/instruction 기록이며 성공 판정은 유지한다.

패키지 버전·에셋·코드 대조와 수치 검증을 완료했지만, 저자의 GPU/드라이버 및 물리 엔진 실행까지
bitwise 동일하다고 확인한 것은 아니다. Expert planner의 장면 검증에서 허용되는 seed가 실행마다
달라질 수 있으므로, 실제 비교 결과에 seed/instruction 일치 여부를 기록한다.
또한 `_base_task.py`는 NumPy·Torch seed를 설정하지만 Python `random.seed`는 주석 처리되어 있다.
공식 지시문 생성기는 `random.shuffle`과 `random.choice`를 사용하므로 같은 scene seed만으로
같은 지시문이 보장되지 않는다. 해당 동작은 유지했으며 실제 선택한 지시문을 로그에 남긴다.

## 실제 평가 기록

호환성 검증용으로 각 모델에서 Clean 4개 작업 × 5회씩 순서대로 실행한다.
작업: `adjust_bottle`, `beat_block_hammer`, `dump_bin_bigbin`, `grab_roller`.
seed=0, unseen instruction, instruction 생성 개수=10으로 고정했다.
이 설정은 기존 10회 평가의 instruction 생성 조건을 유지하면서 rollout 수만 줄인다.

- 55k: `turbovla_release55000_postnorm_probe5_20260928_144533`
- 150k: `turbovla_release150000_postnorm_probe5_20260928_144533`
- tmux: `turbovla_compat_20260928_144533`
- [실행 계획](setup_records/compatibility_20260928/probe_queue.json)

55k 검증은 **16/20 성공(80%)**으로 정상 종료했다. 같은 네 작업의 기존 평가에서
첫 5회씩을 추출하면 **6/20(30%)**이다.

| 작업 | 기존 55k의 첫 5회 | 수정한 55k | 최신 150k |
|---|---:|---:|---:|
| adjust_bottle | 2/5 | 5/5 | 5/5 |
| beat_block_hammer | 1/5 | 3/5 | 5/5 |
| dump_bin_bigbin | 0/5 | 3/5 | 5/5 |
| grab_roller | 3/5 | 5/5 | 5/5 |
| 합계 | 6/20, 30% | 16/20, 80% | 20/20, 100% |

두 작업은 seed 순서가 같고 두 작업은 일부 다르며, 공통 seed는 총 17/20개다.
기존 실행에는 instruction 문자열이 없으므로 엄밀하게 입력이 같은 20쌍의 비교는 아니다.
이 표를 이전 전체 50개 작업의 9.8%와 직접 비교하거나, 80%를 전체 benchmark 성공률로 해석하지 않는다.
수치 동등성 검사와 이 작은 rollout의 개선을 함께 볼 때, 특징 호환성 문제가 낮은 성능의
주요 원인이었다는 근거가 강화됐다.

[55k 원본 로그·CSV·시뮬레이터 점수·20개 영상 일치 검사](setup_records/compatibility_20260928/probe55k_results.json),
[20개 영상 스트림 검사](setup_records/compatibility_20260928/probe55k_video_audit.json).
150k도 로그·CSV·시뮬레이터 점수와 20개 영상 검사를 통과했다.
수정된 55k와 150k의 공통 scene seed는 17개지만, seed와 지시문 문자열까지 같은 것은
1개뿐이었다. 그러므로 80%와 100%의 차이를 통제된 모델 비교의 효과 크기로 해석하지 않는다.
[세 실행 비교 및 완료 검증](setup_records/compatibility_20260928/probe_results.json),
[150k 영상 검사](setup_records/compatibility_20260928/probe150k_video_audit.json).

Randomized 지원은 위 두 Clean 실행이 모두 끝난 후 적용했다.
`start_eval.sh`의 모드 지원을 최신 공식 코드와 맞추고, 실행 기록·진행 조회·집계를 variant별로 구분했다.
기존 Clean 500회 로그를 임시 폴더에서 재집계해 기존 9.8%와 일치함을 확인했다.
Randomized의 `adjust_bottle`, `dump_bin_bigbin`을 각 2회 실행하여 **4/4 성공**했다.
원본 로그·집계·시뮬레이터 결과·영상 스트림 검증을 통과했다.
[Randomized 실행 및 검증 기록](setup_records/compatibility_20260928/randomized_preflight.json).

## 최신 모델의 50개 작업 평가

**2026-09-28 15:15:53 KST 시작 → 2026-09-29 03:10:01 KST 정상 완료.** 호환성을 검증한 최신 150k EMA로
Clean 50개 작업 × 10회 이후 Randomized 50개 작업 × 10회를 순서대로 실행했다.
총 1,000회이며, 공식 규모인 총 10,000회보다 작은 평가다. 추가 학습은 수행하지 않는다.
instruction 생성 개수는 두 조건 모두 공식 기본값인 100으로 유지한다.

- Clean: `turbovla_release150000_postnorm_demo_clean_eval10_20260928_151553`
- Randomized: `turbovla_release150000_postnorm_demo_randomized_eval10_20260928_151553`
- tmux: `turbovla_release150k_20260928_151553`
- [실행 계획 및 큐 상태](setup_records/compatibility_20260928/benchmark_queue.json)
- 각 결과는 `/data/jack/robotwin2_turbovla/evaluation/<run_id>_seed0`의 `summary.json`, `task_results.csv`에 저장한다.
- [통합 결과](setup_records/compatibility_20260928/latest150k_eval10_summary.json): Clean 88.6%, Randomized 87.8%, 평균 88.2%.

두 pipeline은 모두 `completed`, 종료 코드 0이다. 1,000개 trial과 영상·보존 가중치 기록의 검증도 통과했다.
[최종 보고서](EVALUATION_RELEASED150K_KO.md). 과거 55k·5k 결과는 각각 기존 run 폴더에 보존한다.

## 실행 및 확인

```bash
cd /home/jack/TurboVLA
python3 scripts/robotwin/pipeline_status.py turbovla_release150000_postnorm_demo_clean_eval10_20260928_151553
python3 scripts/robotwin/pipeline_status.py turbovla_release150000_postnorm_demo_randomized_eval10_20260928_151553
```

두 명령은 완료 상태와 결과를 조회하며 새 평가를 시작하지 않는다.
`active-evaluation.json`은 가장 최근 실행을 가리키며, 완료된 과거 결과는 각 run 폴더에 보존된다.

다음은 **새 평가를 시작하는** 명령이다. GPU가 사용 중이면 실행을 거부한다.
55k Clean 전체 50개 작업 × 10회:

```bash
cd /home/jack/TurboVLA
ROBOTWIN_TEST_NUM=10 bash scripts/robotwin/evaluate_released_model.sh
```

150k Clean 전체 50개 작업 × 10회:

```bash
cd /home/jack/TurboVLA
ROBOTWIN_TEST_NUM=10 bash scripts/robotwin/evaluate_latest_released_model.sh
```

특정 작업만 확인하려면 명령 마지막에 작업명을 붙인다. `ROBOTWIN_TEST_NUM=100`은 작업당 100회다.
Randomized는 다음처럼 선택한다.

```bash
cd /home/jack/TurboVLA
ROBOTWIN_EVAL_MODE=demo_randomized ROBOTWIN_TEST_NUM=10 \
  bash scripts/robotwin/evaluate_latest_released_model.sh
```

이제 지시문 생성 개수의 기본값은 공식 평가와 같은 100이다. 위 40회 진단만 이전 실행과
생성 개수를 맞추기 위해 명시적으로 10을 사용했다. 필요한 경우 `ROBOTWIN_INSTRUCTION_SAMPLE_NUM`으로 지정한다.
Randomized **평가**에 Randomized 학습 데이터셋 다운로드는 필요하지 않다. 시뮬레이터 에셋을 사용한다.
가중치·참조 패키지는 SSD, 평가 로그와 동영상은 `/data/jack/robotwin2_turbovla`의 HDD에 둔다.
