2026-09-28 공개 55k EMA 평가의 중간 진단

**후속 조치:** 이 문서는 수정 전 평가 중 작성한 원인 분석이다. 이후 코드 수정,
실제 Transformers 4.56.0과의 출력 동등성 검증, bf44 시뮬레이터 준비,
최신 150k 다운로드를 완료했다. 현재 상태와 평가 결과는
[공개 모델 호환성 준비 기록](PUBLIC_MODEL_COMPATIBILITY_KO.md)을 확인한다.

후속 결과: 이 진단 당시 진행 중이던 평가는 2026-09-28 09:28:48 KST에 종료됐으며,
수정 전 조건에서 49/500 성공(9.8%)을 기록했다. [최종 결과](EVALUATION_RELEASED55K_KO.md).
아래 시간과 진행률은 원인 분석 당시의 기록이다.

현재 낮은 성공률의 가장 유력한 원인은 **Transformers 버전에 따른 DINOv3 시각 특징의 의미 변화**다.
같은 55k 가중치에서 특징 선택만 바꾸는 별도 GPU 검사로 예측이 크게 개선되는 것을 확인했다.
아직 수정 조건으로 시뮬레이터 성공률을 측정한 것은 아니므로, 논문 수치 회복이나 단일 원인이라고 확정하지 않는다.

**진행 상황과 종료 예상**

- 실행: `turbovla_released55k_eval10_20260927_2237`.
- 2026-09-27 22:36:45 KST 시작. 사전 검사 뒤 본 평가 첫 시뮬레이터 로그는 22:40:44.
- 2026-09-28 01:22:32 KST 호스트 확인: 프로세스 실행 중, 129/500회 진행, 12/50개 작업 완료.
- 약 48회/시간. 시도 수 기준 외삽은 09:11, 이전 5k 평가의 작업별 시간에 현재 약 1.10배 속도 비율을 적용한 추정은 09:00.
- 계획용 예상은 **9월 28일 오전 9시 전후, 대략 오전 8~11시**. 통계적 신뢰구간이 아니며, 후반 작업 난이도·성공 시 조기 종료·expert seed 검증 시간에 따라 변한다.
- 이번 분석에서 실행 중인 평가의 코드·가중치·환경·설정을 변경하지 않았다. GPU 검사는 별도 프로세스의 고정 관측 추론만 수행했다.
- 시간 추정 상세: [runtime_projection_20260928.json](setup_records/released_model_20260927/runtime_projection_20260928.json).

**확인된 호환성 문제**

우리 policy 환경은 `transformers==4.57.6`이다. 현재 시각 인코더는
`outputs.hidden_states[-1]`를 선택한다(`turbovla/models/vision_encoder.py:112`).
Transformers 4.56.0에서는 최종 `last_hidden_state`가 hidden-state 목록 끝에 들어가지만,
4.57.6의 DINOv3에는 `@check_model_inputs(tie_last_hidden_states=False)`가 적용되어
`hidden_states[-1]`가 최종 LayerNorm 이전 출력이다. `last_hidden_state`는 LayerNorm 이후 출력이다.
4.57.0과 4.57.6의 DINOv3 modeling 파일을 직접 비교해 해당 decorator 차이를 확인했고,
실제 모델에서도 `backbone.norm(hidden_states[-1]) == last_hidden_state`임을 확인했다.

저자는 [공식 이슈 #4 답변](https://github.com/H-EmbodVis/TurboVLA/issues/4#issuecomment-5217436771)에서
4.56.0으로 평가했으며, 새 버전에서 이 출력 동작이 바뀌어 strict loading이 성공해도 행동이 달라질 수 있다고 설명했다.
해당 이슈는 LIBERO에 관한 답변이다. RoboTwin에 대한 영향은 아래 로컬 55k 실험으로 별도 확인했다.

검사는 동일 BF16 EMA 가중치, 동일 이미지·언어·로봇 상태를 사용했다.
모델 입력은 head/left/right RGB 224×224, 관절 12개 뒤 그리퍼 2개 순서,
관절 min-max 정규화 및 그리퍼 0.49 이진화로 맞췄다.
정답은 실제 demonstration의 이후 50개 행동이다. 에피소드 끝에서는 마지막 행동을 반복했다.

| 조건 | 정규화된 행동의 평균 절대 오차 |
|---|---:|
| 현재 55k, 현재 코드, `hidden_states[-1]` | 0.0220924 |
| 동일 55k, 새 공식 RoboTwin 호환성 코드만 적용 | 0.0221023 |
| 동일 55k, 현재 코드, `last_hidden_state` 선택 | **0.00374853** |
| 동일 55k, 새 공식 호환성 코드와 `last_hidden_state` | 0.00374115 |

- 6개 작업의 episode 0에서 각각 두 시점, 총 12개 관측이다.
- 특징 선택만 변경한 경우 평균 오차 **83.03% 감소**, 12개 중 11개 관측에서 개선됐다.
- 1,200개의 그리퍼 예측 중 13개의 0.49 임계값 판단이 달라졌다.
- 새 공식 코드의 attention/precision 호환성 변경만으로는 이 표본의 오차가 개선되지 않았다.
- 이 오차는 학습 데이터에 대한 작은 진단 표본이며, 성공률·일반화 성능·전체 검증 loss를 의미하지 않는다.
- 분석 코드와 출력:
  [compare_runtime.py](setup_records/released_model_20260927/upstream_update_20260927/compare_runtime.py),
  [현재/새 코드 비교](setup_records/released_model_20260927/upstream_update_20260927/runtime_output_comparison.json),
  [특징 선택 비교](setup_records/released_model_20260927/upstream_update_20260927/post_layernorm_comparison.json).

앞서 수행한 가중치 SHA256, EMA, tensor 변환 및 strict loading 검사는 유효하다.
다만 공식 코드와 로컬 코드의 출력 비교는 같은 4.57.6 환경에서 이루어졌다.
양쪽 코드가 같은 잘못된 특징 의미를 사용하면 그 비교는 통과하므로, 논문 당시 환경과의 일치를 보장하지 못했다.

우리 5k 모델은 4.57.6의 같은 특징 경로로 학습·평가했다.
따라서 공개 55k 모델에 대한 수정 필요성을 로컬 5k 모델에 그대로 적용하면 안 된다.
기존 5k/55k 비교에는 checkpoint별 특징 호환성이라는 교란 요인이 있다.

**그 밖에 확인한 사항**

- 현재/기존 평가 모두 EMA. 원본 55k 가중치와 정규화 통계 보존 검사는 통과했다.
- 카메라 및 관절 순서, 절대 관절 위치 제어, min-max, horizon 50, temporal ensemble은 기존 공식 adapter와 일치한다.
- 저자가 권장한 RoboTwin `bf44be5`와 현재 `c3ddfa8`의 upstream 평가 코드·작업 정의는 동일하다.
  설치 스크립트의 의존성 고정 방식에는 차이가 있으므로 설치 환경·에셋까지 같다고 단정하지 않는다.
  [커밋 대조](setup_records/released_model_20260927/simulator_commit_comparison_20260928.json).
- 우리 eval_policy.py 수정은 평가 횟수 조절, expert setup 오류 출력 및 연속 오류 중단 장치다. 성공 판정은 수정하지 않았다.
- 129개 graded rollout 중 실패 109개는 모두 제어 step 제한에 도달했다.
  시간 제한이나 통신 종료를 실패로 집계한 것이 아니다.
  expert 장면 검증의 caught traceback 2개는 graded policy trial 전에 발생했다.
- RTX 3090 및 HDD는 소요 시간에 영향을 줄 수 있지만, 이 synchronous 평가에는 policy 응답 대기 시간을 성공 실패로 판정하는 wall-clock deadline이 없다.
  현재 증거상 우선적인 정확도 원인은 시각 특징 호환성이다.

**평가 시작 뒤 공식 배포도 변경됨**

평가는 당시 고정한 Hugging Face revision `cb5300544693013164c4bb251a13036002a55c81`의 Clean 55k EMA를 사용한다.
실행 뒤 자동으로 최신 파일을 내려받거나 바꾸지 않는다.

- 9월 27일 **22:57:58 KST**: [공식 코드 6727c875](https://github.com/H-EmbodVis/TurboVLA/commit/6727c875666f8d5dda8d8cca0043da200738fe73) 업데이트.
- **23:10:22 KST**: Hugging Face 체크포인트가 150k EMA로 교체됨. 메타데이터 최종 확인 revision은 `9cdb576ed9c56039b9c1149142ed654e876bad76`.
- [새 배포 설명](https://huggingface.co/H-EmbodVis/TurboVLA/blob/9cdb576ed9c56039b9c1149142ed654e876bad76/README.md)은
  Clean+Randomized, Randomized:Clean=10:1, 작업별 균등 샘플링, global batch 512, 150k 학습이다.
- 새 문서는 Clean 및 Randomized 각각 작업당 100회 평가를 안내한다. 기존 v1의 Clean 55k·60.2%와 별도 실험이다.
- 새 코드에는 legacy checkpoint 변환과 RoboTwin 호환성 profile이 추가됐다.
  이번 분석은 새 source/metadata만 별도 폴더에 보관했고, 새 150k 가중치는 내려받거나 평가하지 않았다.

**다음 검증의 순서**

1. 기존 55k 가중치·통계·시뮬레이터를 유지하고, 분리된 평가 환경에서 시각 특징 선택만 `last_hidden_state`로 바꾼다.
2. 동일 작업·동일 episode seed로 수정 전후 실제 rollout을 비교한다. 언어 instruction도 기록해 조건을 확인한다.
3. 개선을 확인한 뒤 50개 작업 전체를 평가한다. 논문 v1의 60.2% 비교에는 작업당 100회가 필요하다.
4. 새 150k 모델은 해당 revision의 설정·정규화 통계를 함께 사용해 별도 결과로 평가한다.

현재 진행 중인 55k 평가가 끝나도, 그 결과만으로 논문 성능이나 공개 가중치의 실제 성능을 판단하면 안 된다.
특징 호환성 수정 후 rollout 검증이 우선이다.
