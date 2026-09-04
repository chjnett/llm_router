# 모델·도메인 일반화 강화 계획

## 1. 현재까지 확인된 사실

- Qwen2.5-1.5B/7B 조합에서 output-aware routing은 ASDiv와 MAWPS의 Upper 품질을 약 98% 유지하면서 비용을 12~20% 줄였다.
- 동일한 0.60/0.60 정책은 MATH-500 수치형 문제에서 안전했지만 Lower 채택률이 2.21%에 그쳐 비용이 2.93% 증가했다.
- 정답 라벨 없이 라우팅률과 모델 비용만 사용하는 feasibility guard를 사후 적용했을 때, 어려운 도메인에서 cascade를 끄고 비용 증가를 제거했다. 아직 신규 traffic의 prospective 검증은 남아 있다.
- 아직 모든 생성 모델이 Qwen 계열이고 대부분의 실험이 수학이므로, 모델 독립성과 도메인 독립성은 입증되지 않았다.

## 2. 새 연구 질문

- **RQ1 모델 일반화:** output-aware confidence와 feasibility guard가 SmolLM2·Gemma·Llama에서도 작동하는가?
- **RQ2 계열 간 전이:** 한 모델 family에서 학습한 capability 표현이 다른 Lower/Upper 조합으로 전이되는가?
- **RQ3 도메인 일반화:** 수학 외 코드·지식·한국어·지시 준수에서 품질–비용 frontier가 유지되는가?
- **RQ4 안전한 비활성화:** cascade가 경제적으로 불가능한 도메인을 정답 라벨 없이 사전에 차단할 수 있는가?

## 3. 모델 실험 행렬

| ID | Lower | Upper | 목적 | 우선순위 |
|---|---|---|---|---|
| M0 | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct | 기존 결과 재현 control | 필수 |
| M1 | SmolLM2-360M-Instruct | SmolLM2-1.7B-Instruct | 공개 가중치로 즉시 가능한 non-Qwen control | 1순위 |
| M2 | SmolLM2-1.7B-Instruct | Qwen2.5-7B-Instruct | cross-family cascade 효과 분리 | 1순위 |
| M3 | Gemma-3-1B-IT | Gemma-3-4B-IT | Google 계열 same-family 재현 | 약관 동의 후 |
| M4 | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Meta 계열 재현 | 접근 승인 후 |

RTX 3090 24GB에서 각 모델은 한 번에 하나만 적재한다. 모든 조합의 비용비 `C_L/C_U`는 파라미터 수로 추정하지 않고 동일 GPU에서 p50/p95 latency, 생성 토큰, peak VRAM을 다시 실측한다.

## 4. 데이터셋·채점 행렬

| 도메인 | 1차 screening | 확인 평가 | 채점 방식 |
|---|---|---|---|
| 산술·수학 | ASDiv 200 + MATH-500 numeric 200 | 기존 전체 결과 | 숫자 exact match |
| Python 코드 | MBPP validation/test 일부 | MBPP test + HumanEval | 격리된 실행 기반 unit test |
| 영문 지식 | MMLU validation 200 | subject-stratified MMLU test | 보기 A/B/C/D exact match |
| 한국어 지식 | KMMLU validation 200 | subject-stratified KMMLU test | 보기 exact match |
| 지시 준수 | IFEval 200 | IFEval 전체 541 | 공식 strict/loose verifier |

코드 생성 결과는 네트워크가 차단된 별도 프로세스/컨테이너에서 시간·메모리 제한을 두고 실행한다. 자유 생성 judge는 사용하지 않고 재현 가능한 자동 채점만 사용한다.

## 5. 실행 순서와 중단 기준

### Phase 0 · 공통 harness

모델 registry, chat template, task adapter, scorer, latency/VRAM logger를 공통 인터페이스로 만든다. 수학 전용 `question/answer/correct` 구조를 `prompt/reference/task_metadata` 구조로 일반화한다.

### Phase 1 · 200문항 모델 screening

M0~M4를 수학 200문항에서 먼저 비교한다. 공개 가중치 M0~M2를 먼저 실행하고, M3/M4는 접근 승인 뒤 추가한다. 아래 중 하나라도 실패하면 해당 조합의 전체 도메인 확장을 중단한다.

- Upper−Lower 정확도 차이 5%p 이상
- Upper 정확도 절대값이 해당 데이터의 유효 기준 이상
- 실측 `C_L/C_U < 0.50`
- 출력 파싱 성공률 98% 이상
- peak VRAM 22GB 이하

### Phase 2 · 도메인 pilot

통과 모델 조합만 MBPP, MMLU, KMMLU, IFEval 각 200문항에서 평가한다.

- Upper 품질 유지율 95% 이상
- observed unsafe 5% 이하 및 exact 95% 상한 별도 보고
- 비용 절감 10% 이상 또는 feasibility guard가 Always Upper로 안전하게 비활성화
- capability AUC 0.60 이상

### Phase 3 · 전체 확인 평가

Pilot gate를 통과한 셀만 전체 test와 3 seeds로 확장한다. Router 학습, 정책 선택, risk certification, 최종 test를 분리하며 test를 본 뒤 threshold를 수정하지 않는다.

### Phase 4 · 일반화 ablation

- family-specific router 대 universal router
- query-only 대 output-aware 대 output-aware+guard
- same-family 대 cross-family cascade
- domain-specific threshold 대 global threshold
- 비용을 latency·token·energy 세 방식으로 계산

## 6. 예상 시간

| 단계 | 예상 작업 시간 | 예상 GPU 시간 |
|---|---:|---:|
| 공통 harness·자동 채점기 | 1~2일 | 1~3시간 |
| 모델 5조합 screening | 0.5~1일 | 5~10시간 |
| 코드·영문 지식 pilot | 1~2일 | 8~16시간 |
| 한국어·지시 준수 pilot | 1~2일 | 8~16시간 |
| 통과 셀 전체 3-seed | 1~3일 | 10~20시간 |
| 통계·표·Figure·HTML | 0.5~1일 | 0~2시간 |
| **총계** | **약 4~7 작업일** | **약 32~65시간** |

## 7. 가장 먼저 구현할 작업

1. ✅ `ModelSpec` registry와 모델별 chat-template adapter를 추가했다 (`src/model_registry.py`, `src/task_harness.py`, `src/run_model_screening.py`).
2. 동일 200문항에서 Qwen control과 공개 SmolLM2 M1/M2의 로딩·정확도·latency·VRAM smoke test를 실행한다.
3. 통과하면 MBPP 격리 실행 scorer를 구현하고 코드 도메인 pilot을 진행한다.
4. 이후 MMLU → KMMLU → IFEval 순서로 scorer와 pilot을 추가한다.
5. Gemma와 Llama는 Hugging Face 약관 동의·접근 권한이 확인된 뒤 M3/M4에 포함한다.

공통 harness는 기존 `question/answer` JSONL과 새 `prompt/reference/task_type/task_metadata` 스키마를 모두 읽는다. 숫자, 객관식, exact-match는 즉시 자동 채점하고 코드·IFEval은 전용 격리 채점기가 연결되기 전까지 `correct=null`로 저장해 임의 판정을 방지한다. 각 실행은 정확도, 파싱률, 생성 토큰, p50/p95 latency, 처리량, peak allocated/reserved VRAM을 기록한다.

새 논문 주장은 “Qwen 수학 라우터”가 아니라 **모델 family와 task 도메인이 바뀌어도 output-aware routing과 feasibility guard가 언제 비용 효율적이며, 언제 자동으로 비활성화되어야 하는가**로 확장한다.

## 8. 실행 환경과 공식 출처

- 현재 환경: RTX 3090 24GB, PyTorch 2.5.1+cu121, Transformers 5.12.1. Gemma 3가 요구하는 Transformers 4.50 이상을 만족한다.
- SmolLM2 360M/1.7B 공식 모델 카드: <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>, <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct>
- Gemma 3 공식 모델 카드: <https://huggingface.co/google/gemma-3-4b-it>
- Llama 3.2 공식 모델 카드: <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>
- MBPP 공식 설명·분할: <https://github.com/google-research/google-research/tree/master/mbpp>
- HumanEval 공식 저장소: <https://github.com/openai/human-eval>
- MMLU 데이터 카드: <https://huggingface.co/datasets/cais/mmlu>
- KMMLU 데이터 카드: <https://huggingface.co/datasets/HAERAE-HUB/KMMLU>
- IFEval 공식 평가 코드: <https://github.com/google-research/google-research/tree/master/instruction_following_eval>
