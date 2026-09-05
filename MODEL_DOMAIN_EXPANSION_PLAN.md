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
2. ✅ 동일 200문항에서 Qwen control과 공개 SmolLM2 M1/M2의 로딩·정확도·latency·VRAM screening을 실행했다.
3. 다음으로 비용을 wall-clock latency, `모델 크기×생성 토큰` compute proxy, GPU energy로 분리하고 Lower 출력 길이 ablation을 실행한다.
4. latency gate를 통과하는 조건이 생기면 MBPP 격리 실행 scorer를 구현하고 코드 도메인 pilot을 진행한다.
5. 이후 MMLU → KMMLU → IFEval 순서로 scorer와 pilot을 추가한다.
6. Gemma와 Llama는 Hugging Face 약관 동의·접근 권한이 확인된 뒤 M3/M4에 포함한다.

공통 harness는 기존 `question/answer` JSONL과 새 `prompt/reference/task_type/task_metadata` 스키마를 모두 읽는다. 숫자, 객관식, exact-match는 즉시 자동 채점하고 코드·IFEval은 전용 격리 채점기가 연결되기 전까지 `correct=null`로 저장해 임의 판정을 방지한다. 각 실행은 정확도, 파싱률, 생성 토큰, p50/p95 latency, 처리량, peak allocated/reserved VRAM을 기록한다.

### 첫 screening 결과 · latency 기준 전 조합 중단

GSM8K validation 200문항, seed 42, batch 8, 기존 concise 프롬프트, 최대 512토큰으로 비교했다. Qwen 1.5B/7B 정확도는 64%/86%, SmolLM2 360M/1.7B는 8%/57%로 능력 차이는 모두 충분했다. 그러나 p50 latency 비율은 M0 1.632, M1 1.538, M2 0.673으로 고정 상한 0.50을 모두 실패했다. batch 1의 Qwen 50문항 확인 측정에서도 비율은 1.683이었다. Lower가 더 긴 출력을 생성했고 4-bit 소형 모델의 GPU kernel 효율도 낮아 파라미터 수가 작아도 더 빠르지 않았다.

반면 `모델 크기×생성 토큰` compute proxy 비율은 M0 0.380, M1 0.163, M2 0.348이었다. 따라서 기존의 “비용 절감”은 계산량 proxy에서는 가능하지만 wall-clock 응답속도 절감으로 해석할 수 없다. 다음 단계는 비용·지연·energy를 분리하고 Qwen Lower의 concise 512 대 micro-reasoning 64 대 answer-only 64를 비교하는 것이다. 128/256토큰 중간 실행은 길이 상한 도달률이 높아 무효 처리했으며 최종 JSON에는 512토큰 결과만 기록했다.

장기 ablation runner는 Qwen 1.5B/7B와 SmolLM2 360M/1.7B의 baseline을 재측정하고, 세 Lower 모델에 micro-reasoning 64 및 answer-only 64를 적용한다. batch 8×200과 batch 1×50을 합쳐 총 20조건이며, 0.5초 간격 gross GPU power, 문항당 joule, 정확도, 토큰, p50/p95 latency를 함께 기록한다. 각 조건 종료 때 aggregate JSON을 갱신하므로 중단 후 재시작할 수 있다.

### 출력 길이 ablation 결과

20/20 조건이 실패 없이 완료됐다. Qwen1.5B의 batch 8 기준 baseline→micro→answer-only는 정확도 64.0%→8.0%→11.5%, p50 1084.1→235.8→41.5ms, 에너지 224.1→46.5→11.5J/item이었다. SmolLM2-1.7B는 57.0%→11.0%→8.5%, p50 417.9→175.4→32.3ms였다. 짧은 출력은 비용을 크게 줄였지만 독립 풀이 정확도를 대부분 잃었다.

정답 Lower만 완벽하게 선택하고 나머지는 Upper로 보낸다는 oracle 상한에서는 latency break-even 후보가 세 개뿐이었다. Qwen answer-only batch 8의 최대 절감 여유는 3.5%p, SmolLM2-1.7B answer-only batch 8은 2.3%p, batch 1은 0.6%p다. 이는 달성 결과가 아니라 완벽한 selector를 가정한 상한이므로, 다음 단계는 answer-only 출력의 logprob/entropy 특징이 희소한 정답을 높은 precision으로 식별하는지 교차검증하는 것이다. 이 단계가 실패하면 짧은 Lower 경로도 중단하고 연구 주장을 compute/energy 절감으로 제한한다.

### Answer-only confidence gate 결과

200문항의 11개 출력 특징을 stratified 5-fold OOF Logistic Regression으로 평가했다. Qwen1.5B는 AUC 0.722, AP 0.380으로 신호가 있었지만 95% 품질 유지 운용점은 Lower 8%를 채택해 system accuracy 82.0%(Upper 86.0%의 95.35%)와 normalized latency 0.99982를 기록했다. 절감은 0.018%에 불과하고, 채택 16건 중 unsafe가 8건으로 조건부 위험 50%, exact 95% 상한 72.14%였다. 10% 실용 절감과 5% 위험 인증은 모두 실패했다. SmolLM2-1.7B는 AUC 0.618, AP 0.142, normalized latency 1.017로 손익분기도 실패했다. OOF 확률에서 임계값까지 선택한 탐색 결과이므로 Qwen의 미세한 손익분기조차 확인 결과로 주장하지 않는다.

따라서 64-token Answer-only Lower는 중단한다. 다음 최소 실험은 Qwen1.5B의 중간 출력 예산(96/128/192/256)을 concise prompt로 고정해 정확도–latency 곡선을 찾는 것이다. 어느 조건도 품질 유지 95%와 latency 10% 절감을 동시에 만족하지 못하면, 현재 하드웨어에서 wall-clock 성능 개선 주장은 종료하고 compute/energy 절감과 feasibility guard를 중심 결과로 유지한다.

### 중간 출력 예산 sweep 결과

8/8 조건이 오류 없이 완료됐다. Batch 8에서 96/128/192/256토큰의 정확도는 24.5/35.0/56.5/62.0%였고 토큰 상한 도달률은 73.0/52.5/17.5/6.0%였다. 짧은 예산의 속도 개선 대부분은 풀이 절단과 함께 발생했다. 256토큰은 512-token Lower 대비 품질 96.88%를 유지하며 batch 8 p50을 11.11% 줄였지만 상한 도달률 6%로 5% gate를 근소하게 실패했다. Batch 1에서는 동일 품질 유지, p50 10.04% 감소, 상한 도달률 4%로 Lower 압축 gate를 유일하게 통과했다.

그러나 256-token Lower는 Qwen7B 대비 p50이 batch 8에서 1.852배, batch 1에서 1.517배였다. 정답 Lower만 완벽하게 채택하는 oracle에서도 latency와 energy 절감 가능 조건은 0개다. 따라서 같은 수학 도메인의 wall-clock cascade 최적화는 중단한다. 다음 GPU 실험은 MMLU 객관식 200문항에서 모든 모델의 생성을 A/B/C/D 짧은 출력으로 통제해 출력 길이 교란을 제거한다. 모델쌍의 실측 비용비가 0.50 미만일 때만 confidence routing과 후속 도메인으로 확장한다.

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
