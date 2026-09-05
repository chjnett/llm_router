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

### MMLU 짧은 정답 screening 결과

8/8 조건이 오류 없이 완료됐다. Batch 8에서 Qwen1.5B/7B 정확도는 49.5/69.0%, p50은 59.78/65.68ms로 latency 비율 0.910이었다. SmolLM2 360M/1.7B는 정확도 22.5/39.0%, latency 비율 1.424였고, SmolLM2-1.7B/Qwen7B cross-family 비율은 0.753이었다. 평균 출력이 4.6~9.0토큰으로 통제되면서 Qwen 수학의 1.632배 속도 역전은 대부분 사라졌으므로 출력 길이가 주요 교란 변수였음은 확인됐다.

하지만 모든 조합이 사전 고정한 latency 0.50 gate를 실패했고, Lower 정답만 완벽하게 채택하는 oracle에서도 latency와 energy 절감 조건은 0개였다. 다음 최소 GPU 진단은 Qwen1.5B FP16 대 기존 4-bit 비교다. 작은 모델에서 bitsandbytes dequantization overhead를 제거해 MMLU와 256-token 수학의 처리 속도가 회복되는지 확인한다. 이 4조건도 실패하면 현재 Transformers 단일 RTX 3090 환경의 wall-clock routing 주장을 종료하고, compute proxy/에너지 및 다른 serving backend 검증을 별도 연구 질문으로 분리한다.

### Qwen serving precision 결과와 option-logit 피봇

FP16 4조건도 오류 없이 완료됐다. MMLU batch 8/1의 p50은 기존 4-bit 59.78/228.64ms에서 51.34/159.25ms로 14.1/30.3% 개선됐다. 수학 256-token batch 8/1은 963.60/4195.32ms에서 529.68/1918.57ms로 45.0/54.3% 개선됐다. 그럼에도 Qwen7B 대비 latency 비율은 MMLU 0.782/0.675, 수학 1.018/0.694로 0.50 gate를 모두 실패했다. MMLU parse 성공률도 96.5/94.0%라 98% gate에 미달했다.

따라서 생성 답변 기반 wall-clock 튜닝은 종료한다. 마지막 최소 진단은 MMLU의 A/B/C/D 후보 토큰 logit을 한 번의 forward로 비교하는 결정론적 option-logit 평가다. 생성 루프와 형식 파싱을 제거한 조건에서도 비용비 0.50을 만족하지 못하면 현재 단일 RTX 3090 환경의 wall-clock routing 주장을 최종 종료한다.

### Option-logit one-forward 결과

8/8 조건이 오류 없이 완료됐다. Batch 8에서 M0 Qwen1.5B FP16/Qwen7B의 p50은 12.14/49.13ms로 비용비 0.247, M2 SmolLM2-1.7B/Qwen7B는 14.17/49.13ms로 0.288이었다. 두 조합 모두 사전 0.50 gate를 처음으로 통과했다. 정답 Lower만 채택하는 oracle에서 M0는 정규화 지연 0.722와 시스템 정확도 74.0%(Upper 65.0%), M2는 0.873과 72.0%를 기록했다. M1은 비용비 0.363이지만 Lower 정확도가 21.5%라 oracle 정규화 지연 1.148로 실패했다.

따라서 M0와 M2의 option probability, top-1 margin, normalized entropy, logit spread를 수집하고 5-fold OOF Logistic Regression으로 실제 선택 가능성을 평가한다. 동일 OOF 예측에서 threshold까지 선택하는 탐색 결과이므로, 95% 품질 유지와 10% latency 절감을 통과해도 독립 split 재인증 전에는 확인 결과로 주장하지 않는다.

### Option probability confidence 결과

5-fold OOF에서 M0 Qwen의 routing AUC/AP는 0.698/0.761이었다. Threshold 0.545에서 Lower 채택률 43.0%, 시스템 정확도 62.0%(Upper 65.0%의 95.38%), 정규화 지연 0.817로 18.30% 절감을 기록해 성능 gate를 처음 통과했다. M2는 AUC/AP 0.610/0.570, 동일 품질점 정규화 지연 1.118로 비용 gate를 실패했다.

M0도 strict risk 인증은 실패했다. 채택 86건 중 unsafe가 10건이고 exact 95% 상한은 18.93%다. 또한 모델과 threshold를 같은 OOF 예측에서 선택했으므로 탐색 결과다. 다음 단계는 M0 모델, 8개 특징, Logistic Regression 설정, threshold 0.545를 고정하고 MMLU test에서 추출한 독립 250 certification 및 250 final test로 재평가하는 것이다.

### 독립 MMLU test 250+250 확인 결과

MMLU validation과 겹치지 않는 test 500문항을 seed 2026으로 과목 균형 추출하고, 앞 250개를 certification, 뒤 250개를 final test로 고정했다. 모델, 8개 특징, Logistic Regression 설정, threshold 0.545, latency ratio 0.247을 변경하지 않았다.

Certification에서 Lower/Upper 정확도는 64.4/75.2%, 채택률 51.2%, 시스템 정확도 70.4%, 품질 유지 93.62%, 지연 절감 26.50%였다. 지연 기준은 통과했지만 품질 95%를 실패했다. Final test는 Lower/Upper 65.2/77.2%, 채택률 48.4%, 시스템 정확도 74.8%, 품질 유지 96.89%, 지연 절감 23.70%로 성능 gate를 통과했다. 그러나 사전 평가 순서에서 certification이 실패했으므로 전체 판정은 not confirmed다. Unsafe exact 95% 상한도 certification 18.36%, final 13.62%로 strict 5% 위험 기준을 실패했다.

두 독립 분할 모두 23% 이상의 지연 절감은 재현됐다. 따라서 one-forward option-logit 비용 효과는 유지되지만, validation에서 고른 confidence threshold가 과목·난이도 분포 이동에 안정적이지 않은 것이 현재 병목이다. 다음 단계는 GPU 확장이 아니라 subject별 calibration drift, threshold 민감도, calibration error를 CPU에서 분석하고 사전 고정 가능한 보정 규칙을 설계하는 것이다.

### Calibration drift 진단

Lower 정답 순위 AUC는 selection OOF 0.698, certification 0.729, final test 0.755로 개선됐지만 ECE는 0.083, 0.105, 0.129로 악화됐다. Brier score는 0.217, 0.207, 0.201이었다. 즉 분류 순위가 무너진 것이 아니라 확률 수준과 전역 threshold의 calibration이 분할 사이에서 이동했다.

독립 500개를 과목군으로 합쳐 고정 threshold 0.545를 평가하면 STEM과 other는 품질 유지 96.61/97.73%로 통과했지만 humanities와 social sciences는 92.13/94.19%로 실패했다. 과목 구성과 Upper-Lower 오류 상보성 차이가 certification 변동의 주요 원인이다.

사후 threshold 민감도에서 certification과 final test가 동시에 성능 gate를 통과한 연속 구간은 0.48~0.49와 0.59~0.62였다. 이는 확인 결과가 아니며 기존 실패를 재분류하지 않는다. 높은 confidence만 채택하는 보수적 구간의 중앙값 0.60을 새 사전 후보로 고정하고, 기존 700개와 겹치지 않는 MMLU test 500개에서 한 번 더 확인하는 AI-10을 다음 조건부 GPU 단계로 둔다.

### AI-10 비중복 재확인 결과

기존 validation 200과 첫 독립 test 500을 제외하고 seed 2027로 MMLU test 500개를 새로 과목 균형 추출했다. ID 중복은 0개였고 threshold 0.60을 실행 전에 고정했다. Certification-2에서 Lower/Upper 정확도 57.6/73.2%, 채택률 40.8%, 시스템 정확도 69.6%, 품질 유지 95.08%, 지연 절감 16.10%였다. Final-2는 58.0/73.2%, 채택률 37.6%, 시스템 정확도 70.4%, 품질 유지 96.17%, 지연 절감 12.90%였다. 두 분할 모두 성능 gate를 통과해 MMLU one-forward routing의 성능 결과를 confirmed로 승격한다.

Strict risk는 여전히 실패했다. Unsafe exact 95% 상한은 Certification-2 16.06%, Final-2 18.63%다. 따라서 논문에서는 “95% 품질 유지와 10% latency 절감의 독립 재현”과 “5% 조건부 unsafe 인증 실패”를 분리해 보고한다. 다음 단계는 구조와 scorer를 유지한 채 KMMLU 200문항에서 한국어 전이 screening을 수행하는 것이다.

### AI-11 KMMLU 한국어 전이 스크리닝 · 실행 중

2026-09-05에 자연과학·공학·경영·법/사회·건강·한국사를 포함한 20개 과목에서 각 10문항씩, 총 200문항의 결정적 표본을 생성했다. KMMLU 원본 정답은 1~4 인덱스이므로 내부 객관식 스키마에 맞춰 0~3으로 변환하며 단위 테스트로 이 오프셋을 고정했다. MMLU와 동일한 one-forward option-logit 방식으로 Qwen2.5-1.5B FP16, Qwen2.5-7B 4-bit, SmolLM2-360M FP16, SmolLM2-1.7B FP16의 batch 8/1 조건을 실행 중이다. 이 단계는 언어·도메인 전이 가능성을 거르는 screening이며, 결과를 본 뒤 곧바로 확인 주장으로 사용하지 않는다.

8개 조건은 실패 없이 완료됐다. Batch 8에서 Qwen1.5B/Qwen7B 정확도는 35.0%/48.5%, p50 지연은 16.02/65.89ms로 비용비 0.243이었다. 완벽한 selector를 가정한 oracle 시스템은 정확도 60.0%, 정규화 지연 0.893으로 고정 gate를 통과했다. 나머지 모델쌍과 batch 1 조건은 통과하지 못했다. 따라서 다음 GPU 사용은 M0 batch 8의 option-probability 특징 수집에만 제한하고 OOF confidence policy가 실제 품질 95%와 지연 10%를 동시에 만족하는지 확인한다. 절대 Upper 정확도 48.5%가 낮다는 한계 때문에, 성공하더라도 이 단계는 한국어 전이 가능성의 탐색 결과로만 해석한다.

### AI-12 KMMLU OOF confidence gate · 중단 판정

M0의 8개 option-probability 특징으로 5-fold OOF Logistic Regression을 적용한 결과 AUC 0.654, AP 0.560이었다. Upper 품질의 95%를 유지하는 선택점은 threshold 0.675, 채택률 20.0%, 시스템 정확도 46.5%, 품질 유지 95.88%였으나 정규화 지연은 1.043으로 Always Upper보다 4.3% 느렸다. 채택 40건 중 unsafe 9건이며 exact 95% 상한은 35.98%였다. 따라서 10% 지연 절감 gate와 strict risk를 모두 실패한다. 동일 Qwen/Smol 조합과 동일 confidence 특징으로 KMMLU를 더 돌리지 않는다. 다음 한국어 실험은 한국어 성능이 더 높은 Upper/Lower 후보를 50~200문항 option-logit으로 먼저 선별하고, 비용비와 oracle gate를 통과한 경우에만 confidence 학습으로 확장한다.

### AI-13 한국어 특화 Lower 교체 · Oracle gate 통과

공개 HyperCLOVAX-SEED-Text-Instruct-0.5B를 Lower, 기존 Qwen2.5-7B를 Upper로 사용했다. 50문항 사전 검사는 Lower/Upper 정확도 32/42%, 지연비 0.139, oracle 정확도 58%, oracle 정규화 지연 0.819로 네 gate를 모두 통과했다. 200문항 확장에서도 정확도 34.5/48.5%, 지연비 0.094, oracle 정확도 60.5%, oracle 정규화 지연 0.749로 재현됐다. 한국어 특화 소형 Lower로 바꾸면 비용 여유가 크게 늘고 오류 상보성도 유지된다.

EXAONE-3.5-2.4B도 공개 Upper 후보로 내려받았으나 공식 원격 코드가 Transformers 5.12의 causal mask API와 호환되지 않아 `create_causal_mask(input_embeds=...)` 단계에서 중단됐다. 정확도 결과는 생성되지 않았으며 기존 환경 재현성을 지키기 위해 전역 라이브러리 downgrade는 하지 않았다.

### AI-14 OOF confidence · 경계선 탐색 통과

HyperCLOVAX의 8개 option-probability 특징으로 5-fold OOF Logistic Regression을 적용했다. AUC 0.524, AP 0.427로 분류력은 약했지만 Lower 비용비가 0.094로 작아 threshold 0.575에서 채택률 19.5%, 품질 유지 95.88%, 정규화 지연 0.8991을 기록했다. 10% 지연 절감 기준을 0.09%p 여유로 통과한 탐색 결과이므로 독립 확인 없이는 주장하지 않는다.

### AI-15 비중복 KMMLU 500 확인 · 미확정

기존 200문항을 제외하고 20과목에서 각 25문항씩 새로 뽑아 certification/final test 250개씩 고정했다. Selection 200개로 confidence 모델을 학습하고 threshold 0.575와 latency ratio 0.0941을 변경하지 않았다. Certification은 품질 유지 96.52%, 지연 절감 14.19%로 성능 gate를 통과했다. Final test는 품질 유지 97.66%였으나 채택률 18.0%로 지연 절감 8.59%에 그쳐 10% 기준을 1.41%p 실패했다. Strict unsafe exact 95% 상한도 25.11/27.24%로 실패했다. 따라서 한국어 전이는 유망한 near-miss이지만 confirmed로 승격하지 않으며, 이 결과를 보고 threshold를 다시 고르지 않는다.

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
