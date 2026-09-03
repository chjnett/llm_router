# KSC Pilot and Performance-focused Results

## Current main result

The current paper direction is performance-first: preserve most Upper-model quality while
reducing cascade latency and normalized inference cost. Safety is reported as a secondary
limitation rather than weakened after observing the result.

| Fixed method / split | Accuracy | Unsafe | Normalized cost | Cost reduction |
|---|---:|---:|---:|---:|
| C3 / risk certification (250) | 88.80% | 2.00% | 1.008 | -0.77% |
| Answer-only / risk certification (250) | 88.00% | 2.80% | 0.777 | 22.29% |
| C3 / official test (300) | 87.67% | 1.00% | 1.045 | -4.45% |
| Answer-only / official test (300) | 85.67% | 3.00% | 0.806 | 19.42% |

The answer-only policy uses thresholds 0.12/0.80 selected on a separate 250-row split. Its
exact one-sided 95% unsafe-risk upper bound on the disjoint certification split is 5.195%,
which misses the prespecified 5% criterion by 0.195 percentage points. This is a near-miss,
not a safety certificate.

### Same-GPU latency benchmark

- Hardware: NVIDIA RTX 3090; 4-bit Qwen2.5-1.5B-Instruct
- Protocol: identical 128 questions, batch size 16, two repeats per condition, ABBA order
- Full second pass: 492.97 ms/item and 124.55 generated tokens/item
- Answer-only verifier: 29.15 ms/item and 5.40 generated tokens/item
- Reduction: 94.09% latency; throughput increased from 2.03 to 34.32 items/s

### Paired statistical comparison against C3

| Split | Accuracy delta | Paired bootstrap 95% CI | McNemar p | Cost delta | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Risk certification | -0.80%p | [-2.40, +0.80]%p | 0.625 | -23.06%p | [-28.70, -17.38]%p |
| Official test | -2.00%p | [-4.33, +0.01]%p | 0.146 | -23.87%p | [-29.58, -18.27]%p |

The paired tests show a clear cost reduction and no statistically significant accuracy
difference at these sample sizes. This does not establish accuracy equivalence or
non-inferiority. The defensible claim is that answer-only verification improves the observed
performance-cost frontier; larger fresh samples are required for a formal non-inferiority or
safety claim.

## KSC 초록 초안

대규모 언어 모델은 높은 추론 성능을 제공하지만 모든 요청을 대형 모델에 할당하면 계산
비용과 지연시간이 증가한다. 본 연구는 소형 모델의 질문 임베딩뿐 아니라 실제 생성
출력의 confidence와 짧은 독립 검증 결과를 이용하는 적응형 SLM–LLM 라우팅 방법을
제안한다. 제안 방법은 Qwen2.5-1.5B가 먼저 답을 생성하고, 경계 사례에 대해서만 설명을
제외한 answer-only 검증을 수행한 후 두 숫자 답의 일치 여부에 따라 Qwen2.5-7B로
escalation한다. GSM8K에서 학습한 confidence router를 SVAMP로 전이하여 분리된 정책
선택·인증 split과 공식 test에서 평가하였다. RTX 3090 동일 조건 실험에서 answer-only
검증은 기존 두 번째 전체 풀이보다 문항당 지연시간을 492.97 ms에서 29.15 ms로 94.09%
감소시켰다. 실측 지연시간으로 환산한 cascade 비용은 독립 인증 split과 공식 test에서
Always-Upper 대비 각각 22.29%와 19.42% 감소했으며, 정확도는 각각 88.00%와 85.67%를
기록하였다. C3 기준선과의 exact McNemar 검정에서는 유의한 정확도 차이가 관측되지
않았다(p=0.625, p=0.146). 한편 95% unsafe-risk 상한은 5.195%로 사전 설정한 5% 기준을
근소하게 충족하지 못했다. 결과적으로 제안 방법은 관측된 성능–비용 frontier를
개선하지만, 안전 보장과 정확도 비열등성의 확증에는 더 큰 독립 표본이 필요하다.

## Historical 3B pilot scope

- Date: 2026-09-02
- Hardware: NVIDIA RTX 3090 24 GB
- Dataset: 2,000 GSM8K train examples
- Split: router train 1,000 / distillation 400 / validation 200 / test 400
- Lower: Qwen2.5-1.5B-Instruct, 4-bit inference
- Pilot Upper: Qwen2.5-3B-Instruct, 4-bit inference
- Frozen embedding: BGE-small-en-v1.5
- Distillation: 16 verified resolvable failures, two epochs, LoRA rank 16
- Router hyperparameters and operating thresholds were selected on validation only.

The planned 7B Upper was replaced by 3B for this storage-constrained pilot. These results test
the mechanism and are not the final KSC confirmation experiment.

## Baseline

| System | Test accuracy | Lower coverage | Upper call rate | Normalized cost |
|---|---:|---:|---:|---:|
| Always Lower | 65.00% | 100.00% | 0.00% | 0.500 |
| Always Upper | 76.25% | 0.00% | 100.00% | 1.000 |
| Oracle | 86.00% | 65.00% | 35.00% | 0.675 |
| LR | 74.00% | 14.25% | 85.75% | 0.929 |
| kNN (k=3) | 72.25% | 45.25% | 54.75% | 0.774 |

The original LR router had eligible-example AUC 0.549. Frozen BGE query embeddings therefore
provide only weak separability for correctness on this single-domain GSM8K pilot.

## H2: random versus cluster-balanced distillation

The distillation split contained 78 resolvable failures. A 16-example budget was used so the
sampling methods had a real diversity contrast: random sampling covered 7/5/7 clusters for
seeds 42/43/44, while balanced sampling covered all eight clusters for every seed.

| Method | Seed 42 | Seed 43 | Seed 44 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Random KD | 68.50% | 66.50% | 72.25% | 69.08% | 2.92% |
| Cluster-balanced KD | 70.00% | 65.50% | 70.25% | 68.58% | 2.67% |

Both methods improved over the original Lower's 65.00%, but cluster-balanced KD did not beat
random KD on average. H2 is not supported by this pilot. The budget is very small, so the result
should be treated as a variance warning rather than a definitive negative conclusion.

## H3: capability shift and router adaptation

Cluster-balanced seed 42 was used as the representative adaptation condition.

| Split | Lower before | Lower after | Label shift on common eligible examples |
|---|---:|---:|---:|
| Router train | 61.60% | 66.00% | 25.13% |
| Validation | 64.00% | 69.00% | 21.82% |
| Test | 65.00% | 70.00% | 22.89% |

| Router condition | Test accuracy | Lower coverage | Upper call rate | Normalized cost |
|---|---:|---:|---:|---:|
| LR, old Lower | 74.00% | 14.25% | 85.75% | 0.929 |
| LR, new Lower, no adaptation | 75.00% | 14.25% | 85.75% | 0.929 |
| LR retrained | 74.50% | 46.50% | 53.50% | 0.768 |
| kNN, old Lower | 72.25% | 45.25% | 54.75% | 0.774 |
| kNN, new Lower, no adaptation | 73.50% | 45.25% | 54.75% | 0.774 |
| kNN pool rebuilt | 73.50% | 53.75% | 46.25% | 0.731 |

H3 is partially supported. Distillation caused a substantial routing-label shift, and both
adaptation mechanisms reduced cost. LR retraining produced the better quality/cost point;
kNN rebuilding produced a smaller cost gain without losing accuracy relative to its own
no-adaptation condition.

## Interpretation and required confirmation

This pilot validates the core mechanism but is not sufficient by itself for a final KSC claim.
The confirmation run should:

1. Restore Qwen2.5-7B-Instruct as Upper and the 4.667 normalized parameter-cost ratio.
2. Use all 7,473 GSM8K training examples so the distillation pool supports a larger budget.
3. Repeat adaptation end-to-end for all three seeds, not only KD test accuracy.
4. Add bootstrap confidence intervals and paired significance tests.
5. Report the full validation and test quality-cost curves; validation-selected points did not
   transfer perfectly to test in this pilot.
6. Add at least one second task family for the final paper to test whether routing is learning
   difficulty rather than GSM8K-specific lexical patterns.
