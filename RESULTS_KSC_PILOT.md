# KSC Pilot Results

## Scope

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

