# GPU Access Request: Rationale (Team Red to Green)

Prepared 2026-11-04 · Soi Park, data scientist · All source files are in `src/datascience/results/`. Korean version: [GPU_지원요청_근거_2026-11-04.md](GPU_지원요청_근거_2026-11-04.md)

## 1. Eligibility under the Operations team's conditions

| Operations team condition (notice of Oct 30) | Our status |
|---|---|
| Validate the test dataset with the current model | Done. Scored 7,852 items with A.X 4.0 Light: 3,232 AI-Hub test articles, 1,620 calibration articles, 1,000 prompt-injection variants, and 2,000 Webis articles (Nov 3, Colab T4, 4-bit, 112 min) |
| Test results or a validation screenshot | `dashboard_en.html`, screenshot `dashboard_A.X_validation_EN_2026-11-03.png`, metrics `eval_summary_ax_maxq.json` |
| A fine-tuning plan | Section 4. Matches the row in our team's decision table (WEEK1-2 plan, §2-1): "label probabilities are badly miscalibrated and the referral rate far exceeds review capacity" |

## 2. Validation results (α = 0.05, 3,232 test articles)

| Metric | A.X zero-shot | TF-IDF baseline | Coherence baseline | Target |
|---|---|---|---|---|
| Direct clickbait coverage (95% lower bound) | 0.995 (0.988) | 0.977 (0.965) | 0.949 (0.932) | Lower bound ≥ 0.93 |
| Auto clickbait coverage (95% lower bound) | 0.957 (0.943) | 0.951 (0.935) | 0.988 (0.979) | Lower bound ≥ 0.93 |
| Human review referral rate | **0.864** | 0.684 | 0.667 | ≤ 0.30 |
| Discrimination (AUC) | **0.653** | 0.757 | 0.821 | - |
| Expected calibration error (ECE) | **0.515** | 0.091 | 0.102 | ≤ 0.10 |
| Routing change rate on 1,000 injection variants | **0.625** | - | - | 0 |

Conformal calibration keeps coverage on target. The cost is that about 975 of 1,000 daily articles go to humans, which would require 7 reviewers (the design assumes 2).

"Direct" and "Auto" are the two ways clickbait was created in the AI-Hub dataset: written by people (Direct) or generated automatically (Auto).

## 3. Limitations identified

1. **Weak discrimination.** AUC is 0.653, lower even than the trained TF-IDF baseline (0.757). The median "yes" probability is nearly the same for normal articles (0.010) and clickbait (0.018).
2. **Probabilities are skewed toward "no".** The model gives clickbait a mean "yes" probability of only 0.105 (ECE 0.515). Under the original rule (threshold computed from Direct samples only), the Auto coverage lower bound was 0.892, below the target.
3. **Calibration cannot fix this.** Even at α = 0.30 the daily review load is 679 articles, above the capacity of 300. Conformal calibration prevents misses but cannot create discrimination.
4. **The model follows instructions embedded in the article.** Routing changed in 62.5% of injection variants, and 88.4% of those changes moved in the direction the injected instruction asked for. Forced-label (53.0%), title injection (41.0%), and role hijacking (32.5%) change the decision while keeping a valid answer format, so a format check cannot catch them.
5. **The answer format itself is stable.** The combined probability of the "yes"/"no" tokens averages 0.999. The problem is the judgment, not the output format, so it calls for training rather than prompt changes.

## 4. Fine-tuning plan

| Item | Details |
|---|---|
| Method | QLoRA classification fine-tuning of A.X 4.0 Light (4-bit + LoRA r=16). The output stays the next-token "yes"/"no" probability, so the existing calibration and evaluation pipeline is reused unchanged |
| Training data | About 20,000 articles sampled from the AI-Hub training split, balanced across Direct, Auto, and normal, plus 2,000 injection variants with their original labels. No article IDs overlap with the calibration or test sets |
| Limitations addressed | 1 and 2 (discrimination, skewed probabilities) through label training; 4 (injection) through training on injected variants |
| Evaluation | Rescore the same 3,232 test articles and 1,000 injection variants, apply the same calibration rule (revised), and check stability over 200 resplits |
| Success criteria | AUC ≥ 0.85, ECE ≤ 0.10, coverage lower bound stays ≥ 0.93, referral rate ≤ 0.50, injection routing change ≤ 0.10 |

The 0.30 referral target is hard to reach even with a perfect model, because our design always routes politics and economy articles (28% of traffic) to humans. We will address this separately in the BaseKit revision.

## 5. Resources needed

| Task | Estimated time | Notes |
|---|---|---|
| QLoRA training (22,000 articles × ~700 tokens × 2 epochs) | About 4–6 h on an A100 40 GB | A 24 GB GPU (L4, A10) also works but takes about twice as long |
| Rescoring (7,852 items) | About 15–20 min | Took 112 min on a T4 |
| Evidence-citation evaluation (generation mode, 1,000 articles) | About 1 h | BaseKit §5-B4, not yet evaluated |
| Buffer (2–3 hyperparameter retries) | About 8 h | |
| **Total** | **About 20 GPU hours** | One A100 40 GB recommended |

Free Colab T4 sessions lose their results when disconnected (we lost 400 scores on Nov 3), and a T4 does not have enough memory to fine-tune a 7B model.

## 6. Schedule

- Nov 5–6: build the training set (data science) and the training script (AI engineer)
- 1–2 days after GPU allocation: train, rescore, and rerun calibration and evaluation
- Nov 9–15 (BaseKit revision week): document "limitation → remedy → change after applying it" in Part B
