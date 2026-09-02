# Playground S6E8 — Predicting Smartphone Addiction

**Metric:** ROC-AUC · **Deadline:** 2026-08-31 23:59 UTC · **Target:** `addicted_label` (binary, 70.94% positive)
**Data:** 691,369 train × 12 features · 296,302 test

---

## 1. What the data actually is

The competition data is *synthetic*, generated from a real 7,500-row source dataset.
In the source, the label is close to a lookup table:

```
addicted = 1   if daily_screen_time_hours > 8  or social_media_hours > 4
addicted = 0   if daily_screen_time_hours <= 6 and social_media_hours <= 4
coin flip      otherwise
```

That rule reads AUC 0.9888 on the source. **It reads only 0.865 here** — the generator
smeared the step into a slope. Do not hard-code it.

What the generator produced instead is a *smooth calibrated probability field*: OOF
predictions land on the calibration diagonal, there are zero duplicate rows, and no
nearest-neighbour spike. The label is an irreducible Bernoulli draw from that field.
There is no leak to find — only the field to fit, and a hard Bayes ceiling everyone is
pressed against.

### Missingness is the only train/test difference

Every column is 4–20% missing, at different rates in train and test. Adversarial
validation: **0.564 with NaNs present, 0.498 on complete rows.** There is no value
drift whatsoever. The missingness pattern *is* the entire shift, so drift-correction
schemes have nothing to correct.

---

## 2. The finding that matters: the lattice lookup table

The generator memorised a **per-exact-value probability table** into the discrete
lattice columns. Measured standalone OOF AUC, 5-fold:

| column | as a float | as exact-value target encoding |
|---|---|---|
| **notifications_per_day** | **0.492** — nothing at all | **0.749** |
| **app_opens_per_day** | 0.541 | **0.735** |
| sleep_hours | 0.527 | 0.591 |
| age | 0.502 | 0.550 |
| gaming_hours | 0.622 | 0.633 |
| work_study_hours | 0.655 | 0.667 |

`notifications_per_day` has correlation −0.012 with the target and dead-flat deciles.
It is invisible to any smooth or monotone view. Yet adjacent integers swing from 0.37
to 0.92 — up to **129 binomial standard errors** off their neighbours. It is a lookup
table, not a curve.

The raw-numeric baseline gives the game away: its top two features *by split count* are
`notifications_per_day` (9,373 splits) and `app_opens_per_day` (7,875), against 2,512
for `daily_screen_time_hours`, the strongest genuine predictor. The trees burn their
entire budget rebuilding the table by hand.

Train and test share **100% of values** for notifications (231), app_opens (166),
age (18), gaming (401), sleep (451) — encoding is safe, there are no unseen levels.

---

## 3. Structural constraint in the screen-time columns

`daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours`
holds **exactly** on all 421,427 rows where the four are present (residual min = 0.000).

So `other_h = daily − (social + gaming + work_study)` is the unaccounted screen time,
and it is the one hand-built feature that reliably pays. It also explains a Simpson's
reversal: `gaming_hours` and `work_study_hours` correlate *positively* with the target
marginally (+0.21, +0.25) but flip *negative* once conditioned on the daily total —
they inflate the total while eating the leisure share that actually drives the label.

`weekend_screen_time` (corr 0.80 with daily) is best read as a second noisy measurement
of the same latent screen-time quantity.

---

## 4. Measured results

| step | OOF AUC | delta |
|---|---|---|
| raw numerics (12 features) | 0.963748 | — |
| + engineered features | 0.964772 | +0.00102 |
| **T1: + TE of 9 lattice columns** | 0.967057 | **+0.00229** |
| T2: + categorical TE + value counts | 0.967285 | +0.00023 |
| **T3: + engineered on top** | **0.968335** | +0.00105 |
| XGBoost, 3-seed avg | 0.968603 | best single |
| LightGBM, 3-seed avg | 0.968529 | |
| CatBoost, 3-seed avg | 0.968227 | |
| nnC — deep-narrow 256x3, ReLU | 0.967748 | best NN |
| nnA / nnB / nn | 0.9674 / 0.9673 / 0.9667 | |
| **9-model rank blend** | **0.969113** | **+0.00051** |
| | | **+0.00537 total** |

### Blend weights (greedy forward selection on OOF)

| model | OOF AUC | weight |
|---|---|---|
| XGBoost | 0.968603 | 0.312 |
| **nnC** | 0.967748 | **0.182** |
| T3 (LightGBM + TE) | 0.968335 | 0.104 |
| LightGBM / nnB | 0.9685 / 0.9673 | 0.078 each |
| lgb_L1 / nn / nnA | 0.9648 / 0.9667 / 0.9674 | 0.065 each |
| CatBoost | 0.968227 | 0.052 |

**The four NNs together carry 0.39 of the weight** despite every one of them scoring
*below* every GBDT. They are the only non-tree members and therefore the only ones wrong
in different places. Even the plain raw-feature model at 0.9648 earns 6.5%.

### 10-fold + saturating CPU/GPU run

`exp_all.py` runs both compute streams concurrently (GPU: xgb, cat, nnC, nnB — CPU: lgb,
hgb, lin) at 10 folds. 7 models x 10 folds x 2 seeds in **52 minutes**. Every model
improved on its 5-fold self:

| model | 5-fold | 10-fold |
|---|---|---|
| xgb | 0.968603 | **0.968729** |
| lgb | 0.968529 | **0.968642** |
| cat | 0.968227 | 0.968302 |
| nnC | 0.967748 | **0.968069** |
| nnB | 0.967339 | 0.967547 |
| HistGradientBoosting | — | 0.967955 |
| Logistic on TE features | — | 0.956688 |

Final blend **OOF 0.969269**, weights: xgb10 0.310, nnC10 0.239, lgb10 0.127, nnB10 0.070,
nnC 0.070, xgb 0.070, lgb_L1 0.056, nn 0.028, T3 0.014, cat10 0.014.

Two negatives worth recording. **`hgb` and `lin` both earned zero weight** — the first
tested candidates that failed the diversity test. HistGradientBoosting is another
histogram booster on the same features, so it lands where lgb already is; and the linear
model is simply too weak (0.9567) for its difference to compensate. Being decorrelated is
necessary but not sufficient — a candidate still has to be *good enough* to carry weight.

### Preprocessing review (the decimal-place artifact)

Every float column's **decimal-place count** carries real signal. Controlling for the
value itself, a row whose `daily_screen_time_hours` is a whole number is far less likely
to be positive:

| value band | whole number | 1 dp | 2 dp |
|---|---|---|---|
| (5.89, 6.8] | **0.126** | 0.541 | 0.545 |
| (6.8, 7.77] | **0.190** | 0.779 | 0.723 |
| (7.77, 8.7] | **0.333** | 0.884 | 0.897 |

~28 sigma, and not a value confound. The mechanism: on whole-number rows the **original
hard source rule holds far better** — the rule=0 region is 0.148 positive there vs 0.328
elsewhere. Decimal count is a per-row measure of how hard the generator smeared that row.

**But it is worth almost nothing once target encoding is present (+0.00003)**, because
exact-value TE already separates 7.0 from 7.01 by construction. Community reports of
+0.00088 were measured against a *raw-feature* baseline; against a TE pipeline the
information is already encoded. The same correction applies to the widely-repeated
"behavioral ratios hurt" finding — **dropping them costs -0.00020 here.**

Full ablation (5-fold XGBoost, everything else fixed):

| change | delta | verdict |
|---|---|---|
| model-based imputation + re-encode recovered value | +0.000048 | keep |
| TE smoothing 20 -> 2 (monotone: 2 > 20 > 50 > 200) | +0.000043 | keep |
| decimal-place features | +0.000032 | keep |
| `n_na` missingness count | -0.000008 | drop |
| dropping the behavioral group | -0.000197 | **rejected** |

Cumulative +0.000123 at 5-fold — **which did not survive at 10-fold.** Re-running the
full stack with the improved preprocessing moved the strongest model *backwards*
(xgb 0.968729 -> 0.968721); cat/lgb/nnC/nnB gained +0.00001..+0.00009. All noise.

The one thing that did pay: the two preprocessing variants are **decorrelated enough to
blend**. The new-preprocessing models take 0.45 of the final blend weight and lifted it
0.969269 -> 0.969337, which is more than either variant achieved alone. Preprocessing
diversity behaves exactly like architecture diversity here.

### Pseudo-labelling — closed, and a prediction I got backwards

Two-stage, teacher retrained inside every outer fold so a pseudo-label used while scoring
fold k comes from a model that never saw fold k. (Skipping that is the standard way to get
a beautiful CV number and no leaderboard movement.) Soft labels throughout — a pseudo-label
of 0.62 contributes 0.62 — because thresholding destroys exactly the boundary cases the
technique is meant to exploit.

| arm | OOF | vs base |
|---|---|---|
| `rows` — test rows as extra training data (weight 0.5) | 0.968493 | **+0.000063** |
| `both` | 0.968481 | +0.000051 |
| `base` | 0.968430 | — |
| `te` — test rows reinforce the encoder statistics | 0.968396 | **−0.000034** |

**The hypothesis was wrong in both directions.** `te` was the predicted winner (the lookup
tables are the model; 296k test rows add ~43% more support per cell) and it was the only
arm that *hurt*. `rows`, included as a control and predicted to fail, was the only one that
helped. Why the reasoning failed: the cells that matter are already well-populated
(notifications ~2,700 rows/value), so extra support buys nothing measurable — and the added
counts are the teacher's own predictions, which pull each cell toward what the model already
believed. That is self-confirmation, not information.

Adding all four to the blend moved it 0.969337 -> 0.969338. Pseudo-labelling is closed.

### The central lesson, measured three times

| what was added | blend moved |
|---|---|
| 3 whole GBDT families (lgb + xgb + cat) | 0.968871 -> 0.968985 (**+0.00011**) |
| 3-seed averaging on every GBDT | 0.968985 -> 0.968989 (**+0.00000**) |
| 3 NN architecture variants | 0.968989 -> 0.969113 (**+0.00013**) |
| the *first* NN, alone | 0.968335 -> 0.968871 (**+0.00054**) |

Seed averaging lifted every individual GBDT (xgb 0.968441 -> 0.968603) yet did nothing for
the blend — **the blend was already averaging away that same variance, so the two
techniques overlap rather than stack.** Adding correlated members has hit its ceiling.
Decorrelated *shapes* are the only thing still paying.

The two feature families are additive, not redundant: `other_h` earns its +0.001 both
with and without target encoding, because it encodes a structural fact no amount of
per-value encoding recovers.

### Dead ends — measured, don't repeat

| idea | result |
|---|---|
| Hard-code the source rule | AUC 0.865 |
| LightGBM **native categorical** on lattice columns | 0.9602 — *worse than raw* |
| Missingness indicator flags (numeric columns) | no change; LGB NaN routing already does it |
| Train on the original 7,500-row dataset | every fold gets worse — different label mechanism |
| Blending T1/T2/T3 together | zero gain; ranks ~0.999 correlated |
| 2D binned TE on (daily, social) etc. | zero gain — 0.967740 vs 0.967744; GBDT finds it unaided |

Only the **three categorical** NaN flags carry any gain (`academic_work_impact` largest).

---

## 5. Strategy

**The leaderboard is saturated.** On 2026-08-11: #1 = 0.97124, #49 = 0.97092 — a 0.0003
spread across ~1,380 teams. The public split is 20% of test (~59,260 rows) and resolves
only about **0.00015** between correlated submissions. Most of the visible ordering is
coin flips, and the private LB is the other 80%.

This dictates everything:

1. **Target-encode the lattice**, out-of-fold, leave-inner-fold-out. Keep the raw floats
   too — the three screen-time columns are genuinely smooth and monotone, so both views
   earn their place.
2. **Win on variance reduction, not signal discovery.** Everyone competent is at the
   ceiling. The edge is model-family diversity (LGBM / XGB / CatBoost / embedding NN),
   more folds, more seeds, and **rank-averaged** blending — the families put probabilities
   on different scales, so arithmetic averaging underperforms.
3. **Select final submissions on local CV, never on public LB.** 691k OOF rows resolve
   far finer than the 59k public split. Picking on public score is selecting on noise.

### CV ↔ LB calibration — MEASURED (`src/measure_offset.py`)

An earlier version of this document asserted that OOF *understates* the LB by +0.001–0.002
because the submission averages K fold-models. **That was wrong.** Measured by holding out
138,274 rows, running the normal 10-fold pipeline on the rest, and scoring the averaged
fold-models on the untouched slice — exactly the structure the test set sees:

| | AUC |
|---|---|
| (a) OOF on dev, one model per fold | 0.967068 ← what CV reports |
| (b) one fold-model on untouched holdout | 0.966570 |
| (c) average of all 10 fold-models on holdout | 0.967052 ← what the LB sees |

**Bagging offset (c − a) = −0.000016.** The ensembling gain is real (c − b = +0.000482)
but OOF is optimistic by almost exactly as much (a − b = +0.000498), because each
fold-model early-stops on the very fold it is then scored on. The two cancel.

**So LB ≈ OOF.** Also measured: blend weight-fitting optimism = **+0.000008** (fit weights
on half the OOF, score on the other half), so the blend score is honest too.

### What the public split can and cannot resolve

| quantity | value |
|---|---|
| sd of AUC on a public-sized (59,260-row) slice | 0.000556 |
| noise in a single LB reading | **±0.00109** (95%) |

Smallest CV difference the public LB could confirm, by how alike the two submissions are:

| pair | rho | resolvable @95% |
|---|---|---|
| xgb10 vs Fxgb | 0.9913 | 0.00015 |
| xgb10 vs nnC10 | 0.9644 | 0.00033 |
| xgb10 vs lgb_L0 | 0.9231 | 0.00046 |

Every gain from the last several rounds — +0.00013 (NN diversity), +0.00007 (preprocessing
diversity) — sits **below** even the best case. They are real on 691k OOF rows and
permanently invisible on 59k. Never select on them from the public board.

### CONFIRMED BY SUBMISSION (2026-08-12)

Blend OOF **0.969337** -> public LB **0.97037**. Predicted 0.9693 ± 0.00109; the actual
landed inside the interval at its upper edge.

| component | value |
|---|---|
| bagging offset (nested holdout) | −0.000016 |
| public-subset bias (residual) | **+0.00105** = 1.9 sd of subset spread |
| **observed CV -> LB offset** | **+0.00103** |

**Calibration for all future work: `LB ≈ OOF + 0.00103`** on this public split.

The systematic offset really is ~0 — the whole gap is the fixed public 59,260 rows being
an easy draw. That bias applies to every team equally, so **only LB-vs-LB comparisons are
valid.** The earlier "0.0019 gap to the leader" was OOF-vs-LB, i.e. wrong. Like for like:

| | public LB |
|---|---|
| leader | 0.97124 |
| rank ~50 cutoff | 0.97097 |
| **this blend** | **0.97037** |
| gap to #1 | **0.00087** |

0.00087 exceeds the resolvable threshold for even weakly-correlated submissions (0.00046),
so **the gap is real signal, not noise** — but it is half what it appeared to be, and it
converts to a concrete target: **OOF 0.97021 would match first place.**

### Superseded: the remaining gap, and why only a submission settles it

Blend OOF 0.969337 with offset ~0 implies a public score near 0.9693. The leader sits at
0.97124 — a **~0.0019 gap with no current explanation.** Two candidate causes, and they
need separating:

1. **The leaders genuinely have signal this pipeline lacks.** Then there is real work left.
2. **The public split is a fixed 59,260 rows, not a random one.** The ±0.00109 figure above
   is the spread over *random* slices; the actual public rows are one specific draw. If
   that draw is on the easy side, every team's public score is inflated together, and the
   apparent gap partly dissolves. My OOF is measured on all 691k train rows and carries no
   such bias, so the two numbers are not directly comparable.

A single submission distinguishes these, and nothing local can. That is the whole argument
for submitting before doing any further modelling.

---

## 6. Code

Run everything with Python 3.12 (`C:\Program Files\Python312\python.exe`) — the default
`python` on PATH is 3.14 and has no ML packages.

| file | role |
|---|---|
| `src/common.py` | data loading, feature engineering, CV harness |
| `src/encoders.py` | leakage-safe out-of-fold target encoding |
| `src/pipeline.py` | fold builder — TE + counts + pairs + engineered |
| `src/nn_model.py` | entity-embedding MLP |
| `src/exp_te.py` | feature-set ladder T1→T4 |
| `src/exp_te2.py` | 2D binned TE + naive-Bayes logit aggregates |
| `src/exp_nn.py` | NN cross-validation |
| `src/exp_models.py` | LGBM / XGBoost / CatBoost |
| `src/blend.py` | greedy rank-average blend → `out/submission.csv` |

**Leakage check on the encoder:** train-OOF TE AUC 0.7489 vs honest holdout 0.7470. If it
leaked, train would be far higher. Re-run this check after any encoder change.

---

## 7. Roadmap for the remaining time

Done: two preprocessing variants x 10-fold x 5 model families, 13-model rank blend at
**OOF 0.969337** (`out/submission.csv`).

**Every lever I can pursue locally is now measured and exhausted:**

| lever | best measured gain |
|---|---|
| target-encoding the lattice | **+0.0033** (the one large win) |
| blending decorrelated members | +0.00060 |
| first non-tree architecture (NN) | +0.00054 |
| three more NN architectures | +0.00013 |
| preprocessing as blend diversity | +0.00007 |
| pseudo-labelling | +0.00006 |
| preprocessing quality (evaporated at 10-fold) | +0.00012 -> 0 |
| more GBDTs / seeds / 2D encodings / hgb / linear | ~0 |

Everything still available sums to well under the **0.00087** needed to reach first place,
and that gap is real — it exceeds the 0.00046 resolution even for weakly-correlated pairs.
There is no remaining hypothesis sized for it.

1. **Read the top public notebooks** (needs a Kaggle login — the Code tab will not render
   without one). This is now the highest-value action by a wide margin: the single biggest
   win in this whole project, target encoding at +0.0033, came from a rank-59 writeup. If
   anyone near 0.9712 has published, that is worth more than any further experiment here.
2. **Final submission selection before Aug 31.** Two picks. Primary = the max-CV blend
   (offset ~0 and weight-fitting optimism ~0, both measured, so CV is trustworthy).
   Secondary = the best single model `xgb10`, as insurance against a blend pathology on
   the private rows.
3. Do **not** invest further in: more GBDT variants, seeds, 2D/pair encodings, another
   histogram booster, pseudo-labelling, or chasing individual-model CV.

### Watch the overfitting risk in the blend

The greedy selector now chooses from 19 candidates on 691k OOF rows. That is still a
favourable ratio, but the last few increments (0.969255 -> 0.969269) are smaller than the
public LB can resolve (~0.00015), so treat them as noise rather than progress. If in
doubt, prefer the simpler blend — the top 4 members carry 0.75 of the weight.

### Compute

RTX 3050 Laptop (4 GB) + 16 CPU cores. Measured on real fold data:

| workload | CPU | GPU | speedup |
|---|---|---|---|
| **NN** (per seed per fold) | 450s | ~21s | **~21x** |
| XGBoost (400 trees) | 23.3s | 7.2s | 3.2x |
| CatBoost (400 iters) | 27.9s | 7.5s | 3.7x |
| LightGBM | 14.9s | — | pip wheel is CPU-only |

GPU results are numerically identical (xgb 0.96691 -> 0.96688, cat 0.96625 -> 0.96628).
Set `USE_GPU=0` to fall back. Because LightGBM is stuck on CPU and the other two are on
the GPU, `exp_models.py` runs LightGBM in a `ThreadPoolExecutor` *concurrently* with them
— both release the GIL in native training, so per-fold cost is the max of the two streams
rather than the sum. End to end: 330s/fold for 3 seeds x 3 models, vs 452s/fold for 1
seed x 3 models — **4.1x throughput**.

**VRAM is the binding constraint, not compute.** CatBoost GPU peaks at ~3.2 GB of the
4 GB, so do not run the NN concurrently with the zoo — it will OOM and kill both. Run
them back to back.

### Operational note

This machine sleeps. A 5-fold zoo run once spanned ~15 hours of wall clock for 4.2 hours
of CPU. Every CV script now checkpoints `_partial_oof_*.npy` after each fold — never
write a long run that only saves at the end here. Partial files are deliberately
underscore-prefixed so `blend.py`'s `oof_*.npy` glob ignores them until promoted.

---

## Forum rescan, 2026-08-26 (6 days to go)

### max_bin — closed

Both boosters had run at their default (~255) all along, while `daily_screen_time` has
1389 distinct values, `weekend` 1437, `social` 721 — values were being merged into shared
bins, which is exactly the per-value signal this competition turns on. A forum result
reported +0.0024 on raw columns, so this looked like the best remaining lever.

| config | OOF | vs xgb default |
|---|---|---|
| xgb 4096 | 0.968460 | +0.000031 |
| xgb 1024 | 0.968444 | +0.000014 |
| xgb 2047 | 0.968440 | +0.000010 |
| xgb 256 (default) | 0.968430 | — |
| lgb 1023 / 2047 / 255 | 0.96837 / 0.96836 / 0.96836 | ~0 |

Non-monotone, all inside noise: **+0.00003**, and **zero weight** in the blend. The source
thread predicted exactly this — max_bin and value-encoding are *substitutes*, both doing
the job of separating individual values, so the gain only appears if you are not already
encoding. They measured +0.0005 alongside frequency encoding on 3 columns; this pipeline
target-encodes all 12 plus the imputed ones, so the substitution is complete.

### The public leaderboard is not a skill ranking

The 0.97101 "Neural Network" notebook weights its NN at **1e-6** — it contributes nothing.
The score is two public CSVs the author downloaded and blended, verified byte-for-byte
(Spearman 1.0000000, max absolute difference 0.0000000). The NN's own CV is ~0.94.

So a meaningful part of the 0.00087 gap to first place is **public-submission recycling
plus weight-tuning against the public score** — public-LB overfitting, which reverses on
the private split. This blend is selected on 691k OOF rows with measured weight-fitting
optimism of +0.000008. Do not chase that gap; it is partly an artifact.

### Not implemented: missingness augmentation for the NN

Masking present values to the missing representation during training only: reported
+0.0014 solo, independently reproduced at **+0.00008** pooled from a baseline near ours,
and +0.00003 in that reproducer's stack. Skipped deliberately — the reproducer notes that
if the network also sees derived features, masking the token alone is a no-op because the
net reads the value back off them. Our NN receives engineered features (`other_h`,
`comp_sum`, shares) computed from the raw values, so a correct implementation means
recomputing feature engineering per batch under the mask. Real work for ~+0.0001, which is
below what the leaderboard can resolve.

---

## Final rescan, 2026-08-30 (deadline 2026-08-31 23:59 UTC)

Four more community leads tested. All closed.

| lead | result |
|---|---|
| `max_bin` 256 -> 4096 | +0.00003, zero blend weight |
| periodicity (sin/cos on hour columns) | **-0.00011** — rejected |
| log1p | +0.00001 — noise |
| RealMLP (pytabkit, GPU, 5-fold) | solo **0.965363**, **zero blend weight** |
| missingness augmentation for NN | not implemented — ~+0.0001 for substantial rework |

**The pattern that explains all of them.** `max_bin`, decimal-place features and
periodicity are all mechanisms for letting a model separate individual lattice values.
Exact-value target encoding already does that job completely, so each one measures large
against a raw-feature baseline and ~zero here. Community numbers quoted without stating
whether the baseline encodes values will not transfer to this pipeline.

RealMLP was recommended as clearing 0.97 easily. Untuned `RealMLP_TD_S` at 64 epochs on
this feature matrix scored 0.9654 — below all four hand-rolled NNs. Not a refutation of
their result (they specified "even without lookup", i.e. raw features, where a strong net
has more room), but it does not help here.

### Final state

Blend **OOF 0.969340**, public LB **0.97037** (offset +0.00103, measured and validated).
Ahead of the pipelines visible in the forum at deadline: pure LightGBM 0.96997, XGBoost
0.97007.

Two final picks, both validated (296,302 rows, ids matching sample_submission, no nulls):

| pick | file | OOF |
|---|---|---|
| primary | `out/submission.csv` — 14-model rank blend | 0.969340 |
| secondary | `out/submission_xgb10.csv` — best single model | 0.968729 |

Spearman between them is 0.99776, so the hedge is thin by construction: it protects against
a fault in the *blending*, not against the feature pipeline, which both share.
