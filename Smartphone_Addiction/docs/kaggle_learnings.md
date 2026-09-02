# Kaggle Learnings — Tabular Competition Playbook

Derived from Kaggle Playground **S6E8** (Predicting Smartphone Addiction), finished
**717 / 3532**. Every number below is measured — from our own runs, or from published
winner write-ups scored on the same fold structure. Reference numbers are kept so a
future run can be calibrated against a known case.

**The one-line lesson:** our feature engineering matched 7th place and our single models
were within 0.0003 of theirs. We lost on **ensemble breadth** and on **retiring a good
lead after one bad run**.

---

## Phase 0 — Setup discipline (do first, costs nothing)

- [ ] **Freeze one CV split** (10-fold stratified, fixed seed) and reuse it for every
      model and every meta-model. Without this you cannot honestly blend later.
- [ ] Every model must emit a **complete OOF vector + matching test vector**, saved with
      clear provenance. These are the currency of the entire endgame.
- [ ] **Checkpoint after every fold.** A machine that sleeps turns a 40-minute run into a
      15-hour one; saving only at the end makes it all-or-nothing. We lost a full run this
      way.
- [ ] Run GPU-only and CPU-only models on **concurrent streams** — LightGBM is CPU-only in
      the pip wheel, while XGBoost takes `device="cuda"` and CatBoost `task_type="GPU"`.
      Per-fold cost becomes max(streams) instead of sum: measured **4.1x throughput**.
- [ ] Watch VRAM, not just compute. CatBoost-GPU peaked at 3.2 GB of 4 GB, so the NN could
      not run alongside it without an OOM that kills both jobs.

---

## Phase 1 — Find the generator artifact

Playground data is synthesised from a small real dataset, and the generator leaves
fingerprints. **This is where the single largest gain lives** (+0.0033 on S6E8).

- [ ] **Per-exact-value target rates.** For each numeric column, group by exact value and
      compare its target rate against a rolling-smoothed version of its neighbours.
      Compute `z = (rate - smoothed) / binomial_se`. `std(z) ~ 1` means a genuinely smooth
      field; **`std(z) = 30` means a memorised lookup table.**
      *S6E8:* `notifications_per_day` scored AUC **0.492 as a float** — nothing at all —
      and **0.749 as an exact-value encoding**. Correlation was -0.012 and its deciles were
      flat, so it is invisible to every standard EDA view.
- [ ] **Lattice check.** Are values on an exact grid (multiples of 0.01)? Compare distinct
      count against the grid capacity. Full saturation means encoding is safe; check
      train/test value overlap for unseen levels.
- [ ] **Decimal-place count**, read from the **raw CSV text** — pandas collapses `7.0` and
      `7.00` into one float and destroys the signal entirely.
      *S6E8:* whole-number rows obeyed the original un-smeared source rule far better
      (0.148 vs 0.328 positive in the negative region), ~28 sigma, not a value confound.
- [ ] **Algebraic constraints.** Test whether one column is always >= the sum of others.
      *S6E8:* `daily >= social + gaming + work` held exactly on all 421,427 complete rows.
      The residual `other_h` was the one hand-built feature that reliably paid.
- [ ] **Find the original source dataset and its labelling rule** — for *understanding*,
      not for training. Training on it made every fold worse (different label mechanism);
      hard-coding its rule scored 0.865 against 0.9888 on the source.
- [ ] **Adversarial validation** train vs test, once, with and without NaNs.
      *S6E8:* 0.564 with NaNs, **0.498 on complete rows** — missingness was the entire
      difference, no value drift, so drift correction had nothing to correct.
- [ ] Check for duplicate rows and nearest-neighbour spikes. None on S6E8: the label was an
      irreducible Bernoulli draw from a smooth field, so there was a hard Bayes ceiling and
      no leak to find.

---

## Phase 2 — Feature engineering: what pays, what does not

### Pays

- **Fold-safe exact-value target encoding on every column** (leave-inner-fold-out).
  7th place measured **+0.00191 on XGBoost from going 4 to 9 encoded columns alone.**
- **Value counts / train+test frequencies** alongside each encoding.
- **Structural residuals** derived from generator constraints (`other_h`).
- **Decimal / rounding artifacts** — subject to the substitution rule below.
- **TE smoothing is a real hyperparameter.** We set it to 20 in the first hour and never
  revisited it across the entire competition. Lower was monotonically better
  (2 > 20 > 50 > 200).
- **Model-based imputation of strong columns, then re-encode the recovered value** back
  onto the lattice, so the per-value lookup exists for rows that previously had none.
  (`daily_screen_time` recovered at R2 = 0.818.)

### Does not pay (measured by us and independently by 7th place)

- Hand-built domain ratios, reconstructed totals, kNN features — "almost nothing" once
  encoding is in place.
- `n_na` / missingness-count features. Missingness was MCAR; the count scored AUC 0.502.
- Hard-coding the source rule (0.865).
- **LightGBM native categorical** on high-cardinality lattice columns — actively *worse*
  than raw numeric (0.9602 vs 0.9637). Its one-vs-rest grouping overfits 231 levels.
  Target encoding is the right tool, not native categorical.
- Pseudo-labelling: +0.00006 at best. The encoder-enrichment variant was *negative*
  (-0.00003) because the added counts are the teacher's own predictions pulling each cell
  toward what the model already believed — self-confirmation, not information.

### The substitution rule — how to read any forum claim

`max_bin`, decimal features, frequency encoding and sin/cos periodicity are all
**mechanisms for letting a model separate individual values**. Exact-value target encoding
does that job completely, so each measures large against a raw-feature baseline and
**about zero against a TE pipeline**:

| lead | claimed elsewhere | ours, with TE |
|---|---|---|
| max_bin 255 to 2047 | +0.0024 | +0.00003 |
| decimal-place features | +0.00088 | +0.00003 |
| periodicity sin/cos | "certain to push past 0.97" | **-0.00011** |
| "behavioural ratios hurt" | -0.00023 | **+0.00020 to keep them** |

**Always ask what baseline a quoted number was measured against.** If it did not encode
values, the gain will not transfer. This held on four separate leads.

---

## Phase 3 — The model zoo (this is where we lost)

We ran **5 families**. 7th place ran **~20**. Their published level-1 OOF table, with ours
beside it:

| family | 7th place | ours | note |
|---|---|---|---|
| XGBoost | 0.969080 | 0.968729 | |
| **TabM** | 0.968948 | *never tried* | **try this first next time** |
| **RealMLP** | 0.968781 | 0.965363 | we mis-ran it — see below |
| LightGBM | 0.968779 | 0.968653 | |
| CatBoost | 0.968677 | 0.968395 | |
| Lookup-Transformer | 0.968672 | — | |
| FT-Transformer | 0.968558 | — | |
| HistGradientBoosting | 0.968215 | 0.967955 | earned zero blend weight |
| EBM | 0.968066 | — | |
| AutoInt | 0.967759 | — | |
| MLP (hand-rolled) | 0.967608 | **0.968106** | **ours was better** |
| Spline-Transformer | 0.967516 | — | |
| XGBoost-DART | 0.967236 | — | naturally decorrelated |
| YDF / GANDALF / GATE / DANet / NODE / TabICL | 0.9656–0.9671 | — | weak but diverse |

**Install list for next time:**
`pytabkit` (RealMLP, TabM) · `pytorch-tabular` (FT-Transformer, GANDALF, GATE, DANet,
AutoInt, NODE, Category-Embedding MLP) · `interpret` (EBM) · `ydf`.

### The RealMLP lesson — do not close a lead on one run

We ran `RealMLP_TD_S` — the **small** variant — at 64 epochs, default settings, one
configuration. It scored 0.9654 and we reported the lead closed.

7th place got **0.968781** from RealMLP. The winner got a **single RealMLP at
CV 0.97070 / LB 0.97174 — one model that beat our entire 14-model blend.**

We even wrote the correct caveat at the time ("evidence about untuned RealMLP on my
feature matrix, not a refutation") and then acted as though it were a refutation.

**Rule:** when a credible source names a specific method and your first run lands ~0.003
below every other model in your stack, that asymmetry means *misconfigured*, not
*disproven*. Try the full-size variant and a second configuration before retiring it.
Contrast with genuinely dead leads (max_bin, periodicity): those failed **with an
explanation**. "It failed and I know why" is closed; "it failed and the number looks
anomalous" is not.

---

## Phase 4 — Ensembling (the largest single gap)

| | best single | final blend | gain |
|---|---|---|---|
| 7th place | 0.969080 | 0.970879 | **+0.0018** |
| us | 0.968729 | 0.969340 | **+0.00061** |

Three times the extraction. That difference alone was ~0.0012 of our ~0.0015 shortfall.

- [ ] **Breadth beats strength.** Judge a candidate by how much it *decorrelates* from the
      current blend, not by its solo score — but decorrelation is necessary, **not
      sufficient**: it must also be strong enough to carry weight. HistGradientBoosting
      (another histogram booster, lands where LightGBM already is) and a logistic model at
      0.9567 (too weak) both earned **exactly zero**.
      7th place quantified it: 50 weak but *diverse* models added +0.00006 to the blend;
      a random same-sized control added +0.00001.
- [ ] **Preprocessing variants count as diversity.** Re-running the same 5 models on a
      second feature recipe earned **0.45 of our final blend weight** and beat either
      recipe alone — even though the new recipe was individually slightly *worse*.
- [ ] **Use a cross-fitted linear meta-model, not greedy forward selection.** 7th place
      rank-normalised every stream and averaged six logistic regressions
      (C = 0.01, 0.03, 0.1, 0.3, 1, 3), then re-ranked. Their weakest member alone still
      improved the bag.
- [ ] **Allow negative weights.** Subtracting 25% of one stack and 10% of another gained
      them +0.00003. Greedy forward selection is non-negative by construction and cannot
      express "this direction is over-represented" at all.
- [ ] **Fit blend weights on 9 folds and evaluate on the 10th.** Never let the rows you
      score on choose their own weights.
- [ ] **Collect public OOF/test pairs** where competitors share them, with credit.
      **350 of 7th place's 556 streams were community-shared.** Requires a Kaggle login —
      worth having set up before the competition starts.
- [ ] Seed averaging is **not** a substitute for diversity. It lifted every individual
      model (xgb 0.968441 to 0.968603) and moved the blend by **+0.000004** — the blend was
      already averaging away that same variance.

---

## Phase 5 — Validation and submission discipline

- [ ] **Measure the CV↔LB offset, never assume it.** Hold out 20%, run the K-fold pipeline
      on the remaining 80%, then score the *averaged* fold-models on the untouched slice —
      structurally identical to what the submission does to the test set.
      *S6E8:* bagging offset **-0.000016**, not the +0.001 to +0.002 we assumed for most of
      the competition. The ensembling gain (+0.00048) is cancelled by OOF optimism
      (+0.00050), because each fold-model early-stops on the fold it is then scored on.
      **LB ~= OOF.** Validated against a real submission (predicted 0.9693 +/- 0.0011,
      actual 0.97037).
- [ ] **Measure blend weight-fitting optimism** — fit weights on half the OOF, score the
      other half. Ours was +0.000008 (negligible), but it is in-sample by construction and
      deserves a number whenever claimed gains are ~1e-4.
- [ ] **Know what the public LB can resolve.** Resample public-sized slices from your OOF.
      *S6E8:* sd 0.000556, so a single reading carries **+/-0.00109**. For two correlated
      submissions, `sd(gap) = sd(move) * sqrt(2(1-rho))`, giving resolvable gaps of
      0.00015 (rho=0.99) to 0.00046 (rho=0.92). Most late-stage gains are permanently
      invisible there. **Never select on the public score.**
- [ ] **Compare like with like.** Comparing our OOF against someone else's LB made a
      0.00087 gap look like 0.0019. The public split is a fixed subset whose bias hits
      every team equally, so only LB-vs-LB is meaningful. Once you have one submission,
      `LB ~= OOF + observed_offset` calibrates every future CV number.
- [ ] **Sanity-check the leaderboard itself.** On S6E8 a top public "Neural Network"
      notebook weighted its NN at 1e-6 and was really two downloaded public CSVs blended.
      A meaningful part of the visible top is recycled submissions tuned on public score.
- [ ] **Two final picks:** the max-CV blend as primary, plus one genuinely different bet.
      Note that a best-single-model hedge is thin — ours correlated 0.99776 with the blend,
      so it protected against a blending fault but not against the shared feature pipeline.
- [ ] Validate every submission against `sample_submission`: column names, row count, id
      order, no nulls, all values finite.

---

## Time allocation, in hindsight

Roughly what we should have done differently with the same hours:

| activity | we spent | should spend |
|---|---|---|
| find the generator artifact | ~15% | 15% — this was right |
| feature engineering beyond it | ~35% | **10%** — diminishing returns hit fast |
| breadth of model families | ~15% | **45%** — the actual gap |
| meta-model / stacking design | ~5% | **20%** |
| validation & calibration | ~20% | 10% — valuable, but over-invested |
| chasing forum leads | ~10% | 5% — apply the substitution rule first |

---

## Reusable code from S6E8 (`src/`)

| file | what it does |
|---|---|
| `encoders.py` | leave-inner-fold-out target encoding; sort-based dense mapping handles sparse interaction keys; `oof_target_encode_extra` supports soft pseudo-labels |
| `features2.py` | toggleable feature groups (structural / decimals / behavioural / rule / weekend) for clean ablation |
| `impute.py` | model-based imputation respecting generator constraints, snapped back to the lattice |
| `pipeline.py` | per-fold builder: TE + counts + engineered |
| `nn_model.py` | entity-embedding MLP, configurable architecture, GPU-resident data |
| `exp_all.py` | concurrent CPU+GPU multi-family runner with per-fold checkpointing |
| `blend.py` | greedy rank blend — **replace with cross-fitted LogReg allowing negative weights** |
| `measure_offset.py` | nested-holdout CV↔LB offset, noise resampling, pairwise resolvability |
| `blend_honesty.py` | split-half blend weight-fitting optimism |

**Highest-value upgrade for next time:** rewrite `blend.py` as a cross-fitted
rank-LogReg C-bag with negative weights permitted, and expand the model zoo before
spending any further time on features.
