# Predicting Smartphone Addiction — Kaggle Playground S6E8

Solution for [Kaggle Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8).
Metric: **ROC-AUC**. Final rank **717 / 3532**.

| | |
|---|---|
| Best single model | XGBoost, 10-fold, OOF **0.968729** |
| Final blend | 14-model rank blend, OOF **0.969340** |
| Public LB | **0.97037** |
| Winner (reference) | 0.97176 private |

---

## The core finding

The competition data is synthetic, generated from a real 7,500-row source dataset. The
generator **memorised a per-exact-value probability table** into the discrete lattice
columns:

| column | as a float | as exact-value target encoding |
|---|---|---|
| `notifications_per_day` | **0.492** — nothing at all | **0.749** |
| `app_opens_per_day` | 0.541 | **0.735** |
| `sleep_hours` | 0.527 | 0.591 |
| `age` | 0.502 | 0.550 |

`notifications_per_day` has correlation −0.012 with the target and dead-flat deciles — it is
invisible to every standard EDA view. Yet adjacent integers swing from 0.37 to 0.92, up to
**129 binomial standard errors** apart.

A raw-numeric LightGBM confirms the mechanism: its top two features *by split count* are
`notifications_per_day` (9,373 splits) and `app_opens_per_day` (7,875), against 2,512 for
`daily_screen_time_hours`, the strongest genuine predictor. The trees spend their entire
budget rebuilding the lookup table by hand. **Out-of-fold target encoding hands it over
directly: +0.0033 AUC**, versus +0.0010 for all conventional feature engineering combined.

### Contribution of each step

| step | OOF AUC | delta |
|---|---|---|
| raw numerics (12 features) | 0.963748 | — |
| + engineered features | 0.964772 | +0.00102 |
| **+ exact-value target encoding** | 0.967057 | **+0.00229** |
| + categorical TE + value counts | 0.967285 | +0.00023 |
| + engineered on top | 0.968335 | +0.00105 |
| + 10-fold, seed averaging, GPU | 0.968729 | +0.00039 |
| **+ 14-model rank blend** | **0.969340** | **+0.00061** |

---

## Reproducing

1. **Get the data.** Download `train.csv`, `test.csv`, `sample_submission.csv` from the
   [competition data page](https://www.kaggle.com/competitions/playground-series-s6e8/data)
   into a `data/` folder beside this README. The CSVs are not committed here.

   ```
   Smartphone_Addiction/
     data/          <- train.csv, test.csv, sample_submission.csv  (you provide)
     notebooks/     <- the end-to-end solution
     src/           <- module form of the same pipeline, plus every experiment
     out/           <- submissions and measurement reports
     docs/          <- full write-up and the cross-competition playbook
   ```

2. **Environment** (Python 3.12):

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; use /cpu otherwise
   pip install -r requirements.txt
   ```

   or `conda env create -f environment.yml`.

3. **Run** `notebooks/smartphone_addiction_final.ipynb` top to bottom. Set `QUICK = True`
   in the config cell for a fast 5-fold single-seed smoke run.

**Runtime:** roughly 2 hours for the full pipeline on an RTX 3050 (4 GB) plus 16 CPU cores;
a few minutes with `QUICK = True`.

**Note on VRAM:** CatBoost-GPU peaks near 3.2 GB. On a 4 GB card, do not run the notebook
and a separate GPU job at the same time — they will OOM and both die.

---

## What is here

### `notebooks/smartphone_addiction_final.ipynb`
The complete solution end to end: the per-value diagnostic that motivates everything,
feature engineering, leakage-safe target encoding, the five-model zoo on concurrent
CPU/GPU streams, rank blending, and submission with validation.

### `src/`
The same pipeline as importable modules, plus every experiment run during the competition
(kept deliberately — the negative results are the useful part).

| file | role |
|---|---|
| `encoders.py` | leave-inner-fold-out target encoding; sort-based dense mapping so sparse interaction keys do not blow up memory; soft-pseudo-label variant |
| `features2.py` | toggleable feature groups (structural / decimals / behavioural / rule / weekend) for clean ablation |
| `impute.py` | model-based imputation respecting generator constraints, snapped back to the lattice |
| `nn_model.py` | entity-embedding MLP, configurable architecture, GPU-resident data |
| `exp_final.py` | the production 10-fold run, concurrent CPU+GPU, per-fold checkpointing |
| `blend.py` | greedy rank blend → `submission.csv` |
| `measure_offset.py` | nested-holdout CV↔LB offset, noise resampling, pairwise resolvability |
| `blend_honesty.py` | split-half blend weight-fitting optimism |
| `exp_*.py` | the ablations: preprocessing, pseudo-labelling, max_bin, periodicity, RealMLP |

### `out/`
`submission.csv` (the 14-model blend, 0.97037 public) and `submission_xgb10.csv` (best
single model), plus the JSON reports from every measurement run.

### `docs/`
- `STRATEGY.md` — the full competition write-up, including every dead end with its number
- `kaggle_learnings.md` — a general playbook for future tabular competitions

---

## What did not work (all measured)

| idea | result |
|---|---|
| Hard-coding the source rule (`daily>8 or social>4`) | 0.865 — the generator smeared it away |
| LightGBM native categorical on lattice columns | 0.9602, *worse* than raw numeric |
| Numeric missingness flags / `n_na` | ~0 — missingness is MCAR (AUC 0.502 alone) |
| Training on the original 7,500-row dataset | every fold worse — different label mechanism |
| 2D binned target encoding | 0.967740 vs 0.967744 |
| `max_bin` 256 → 4096 | +0.00003, zero blend weight |
| Periodicity (sin/cos) features | **−0.00011** |
| Pseudo-labelling (4 variants) | +0.00006 at best |
| Seed averaging on top of blending | +0.000004 |
| HistGradientBoosting, logistic on TE features | zero blend weight |

**The unifying explanation:** `max_bin`, decimal features, frequency encoding and
periodicity are all mechanisms for *separating individual values*. Exact-value target
encoding already does that completely, so each one measures large against a raw-feature
baseline and about zero here. Always ask what baseline a quoted number was measured against.

---

## Post-mortem: why 717th

Against 7th place's published level-1 OOF table (same fold structure), the individual models
were competitive — XGBoost, LightGBM and CatBoost within 0.0003, and the hand-rolled MLP
actually *better* than theirs (0.968106 vs 0.967608).

The gap was **ensembling**:

| | best single | final blend | gain |
|---|---|---|---|
| 7th place | 0.969080 | 0.970879 | **+0.0018** |
| this solution | 0.968729 | 0.969340 | **+0.00061** |

Three times the extraction, accounting for most of the shortfall. They ran ~20 model
families to our 5, pooled 556 prediction streams to our 14 (350 of theirs community-shared),
and used six cross-fitted logistic regressions on rank-normalised predictions that **allowed
negative weights** — greedy forward selection is non-negative by construction and cannot
express "this direction is over-represented".

A second, self-inflicted miss: RealMLP was retired after a single untuned run of the *small*
variant (0.9654). 7th place got 0.9688 from it, and the competition winner got a single
RealMLP at CV 0.97070 / LB 0.97174 — one model that beat this entire blend.

Both lessons are written up in `docs/kaggle_learnings.md`.

---

## Validation notes

- **CV↔LB offset was measured, not assumed.** Nested holdout: run the K-fold pipeline on
  80%, score the averaged fold-models on an untouched 20%. Bagging offset came out at
  **−0.000016** — the ensembling gain (+0.00048) is cancelled by OOF optimism (+0.00050),
  since each fold-model early-stops on the fold it is scored on. So `LB ≈ OOF`.
- **Blend weight-fitting optimism** (fit on half the OOF, score the other half): +0.000008.
- **Public-LB resolution:** sd 0.000556 on a 59,260-row slice, so a single reading carries
  ±0.00109. Correlated submissions resolve at 0.00015–0.00046. Most late-stage gains are
  invisible there — never select on the public score.

## Licence

MIT. The competition data itself is not redistributed here.
