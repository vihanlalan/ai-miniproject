# Methodology

This document explains the design decisions behind SynthDx: the data protocol,
the choice of generative models, the certainty-factor model of the expert system,
and the evaluation design. Results are in [`results/report.md`](../results/report.md)
(auto-generated) and summarised in the README.

> Academic demonstration only. Nothing here is clinical guidance.

---

## 1. Data protocol

**Dataset.** UCI Heart Disease, Cleveland subset (303 patients, 13 features,
binary target `num > 0`). We chose it over Pima Indians Diabetes because it holds
two usable sensitive attributes (sex and age), so the fairness analysis means
something. The file is downloaded from the UCI repository and cached in `data/raw/`.
A bundled copy (`synthdx/data/resources/`) is the offline fallback. The 6 missing
values (`ca`, `thal`) are imputed with the column mode.

**Imbalance simulation.** The natural prevalence is about 46%, so there is little
imbalance to fix. For each seed:

1. A stratified **30% real test set** is held out *first*, at natural prevalence.
2. Positives in the remaining training pool are downsampled until they make up
   **12%** of the training set (`data.positive_fraction`). All negatives are kept.

The test set keeps its natural prevalence on purpose. At 12% it would hold only
about 5 positives, and recall would move in steps of 20 percentage points.
`data.imbalance_test: true` turns on the fully imbalanced variant. The test set is
never augmented, resampled or used for fitting. It is used only to score models,
and as the "non-member" reference in the privacy tests.

**Seeds.** Every seed (0-4) gives a new split, newly trained GANs, a new SMOTE
sample and newly trained models, so the reported standard deviations cover all
sources of variance.

## 2. Generative models

### Why CTGAN
Tabular medical data mixes continuous columns (often multi-modal or skewed, such
as `oldpeak` with many zeros) and categorical columns. A vanilla GAN struggles with
both. CTGAN (Xu et al., 2019) adds two fixes:

* **Mode-specific normalisation**: each continuous column is modelled with a
  variational Gaussian mixture, and values are encoded as (mode, offset within
  the mode).
* **Conditional generator + training-by-sampling**: the generator is conditioned
  on a chosen discrete value, sampled so rare categories are seen often. Through
  the same mechanism we condition on `target = 1` to oversample the minority class.

### Custom conditional GAN (from scratch)
A small PyTorch cGAN (`synthdx/gan/cgan.py`, about 200 lines) shows how the
pieces work:

| Component | Choice | Reason |
|---|---|---|
| Encoding | numerics min-max to [-1, 1], categoricals one-hot | simple and reversible |
| Generator | MLP(z ‖ y) -> tanh (numerics) + per-column softmax | outputs live in the encoded space |
| Categorical outputs | Gumbel-softmax during training | discrete samples that keep gradients |
| Discriminator | MLP(x ‖ y), LeakyReLU, dropout 0.3 | conditioning on the label |
| Loss | non-saturating BCE, one-sided label smoothing (0.9) | stabilises early training |
| Batches | class-balanced sampling | the 12% class is seen in half the batches |

Both generators share one interface, `fit(df)` and `generate(n, condition)`.
Generated rows go through `enforce_schema`, which clips to plausible ranges,
rounds integer columns and snaps categoricals to valid levels. When a condition is
given, rejection sampling guarantees that every returned row has the requested
class. Loss curves are recorded per epoch, and the cGAN can write per-epoch
checkpoints.

**Mode collapse** (the generator covering only part of the data distribution) is
checked through per-column KS / Jensen-Shannon distances, correlation-matrix
differences and the real-vs-synthetic detectability classifier. A collapsed
generator shows high JS distance on categorical columns and is easy to detect.

## 3. Expert system

### Knowledge representation
41 IF-THEN rules in `synthdx/expert/knowledge/rules.yaml`. Each rule has an id,
name, category, priority, conditions, conclusion, certainty factor in [-1, 1] and
a rationale/source note. Conditions support `== != > >= < <= in` on facts, plus
`is true/false` for testing intermediate hypotheses by their CF. The rules reason
in three levels:

1. **Findings** (priority 3): raw data -> `ischemia_evidence`, `perfusion_defect`,
   `vessel_disease`, `exercise_intolerance`, `risk_factor_burden`,
   `structural_abnormality`, each with positive and negative evidence.
2. **Diagnosis** (priority 2): findings -> `heart_disease`.
3. **Recommendation** (priority 1): diagnosis -> referral / risk-factor management.

### Working memory and fuzzy facts
Each patient record becomes facts: raw values (CF 1), discretised categories (age
band, ACC/AHA BP category, NCEP cholesterol category, ST-depression level, ...),
and "fuzzy-ish" facts whose CF is a linear membership degree. For example,
`hypertension` rises from 0 at 125 mm Hg to 1 at 160 mm Hg, and
`low_heart_rate_response` from 0 at 85% to 1 at 65% of age-predicted maximum
heart rate. A borderline value therefore gives weak evidence instead of flipping a
hard switch.

### Inference
Forward chaining with a recognise-act cycle:

1. **Match**: the agenda holds every unfired rule whose premises all hold. A premise
   holds only if its fact's CF >= 0.2, the MYCIN threshold.
2. **Conflict resolution**: highest priority, then highest specificity (most
   premises), then knowledge-base order.
3. **Act**: fire the rule and combine its contribution into the conclusion.

Loop safety comes from *refractoriness* (each rule fires at most once per patient)
plus a `max_cycles` guard. Matching is incremental (a light form of RETE): after a
firing, only rules that test the changed fact are re-matched. This sped inference
up about 11x without changing the results.

### Certainty-factor model (MYCIN / EMYCIN)
* Conjunction: `CF(p1 ∧ p2) = min(CF(p1), CF(p2))`
* Rule firing: `CF_contribution = CF_rule × min(CF_premises)`
* Parallel combination for two pieces of evidence `a`, `b` on the same conclusion:
  * both > 0: `a + b(1 − a)`
  * both < 0: `a + b(1 + a)`
  * mixed: `(a + b) / (1 − min(|a|, |b|))`

Worked example (unit-tested in `tests/test_expert.py`): for the high-risk test
patient, `ischemia_evidence` accumulates R05 (0.5) -> R01 (0.75) -> R03 (0.875) ->
R04 (0.6 × 0.75 = 0.45), giving 0.875 + 0.45 × 0.125 = **0.93125**.

**CF vs probability.** A CF is a measure of *change in belief*, not a probability.
Under Heckerman's (1986) reading, CF = (P(h|e) − P(h)) / (1 − P(h)) for supporting
evidence. CFs are modular: each rule can be written and inspected on its own. The
cost is that the combination rule implicitly assumes conditionally independent
evidence, which is why correlated rules (for example three ischaemia rules) can
overstate certainty. That is a known limitation of MYCIN-style systems.

### Explanation facility
* **How**: the full firing trace in order, showing for each step the premise CF,
  rule CF, contribution and the conclusion's CF before and after.
* **Why not**: unfired rules that were at most one condition short (or, for
  single-premise rules, missed a numeric threshold by under 10%), listing exactly
  which conditions failed and why.

### Calibration per data arm (how the arms reach the expert system)
The hand-written rules do not learn from data. We therefore evaluate three variants:

| Variant | What is fitted on the training arm |
|---|---|
| `expert_fixed` | nothing: hand-written CFs, threshold 0.2 (the same in every arm, a control) |
| `expert_calibrated` | the CF decision threshold that maximises training accuracy |
| `expert_tuned` (stretch goal) | diagnosis-rule CFs re-estimated via Heckerman's formula, shrunk toward the hand-written CF with a pseudo-count of 10, plus the threshold |

Accuracy is the calibration objective because it mirrors what the ML baselines
optimise by default (log-loss with a 0.5 threshold). An imbalanced arm therefore
pushes the threshold toward "no disease", and balancing should help recall,
which is exactly the effect under study. Other objectives (`balanced_accuracy`,
`f1`) are available in `config.yaml`.

## 4. Evaluation design

### Arms
| Arm | Training data | Size (seed mean) |
|---|---|---|
| A_real | real imbalanced training set | ~131 (16 positive) |
| B_ctgan / B_cgan | A + GAN minority rows until 1:1 | ~230 |
| C_smote | A + SMOTE-NC minority rows until 1:1 | ~230 |
| D_ctgan / D_cgan | balanced, fully synthetic set the size of B (TSTR) | ~230 |

SMOTE-NC is used instead of plain SMOTE because the data has 8 categorical
columns; plain SMOTE would interpolate them into invalid values.

**Why TSTR (Train on Synthetic, Test on Real)?** It is the strictest utility test
for synthetic data. If a model trained only on synthetic records works on real
patients, the generator has captured the feature-label relationship, not just the
marginal distributions. It is also the setting where a data owner shares only
synthetic data.

### Metrics
* **Classification**: accuracy, balanced accuracy, precision, recall
  (= minority / disease sensitivity, the headline metric), specificity, F1,
  ROC-AUC, PR-AUC, confusion matrix. ML models use a 0.5 probability threshold.
  The expert system uses its (calibrated) CF threshold, and its CF is the ranking
  score for the AUCs.
* **Fidelity** (a synthetic sample at the training class prior vs real training
  data): per-column KS statistic and Jensen-Shannon distance (base 2, 0 =
  identical), mean/max absolute correlation-matrix difference, and the 5-fold
  random-forest real-vs-synthetic ROC-AUC (0.5 = indistinguishable).
* **Privacy**:
  * DCR, distance to the closest real training record in a standardised one-hot
    space, compared with the same distance for unseen real test rows. A ratio
    well below 1 suggests memorisation.
  * The number of exact copies of training rows.
  * A distance-based membership-inference attack: score = −distance to the
    nearest synthetic row, AUC of members (training rows) vs non-members (test
    rows), computed within each class and averaged so different class mixes
    cannot fake a signal.
* **Fairness** (on the test set): per-group accuracy, recall and selection rate
  for sex and age group (<55 / ≥55). Demographic parity difference = max − min
  selection rate. Equal opportunity difference = max − min true-positive rate.

### Statistics
Five seeds, reported as mean ± std. Each arm is compared with arm A by a **paired
t-test over seeds** (primary) and a Wilcoxon signed-rank test. With n = 5, the
smallest attainable Wilcoxon p-value is 0.0625, so it can never reach 0.05. We do
not correct for multiple comparisons, so p-values near 0.05 should be read as
suggestive only.

## 5. Threats to validity
* **Small data**: 303 patients, about 16 training positives per seed, and test
  groups of tens of patients. Variances are large and fairness gaps are noisy.
* **Test prevalence differs from training prevalence** (46% vs 12%) by design.
  That is a deliberate label shift which penalises models that copy the training
  prior.
* **Knowledge leakage into the rules**: the rules were written by someone who
  knows this dataset's literature (e.g. "asymptomatic chest pain" is predictive in
  Cleveland). The fixed expert system is therefore not a blind prior.
* **GAN training on 16 positives** is close to the lower limit of what any
  generative model can learn. Results may differ on larger datasets.
