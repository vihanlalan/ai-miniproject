# Demo script (5-7 minutes)

**Before the demo**
- Run `make demo` (or `python tasks.py demo`) from the project root and open
  http://localhost:8501. Everything loads from cached results and models; no
  training happens unless you press *Train now*.
- Keep `results/report.md` open in a second tab as a backup.

---

### 0:00 - Overview (45 s)
- Read out the research question from the blue callout.
- Point at the pipeline diagram: *real data -> held-out test set that is never
  touched -> GANs -> four training arms -> ML models and the expert system ->
  evaluation*.
- Point at the four stat cards: 303 records, training positives cut to 12%, 41
  rules, 5 seeds.
- Point at the disclaimer banner: academic demo, not for clinical use.

### 0:45 - Data Explorer (45 s)
- Class-balance chart: the full data is about 46% positive. *We deliberately cut
  positives to 12% in training only, so augmentation has something to fix, and we
  keep the test set at natural prevalence so recall can be measured.*
- Pick *Max heart rate*, then *Chest pain type*, to show that the features
  separate the classes.

### 1:30 - GAN Lab (1 min 15 s)
- Model = CTGAN, Source = Cached. Show the loss curves: *GAN losses don't fall
  steadily like a supervised loss; the two networks compete and settle into a band.*
- Generate 200 records with Class = *Disease (1)*. Every row has target 1
  (conditional generation). Click **Download CSV**.
- Real vs synthetic: pick *Cholesterol*, then show the correlation heatmaps side
  by side and the per-column KS / JS table.
- Optional, if time allows: switch to *Custom cGAN*, then *Quick retrain*, 30
  epochs, *Train now* (a few seconds). *This is our own PyTorch GAN, about 200
  readable lines.*

### 2:45 - Expert System Diagnosis (1 min 30 s)
- Click **Random real patient**, then **Diagnose**.
- Show the label, risk pill and CF gauge (the red tick is the decision threshold).
- **How** tab: *rules fire in priority order: findings first, then diagnosis, then
  recommendations. Each row shows rule CF × premise CF and how the conclusion's CF
  grows by MYCIN parallel combination.*
- **Why not** tab: open one near miss and read the failed condition.
- **What-if** tab: drag *ST depression* up, or *Major vessels* from 0 to 2, and
  watch the CF move. Use the sweep chart to show where the diagnosis flips.

### 4:15 - Experiments Dashboard (1 min)
- Metric = recall. *Each bar is the mean over 5 seeds on the same real test set;
  the whiskers are the std.*
- Compare arm A with B / C / D for one model, then for the expert system.
- Seed-variance strip: *one dot per seed; small data means wide spread.*
- ROC curves, then the significance table: *paired t-test against arm A.*
- **State the honest finding** (see README "Results summary").

### 5:15 - Fairness & Privacy (45 s)
- Fairness: attribute = sex, read the computed interpretation lines and stress
  that test groups are small.
- Privacy tab: read the plain-English lines (MIA AUC, DCR ratio, duplicates,
  detectability). *Synthetic does not automatically mean private.*

### 6:00 - Rule Browser (30 s)
- Search "vessel" and show the CFs and rationales. If results exist, show the
  hand-written vs tuned CF chart.
- Close: *Three AI paradigms in one pipeline: generative models, symbolic
  reasoning with uncertainty, and statistical ML, evaluated honestly.*

---

## Likely questions and concise answers

**1. Why CTGAN rather than a plain GAN?**
Tabular data mixes skewed continuous columns and categorical columns. CTGAN models
each continuous column with a Gaussian mixture (mode-specific normalisation) and
conditions the generator on discrete values with training-by-sampling. That same
conditioning lets us generate only the minority class. Our own cGAN is the
transparent baseline that shows the basic mechanics.

**2. What is mode collapse, and how would you detect it here?**
The generator produces only a few kinds of samples instead of the whole
distribution. We check it with per-column Jensen-Shannon distances (collapsed
categoricals score high), correlation differences, and a real-vs-synthetic
classifier (a collapsed generator is easy to detect). Class-balanced batches and
label smoothing in the cGAN reduce the risk.

**3. Why certainty factors instead of probabilities?**
CFs encode a *change in belief* from each rule independently, so experts can write
and audit rules one at a time without a full joint distribution. MYCIN's
combination formula is cheap and order-independent. The downside is that it
assumes conditionally independent evidence and is not a calibrated probability.
That is why we calibrate the decision threshold on data and report ROC-AUC, which
only needs a correct ranking.

**4. Walk me through one CF calculation.**
R04 has rule CF 0.6, and its premise *significant ST depression* has fuzzy
membership 0.75, so the contribution is 0.6 × 0.75 = 0.45. Ischaemia evidence was
already 0.875, so the combined value is 0.875 + 0.45 × (1 − 0.875) = 0.931. This
example is unit-tested.

**5. How does the expert system benefit from synthetic data if its rules are
hand-written?**
The rules stay fixed. What changes is calibration: the decision threshold (and, in
the tuned variant, the CFs of the diagnosis rules via Heckerman's formula) is fitted
on each training arm. The fixed variant is the control and is identical across
arms.

**6. Why TSTR?**
Train-on-synthetic-test-on-real is the strictest utility test. It shows whether
the generator learned the feature-label relationship, not just how each column
looks, and it mirrors the real use case of sharing only synthetic data.

**7. Is the synthetic data private?**
Not automatically. GANs can memorise rare records, which is a particular risk with
only 16 minority training rows. We measure distance to the closest record against
a holdout baseline, exact copies, and a membership-inference attack. These are
empirical checks, not a guarantee. For a guarantee you need differential privacy
(e.g. DP-SGD / PATE-GAN) at some cost in fidelity.

**8. Why not augment the test set or use SMOTE on everything?**
Any change to the test set would leak synthetic artefacts into the evaluation and
inflate scores. The test set is real, held out first, and identical for every arm
within a seed. SMOTE-NC is applied to the training data only.

**9. Are the improvements statistically significant?**
We use 5 seeds with a paired t-test against arm A (see the significance table).
With n = 5, the Wilcoxon test cannot go below p = 0.0625, and we apply no
multiple-comparison correction, so borderline p-values are suggestive only.

**10. What are the main limitations?**
The data is tiny (303 records, about 16 training positives), which gives high
variance and makes fairness gaps noisy. The rules were written with knowledge of
this dataset's literature. The test prevalence differs from training on purpose.
The privacy tests are heuristics. Future work: larger datasets such as MIMIC with
proper approvals, differentially private generators, and Bayesian-network
comparisons with the CF model.
