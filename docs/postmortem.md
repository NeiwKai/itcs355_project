# Post-mortem — Unchecked Upstream Data Shift on Volatile Acidity

**What fired:**
Alert fired at 2026-10-10T16:12:51Z on metric `drift.psi.volatile_acidity` with PSI score `6.56710` (KS = `1.00000`), exceeding the operational threshold of `0.20`.
Email receieved at 2026-10-10T16:17:17Z, with manual trigger. (GitHub cron was set to every 6hrs)

**True cause:**
Broken upstream pipeline / artificial feature injection (+6.0 magnitude shift). The underlying relationship ($P(Y\vert{}X)$) did not change naturally; rather, the incoming data source ($P(X)$) delivered unphysical values far beyond realistic chemical bounds for wine.

**Retrain, roll back, or no action — and why:**
No action / Do NOT retrain. Retraining on corrupted feature distributions poisons model weights, permanently ruining inference accuracy on valid samples. I would roll back upstream ingestion to the last known clean batch; I would only consider retraining if an audit confirmed this distribution change reflected true, uncorrupted production wine batches.

**What this would have cost if unnoticed for a week:**
Assuming 5,000 daily prediction requests for wine quality classification, an uncalibrated +6.0 acidity shift causes near 100% incorrect low-quality predictions, leading to an estimated $17,500/week loss in misclassified rejected inventory and degraded buyer trust.

**How to prevent or detect it faster:**
Add an upstream schema contract test (`great_expectations` or `pydantic` range check) in the CI ingestion pipeline that validates `0.0 <= volatile_acidity <= 2.6` and immediately rejects batches with out-of-bounds values before they reach `data/current.csv`.
