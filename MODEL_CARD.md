---
library_name: scikit-learn
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - wine-quality
  - scikit-learn
  - svr
# license: [More Information Needed]
---

<!-- Metrics come from the run of 7 Oct 2026 (reports/metrics.json). "[More Information Needed]" marks values we do not have yet. -->

# Model Card for itcs355-proj-6688101-6688132-model

An SVR regression pipeline that estimates the sensory quality score (3 to 8) of a red wine from 11 physicochemical lab readings. It is served through a FastAPI endpoint as a quick screening aid for quality-assurance staff. It is a course project (ITCS355) and is advisory only.

## Model Details

### Model Description

The model is a scikit-learn pipeline: `RobustScaler`, then `SVR` with an RBF kernel (C=1.0, epsilon=0.1). It predicts a continuous score, which the service rounds to the nearest integer and clips to the range 3 to 8. It was trained from scratch on the UCI Red Wine Quality data (1,599 rows) with the fixed seed 20260518.

- **Developed by:** Chavis Rittigun and Suppanat Timngarm (ITCS355 capstone, Faculty of ICT, Mahidol University)
- **Funded by :** Not applicable (course project)
- **Shared by :** Hugging Face namespace `NeiwKai` (private repository)
- **Model type:** Tabular regression, `RobustScaler` + `SVR` (RBF)
- **Language(s) (NLP):** Not applicable (11 numeric inputs)
- **License:** Not specified
- **Finetuned from model [optional]:** None, trained from scratch

### Model Sources [optional]

- **Repository:** https://github.com/pasdptt/public_teaching_mlaiops (course GitHub repository)
- **Paper [optional]:** Dataset paper: Cortez et al. (2009), see Citation
- **Demo [optional]:** Render service, default address in the repo `https://itcs355-project.onrender.com` (free tier, sleeps when idle)

## Uses

### Direct Use

Estimate the sensory quality rating of a red wine batch from 11 lab readings, as a quick screen through the online API. Intended users are winery quality-assurance technicians doing ad-hoc checks. The output is advisory and does not replace a sensory panel or certification.

### Downstream Use [optional]

None planned.

### Out-of-Scope Use

White wine, other regions or grape types, certification, and any decision with legal or safety consequences. 
## Bias, Risks, and Limitations

- **Small data:** the validation and test sets have 320 rows each. Validation R² (0.340) and test R² (0.451) differ noticeably for the same model, so small differences in any metric are not meaningful.
- **Exactness is modest:** about 62.5% of test wines get exactly the right score, and about 2% are off by two points or more.
- **Imbalanced scores:** 82% of wines are rated 5 or 6. Scores 3 and 8 have only 10 and 18 rows, so there is little data to learn the extremes from.
- **Regression to the mean:** an SVR rarely predicts the rare extreme scores.  (check the spread of predictions).
- **Duplicates:** 240 of the 1,599 rows (15%) are exact duplicates and were not removed. A random split can put the same wine in train and test, which can flatter the metrics.
- **One wine type:** red wine from this dataset only. 
- **Subjective labels:** the target is a human rating, not a chemical ground truth. 
- **Only 11 features:** nothing about grape, vintage or storage.
- **Over- or under-rating a batch:** 
- **No production feedback:** the dataset is static, so real-world performance is unmeasured.

### Recommendations

Treat the score as a screening hint and confirm borderline or high-stakes batches with a sensory panel. Report the uncertainty (a typical error of about half a point) alongside any score. Re-validate on your own winery's data before relying on it. Users should be made aware of the risks and limitations above.

## How to Get Started with the Model

Call the live service. The bearer token is required only when the service has `API_SECRET_KEY` set. Field names contain spaces, and unknown or out-of-range fields return 422.

```bash
curl -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -d '{"fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
       "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
       "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
       "sulphates": 0.56, "alcohol": 9.4}'
# reply fields: quality_score (int 3-8), raw_score (float), model_version
```

Or load the model file directly:

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="<namespace>/<model-repo>", filename="model.joblib", token="<HF token>")
model = joblib.load(path)

row = pd.DataFrame([{
    "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}])
raw = float(model.predict(row)[0])
score = int(min(8, max(3, round(raw))))   # same rounding and clipping as the service
```

Allowed input ranges (enforced by the service):

| Field | Allowed range | Field | Allowed range |
|---|---|---|---|
| fixed acidity | 4.50 - 16.00 | free sulfur dioxide | 0 - 75 |
| volatile acidity | 0.10 - 1.60 | total sulfur dioxide | 5 - 290 |
| citric acid | 0.00 - 1.00 | density | 0.990 - 1.005 |
| residual sugar | 0.50 - 15.50 | pH | 2.70 - 4.05 |
| chlorides | 0.010 - 0.615 | sulphates | 0.30 - 2.00 |
| alcohol | 8.00 - 15.00 | | |

## Training Details

### Training Data

UCI Red Wine Quality (Cortez et al., 2009), downloaded through the Kaggle mirror `uciml/red-wine-quality-cortez-et-al-2009` with `kagglehub`. 

- **Size:** 1,599 rows, 11 features. Target: `quality`, an integer score.
- **Class counts:** score 3: 10, 4: 53, 5: 681, 6: 638, 7: 199, 8: 18.
- **Split:** random, stratified on `quality`, seed 20260518, about 60/20/20: train 959, validation 320, test 320 rows (sizes inferred from the accuracy fractions; confirm with the MLflow tags `n_train_rows`, `n_val_rows`, `n_test_rows`).
- **Versions:** data fingerprint `d6a0d9bd24806944` (SHA-256 prefix of the CSV); DVC directory hash `8c4ae7c2f53adc246c2d64dc8e264738` (`data/raw.dvc`).

### Training Procedure

#### Preprocessing [optional]

`RobustScaler` is fitted inside the pipeline on the training rows only, and the same pipeline is used at inference. No rows were removed, including the 240 duplicates.

#### Training Hyperparameters

- **Training regime:** Not applicable (classical SVR on CPU, no mixed precision). Kernel `rbf`, C=1.0, epsilon=0.1, seed 20260518. Note: `src/train.py` hard-codes C and epsilon (line 80), so its `--C` and `--epsilon` flags have no effect. `scripts/export_model.py` does use them.

#### Speeds, Sizes, Times [optional]
Measured serving latency and cold-start time are also pending the load test. The target is p95 under 300 ms.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The 320-row test split described above, never used for fitting.

#### Factors

Results are not disaggregated. The data covers a single wine type, and no subgroup analysis was done.

#### Metrics

- **MAE:** average error in quality points.
- **RMSE:** like MAE but penalises large errors.
- **R²:** share of variance explained.
- **Exact accuracy:** the rounded prediction equals the true score. The code names this `*_exact_auc`, but it is accuracy.
- **Within-1 accuracy:** the rounded prediction is within one point. The code names this `*_within_1_auc`, also accuracy.

Quality labels are ordinal, so error in points and "within one point" matter more than exact accuracy alone.

### Results

Headline seed 20260518 (the pinned seed used by `make reproduce`):

| Metric | Validation | Test |
|---|---|---|
| MAE | 0.495 | **0.444** |
| RMSE | 0.657 | **0.595** |
| R² | 0.340 | **0.451** |
| Exact accuracy | 0.5875 | **0.625** |
| Within-1 accuracy | 0.966 | **0.978** |

Repeating the same seed gave identical results to full floating-point precision across 2 Docker runs on the same machine. Cross-machine variation was not measured.

Changing the seed changes the split, so this is a different quantity. Test results for seeds 1-5:

| Test metric | Headline seed | Mean, seeds 1-5 | Range, seeds 1-5 |
|---|---|---|---|
| MAE | 0.444 | 0.483 | 0.444 - 0.515 |
| RMSE | 0.595 | 0.646 | 0.607 - 0.681 |
| R² | 0.451 | 0.353 | 0.282 - 0.429 |
| Exact accuracy | 0.625 | 0.611 | 0.581 - 0.662 |
| Within-1 accuracy | 0.978 | 0.966 | 0.956 - 0.981 |

#### Summary

The model is right to the exact score about six times in ten and within one point about 97% of the time. The headline seed is at the favourable end (its test RMSE and R² are the best of all six runs). On a different split, expect roughly R² 0.28 - 0.45 and MAE 0.44 - 0.52.

## Model Examination [optional]

No interpretability work was done. CI runs behaviour tests on the model (`tests/test_model_behaviour.py`): predictions stay within 3 to 8, the model is not constant, high-alcohol low-acidity wines score well, and quality decreases as volatile acidity rises.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** CPU only (team laptops, GitHub Actions runners)
- **Hours used:** not measured; the model trains on 959 rows
- **Cloud Provider:** GitHub Actions (retraining in CI and CD), Hugging Face Hub (storage), Render (serving)
- **Compute Region:** Not specified
- **Carbon Emitted:** Not measured

## Technical Specifications [optional]

### Model Architecture and Objective

Epsilon-insensitive support vector regression with an RBF kernel on robust-scaled inputs. The objective is the continuous quality score. Post-processing rounds to the nearest integer and clips to 3 to 8.

### Compute Infrastructure

Training runs in Docker (`make reproduce`) and again on GitHub Actions runners during CD. Serving runs on Render's free tier. The model file is stored in a private Hugging Face model repository, and the service downloads it at startup.

#### Hardware

CPU only.(Render instance size is not recorded in the repo).

#### Software

Python 3.11, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, joblib 1.4.2, MLflow 3.16.0, FastAPI 0.141.1, pydantic 2.13.5. Docker images use a base pinned by digest (`python@sha256:a2bc8c35...`), hash-locked requirements and a non-root user. Training commit `3acb105bfbe03ef088ce55b3b17f9dee12b14264` (the MLflow `git_commit` tag reads `unknown` because the training container has no git).

## Citation [optional]

The data comes from this paper:

**BibTeX:**

```bibtex
@article{cortez2009wine,
  title   = {Modeling wine preferences by data mining from physicochemical properties},
  author  = {Cortez, Paulo and Cerdeira, Ant{\'o}nio and Almeida, Fernando and Matos, Telmo and Reis, Jos{\'e}},
  journal = {Decision Support Systems},
  volume  = {47},
  number  = {4},
  pages   = {547--553},
  year    = {2009}
}
```

**APA:**

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547-553.

## Glossary 

- **MAE / RMSE / R²:** error and fit measures defined under Metrics.
- **Exact / within-1 accuracy:** share of wines whose rounded prediction equals the true score, or is within one point of it.
- **PSI:** Population Stability Index, a measure of how far an input's distribution has moved from the training data. Our alert threshold is 0.20.
- **KS:** Kolmogorov-Smirnov statistic, the largest gap between two cumulative distributions.
- **p95:** the response time that 95% of requests beat.
- **SLO:** service level objective, a target such as "p95 under 300 ms".
- **Cold start:** the slow first request after the free-tier service has been idle.

## More Information 

- **Service:** FastAPI with `/predict`, `/predict/batch` (max 100 rows), `/health` and `/ready`. The response has `quality_score`, `raw_score` and `model_version`. Missing, extra or out-of-range input returns 422 and the service stays up.
- **Delivery:** GitHub Actions CI (secret scan, lint, tests, image builds, integration test) then CD (retrain, register in a private Hugging Face model repository, deploy to Render through a deploy hook, smoke test).
- **Targets (`monitoring/slo.yaml`):** 99.0% availability over 30 days, p95 latency under 300 ms over 7 days (cold starts excluded), model no older than 30 days. for measured latency, throughput and cold-start time.
- **Monitoring and drift:** `monitoring/drift.py` scores PSI and KS per feature. PSI is written to a private telemetry log on Hugging Face. A scheduled GitHub Action emails an alert when volatile acidity PSI exceeds 0.20. The dashboard is `monitoring/dashboard.html`.
- **Designed failure:** `scripts/inject_drift.py` shifts volatile acidity to simulate a faulty sensor.  for the post-mortem and the CI test it produced. Do not retrain on drifted data before checking whether the upstream sensor is broken.
- **Retraining and rollback:** CD retrains on every push to main. for the rollback procedure.
- **Cost:** hosting is currently on free tiers. [ for cost per 1,000 predictions (the planned method is hourly price ÷ (requests per second × utilisation × 3600) × 1000, at 5%, 25% and 80% utilisation, with throughput from the load test).
- **Format:** sections follow the Hugging Face model card template, which builds on Mitchell et al., "Model Cards for Model Reporting" (2019).

## Model Card Authors 

Chavis Rittigun and Suppanat Timngarm

## Model Card Contact

Chavis.rit@student.mahidol.ac.th 
