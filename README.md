# Wine Quality Prediction & MLOps Infrastructure (`itcs355_project`)

An end-to-end Machine Learning Operations (MLOps) pipeline for serving, monitoring, and validating wine quality predictions. Built as part of the **ITCS355 Machine Learning Operation and Deployment** capstone course.

### Member

* 6688101 Chavis Rittigun
* 6688132 Suppanat Timngarm

---

## 🏗 System Architecture

```
[ Git / PR ] ──> [ GitHub Actions CI ] ──> [ Render CD Deployment ]
                         │                        │
                         ▼                        ▼
                [ Data Contract Tests ]  [ FastAPI Model Endpoint ]
                                                  │
                                                  ▼
                                         [ Ingestion / Drift ]
                                                  │
                                                  ▼
[ Alert Check Watchdog ] <── [ Hugging Face Telemetry Stream ]

```

---

## 🛠 Tech Stack

* **Machine Learning & Data:** Python 3.11, PyTorch / Scikit-Learn, Pandas, NumPy
* **API & Deployment:** FastAPI, Uvicorn, Render (Free Tier Container Hosting)
* **Telemetry & Storage:** Hugging Face Datasets (`NeiwKai/itcs355-metrics`, `NeiwKai/itcs355-proj-serve`), 
    * Hugging Face Models (`NeiwKai/itcs355-proj-6688101-6688132-model`)
* **CI/CD & Monitoring:** GitHub Actions, Pytest, Gmail SMTP Alerting
* **Automation:** GNU Makefile

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Ensure Python 3.11+ is installed, then set up your local environment:

```bash
# Clone the repository
git clone https://github.com/NeiwKai/itcs355_project.git
cd itcs355_project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Quick reproduce
make reproduce

```

### 2. Set Environment Variables

Create a `xxx.env` file or export required keys in your terminal:

```bash
export HF_TOKEN="your_huggingface_write_token"
export HF_NAMESPACE="your_namespace_huggingface_repo"
export ALERT_SENDER_EMAIL="your_email@gmail.com"
export ALERT_SENDER_PASSWORD="your_google_app_password"
export ALERT_RECIPIENT_EMAIL="your_email@gmail.com"

```

For config.py, it does require `.env` file which can be found as `example.env.example`.

---

## ⚡ Makefile Operations

The pipeline is fully orchestrated via `make` targets:

| Command | Action |
| --- | --- |
| `make setup` | Set up the environment for the repository project. |
| `make data` | Fetch and download the dataset via kagglehub. |
| `make test` | For pytest executaion. |
| `make train` | Train the model with given seed. |
|` make image` | Build the training image for linux/amd64. |
| `make reproduce` | The one command to test on fresh machine, make sure to run `make setup` first. |
| `make export_model` | Export the model into model.joblib format with desire hyperparameters. |
| `make serve` | Serve the app service localy. |
| `make serve-image` | Build the serving service image for linux/amd64. |
| `make serve-image-local` | Host the serving image locally. |
| `make serve-image-push` | Push the serving image/context via cloud adapter. |
| `make inject-drift` | Deliberately injects artificial feature drift (+6.0 shift) into `volatile acidity`. |
| `make drift` | Computes PSI/KS feature drift on `data/current.csv` and logs telemetry to Hugging Face. |
| `make register` | Register the model to cloud bucket. |
| `make deploy` | Deploy the model inference service to cloud (Render). |
| `make smoke` | Sends a pre-warmup ping to Render and executes a live model smoke test. |
| `make loadtest` | Loadtest via k6. |
| `make loadtest-batch` | Compare single vs. batch prediction latency. |
| `make loadtest-payload` | Sweep payload size to measure serialization latency. |
| `make loadtest-coldstart` | Measure cold-start latency vs warm steady-state latency on cloud (Render). |
| `make gen-env` | Generate env.js for dashboard from active environment variables. |
| `make alert-test` | Executes the watchdog script to verify thresholds and dispatch alert emails. |

---

## 📊 Telemetry, Alerting & Deduplication

1. **Streaming Metrics:** Continuous feature drift metrics (Population Stability Index & Kolmogorov-Smirnov) are streamed to `telemetry.jsonl` hosted on Hugging Face.
2. **Watchdog Pipeline:** `scripts/check_telemetry_alerts.py` evaluates metric breaches, e.g. (`drift.psi.volatile_acidity > 0.20`).
3. **Automated Acknowledgment:** Once an alert email is dispatched, an `alert_acknowledged` record is committed directly back to Hugging Face to prevent email spam on subsequent runs.

---

## 🚨 Incident Response & Post-Mortem Workflow

When a metric breach occurs in production:

1. **Alert Dispatched:** Email alert is triggered via Gmail SMTP.
2. **Post-Mortem Analysis:** Evaluate whether the breach is **true concept drift** (requires retraining) or an **upstream data pipeline failure** (requires rollback, **not** retraining on corrupted data).
3. **CL Test Integration:** Lessons learned from post-mortems are fed back into `tests/test_data.py` as new automated contract tests to ensure bad data is caught in CI before reaching production.

---
