SHELL := /bin/bash
IMAGE ?= itcs355-proj
TAG := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
PLATFORM ?= linux/amd64
SEED := 20260518
DATA_PATH := data/raw/winequality-red.csv
MODEL_PATH := reports/model.joblib

.PHONY: help setup test
 

help:
	@grep -E "^[a-zA-Z_-]+:.*?## .*$$" $(MAKEFILE_LIST) | awk -F":.*?## " "{printf \"  %-20s %s\\n\", \$$1, \$$2}"

setup: ## Install dependencies and print environment status
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "environment ok"

data: ## Check that the data exist, if not tell user to download it
	@if [ ! -f $(DATA_PATH) ]; then \
		echo "Downloading dataset from kagglehub..."; \
		python -m scripts.get_dataset; \
		echo "Download finish!"; \
	fi
	@echo "✓ Data verified: '$(DATA_PATH)' exists."

test:
	pytest -q tests/

train:
	python -m src.train --seed $(SEED) --metrics-out reports/metrics.json

image: ## Build the training image for linux/amd64
	@echo "Building docker image..."
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) --load .

reproduce: data image 
	@echo "Running reproduction pipeline..."
	docker run --rm \
	  -v "$$PWD/data:/app/data:ro" \
	  -v "$$PWD/reports:/app/reports" \
	  -e MLFLOW_TRACKING_URI=sqlite:////app/reports/mlflow.db \
	  $(IMAGE):$(TAG) --seed $(SEED) --metrics-out /app/reports/metrics.json

export_model:
	@if [ ! -f reports/model.joblib ]; then \
		echo "Exporting model joblib..."; \
		python scripts/export_model.py --C 1.0 --epsilon 0.1 --out $(MODEL_PATH); \
	else \
		echo "✓ Model verified: '$(MODEL_PATH)' exists."; \
	fi

serve: export_model ## Run the inference service locally on :8080
	MODEL_PATH=reports/model.joblib MODEL_VERSION=local uvicorn service.app:app --port 8080

serve-image: export_model ## Build the serving image
	docker buildx build --platform $(PLATFORM) -f service/Dockerfile.serve -t $(IMAGE)-serve:$(TAG) --load .

serve-image-local: serve-image ## Host the serving image locally (test)
	@echo "Hosting service from serving image..."
	docker run --rm \
	  -p 8080:8080 \
	  -v "$$PWD/data:/app/data:ro" \
	  -v "$$PWD/reports:/app/reports" \
	  -e MLFLOW_TRACKING_URI=sqlite:///app/reports/mlflow.db \
	  $(IMAGE)-serve:$(TAG)

serve-image-push: serve-image ## Push image/context via cloud adapter 
	@echo "Pushing image/context via cloud adapter..."
	python -c "\
from src import config; \
from cloudlayer.factory import get_adapter; \
cfg = config.load(strict=False); \
adapter = get_adapter(cfg); \
uri = adapter.push_image('$(IMAGE)-serve:$(TAG)'); \
print(f'✓ Image/Context successfully pushed to: {uri}')"

inject-drift: ## Shift a feature's distribution on purpose
	python scripts/inject_drift.py --feature "volatile acidity" --mode shift --magnitude 0.3

drift: ## Score drift against the reference window
	python -m monitoring.drift --current data/current.csv

register: export_model ## Register the model to bucket
	python -c "\
	from src import config; \
	from cloudlayer.factory import get_adapter; \
	cfg = config.load(strict=False); \
	adapter = get_adapter(cfg); \
	model_uri = adapter.register_model(model_path='$(MODEL_PATH)',model_name=cfg.model_registry_name); \
	print(f'✓ Model successfully registered at: {model_uri}')"


deploy: ## Deploy model inference service to Render
	@echo "Deploying inference service to Render..."
	python -c "\
	from src import config; \
	from cloudlayer.factory import get_adapter; \
	cfg = config.load(strict=False); \
	adapter = get_adapter(cfg); \
	url = adapter.deploy(model_ref=cfg.model_registry_name, endpoint='itcs355-proj-service'); \
	print(f'✓ Deployed live at: {url}')"




