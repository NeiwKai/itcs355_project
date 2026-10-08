from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

# The eight capability slots every lab depends on. scripts/cloud_check.py resolves each.
CAPABILITY_SLOTS = (
    "CLOUD_PROVIDER",
    "TOKEN",
    "DEPLOY_HOOK",
    "NAMESPACE",
    "PROJECT_ID",
    "MLFLOW_TRACKING_URI",
    "MODEL_REGISTRY_NAME",
)


def _load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


@dataclass(frozen=True)
class Config:
    provider: str
    token: str
    deploy_hook: str
    namespace: str
    mlflow_tracking_uri: str
    model_registry_name: str
    data_dir: Path = field(default=REPO_ROOT / "data")
    reports_dir: Path = field(default=REPO_ROOT / "reports")

    @property
    def raw_path(self) -> Path:
        return self.data_dir / "raw" / "winequality-red.csv"


def load(strict: bool = True) -> Config:
    missing = [s for s in CAPABILITY_SLOTS if not os.environ.get(s)]
    if missing and strict:
        raise RuntimeError(
            "Missing capability slots: " + ", ".join(missing)
            + "\nCopy .env.example to .env, fill it in."
        )
    get = os.environ.get
    return Config(
        provider=get("CLOUD_PROVIDER", "local"),
        token=get("TOKEN", ""),
        deploy_hook=get("DEPLOY_HOOK", ""),
        namespace=get("NAME_SPACE", "NeiwKai/itcs355-project"),
        mlflow_tracking_uri=get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        model_registry_name=get("MODEL_REGISTRY_NAME", "itcs355-proj"),
    )
