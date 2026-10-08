from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from cloudlayer.base import CloudAdapter
from huggingface_hub import (
    create_repo,
    hf_hub_download,
    upload_file,
    upload_folder,
)


class HuggingFaceAdapter(CloudAdapter):
    """CloudAdapter implementation mapping MLOps capabilities to Hugging Face Hub & Spaces.

    Environment Variables Required:
        HF_TOKEN: Hugging Face User Access Token with write permissions.
        HF_NAMESPACE: User or Organization namespace (e.g., 'NeiwKai').
    """
    def _get_repo_id(self, name: str) -> str:
        """Ensures the repository name is formatted with a namespace (e.g., 'NeiwKai/repo-name')."""
        if "/" in name:
            return name
            
        namespace = (
            getattr(self, "namespace", None) 
            or os.environ.get("NAMESPACE") 
            or "NeiwKai"
        )
        return f"{namespace}/{name}"

    # --- Storage / Artifact Capabilities ---
    def upload(self, local_path: str, key: str) -> str:
        pass

    def download(self, uri: str, local_path: str) -> None:
        """Downloads a model or file artifact from Hugging Face Hub to a local path.

        Args:
            uri: Hugging Face model/dataset URI (e.g. 'hf://models/NeiwKai/wine-quality-model/model.joblib'
                 or repo ID 'NeiwKai/wine-quality-model').
            local_path: Local target file path to save the downloaded file (e.g. 'reports/model.joblib').
        """

        out_path = Path(local_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse URI format
        clean_uri = uri.replace("hf://", "")
        parts = clean_uri.split("/")

        repo_type = "model"
        if parts[0] in ["models", "datasets"]:
            repo_type = parts[0][:-1]  # 'model' or 'dataset'
            parts = parts[1:]

        # Handle 'namespace/repo/filename' vs 'namespace/repo'
        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:]) if len(parts) > 2 else "model.joblib"
        else:
            repo_id = self._get_repo_id(parts[0])
            filename = "model.joblib"

        token = self.cfg.token or os.environ.get("TOKEN")

        # Download from Hugging Face Hub using the adapter's token
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token=token  # Required for private repositories
        )

        print(f"✓ Downloaded '{repo_id}' to HF cache: '{cached_model_path}'")
        return cached_model_path


    def push_image(self, local_tag: str) -> str:
        """Pushes the application context (Dockerfile, service/, src/) 

        to a free Hugging Face Dataset repository as an artifact backup.
        """
        space_name = local_tag.split(":")[0].replace("_", "-")
        repo_id = self._get_repo_id(space_name)
        token = self.cfg.token or os.environ.get("TOKEN")

        # 1. Store build context in a free HF Dataset repo (bypasses HF 402 error)
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            private=True,
            exist_ok=True,
        )

        upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            allow_patterns=["service/*", "src/*", "reports/*", "Dockerfile*", "requirements*"],
        )

        return f"https://huggingface.co/datasets/{repo_id}"

    def register_model(self, model_path: str, model_name: str) -> str:
        """Registers and uploads a trained model artifact to HF Model Hub."""
        repo_id = self._get_repo_id(model_name)
        token = self.cfg.token or os.environ.get("TOKEN")
        create_repo(
            repo_id=repo_id, 
            repo_type="model", 
            token=token, 
            private=True,
            exist_ok=True
        )

        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.joblib",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        return f"https://huggingface.co/models/{repo_id}"

    def deploy(self, model_ref: str, endpoint: str, instance: str = "free") -> str:
        """Triggers live container hosting on Render via Deploy Hook."""
        import requests

        render_hook_url = self.cfg.deploy_hook or os.environ.get("DEPLOY_HOOK")
        if not render_hook_url:
            raise RuntimeError("RENDER_DEPLOY_HOOK environment variable is not set.")

        # 2. Trigger Render to pull code and host the live container
        res = requests.post(render_hook_url, timeout=30)
        res.raise_for_status()

        return os.environ.get("RENDER_SERVICE_URL", "https://itcs355-project.onrender.com")

    def emit_metric(self, metric_name: str, value: float) -> None:
        """Emits tracking metrics to Hugging Face Space runtime logs/telemetry."""
        print(f"[HF Telemetry] {metric_name}: {value}")


