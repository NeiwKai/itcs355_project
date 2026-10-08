from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CloudAdapter(ABC):
    """Provider-neutral interface. The grader only ever calls these."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    # --- Lab 1 ---------------------------------------------------------------
    @abstractmethod
    def upload(self, local_path: str, key: str) -> str:
        """Upload a file under BLOB_URI. Returns the full URI of the stored object."""

    @abstractmethod
    def download(self, uri: str, local_path: str) -> None:
        """Fetch an object to a local path. Creates parent directories."""

    @abstractmethod
    def push_image(self, local_tag: str) -> str:
        """Push a locally built image to CONTAINER_REGISTRY. Returns the remote reference,
        which must be digest-pinned (repo@sha256:...), not tag-pinned."""

    def register_model(self, model_uri: str, name: str) -> str:
        raise NotImplementedError()

    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        raise NotImplementedError()

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError()

    def emit_metric(self, name: str, value: float, unit: str = "None") -> None:
        raise NotImplementedError()

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete every resource carrying these tags. Returns what was deleted.

        Deletion is asynchronous on all three providers — returning successfully does
        not mean the resource is gone. Re-check, and check the bill.
        """
        raise NotImplementedError()


class LocalAdapter(CloudAdapter):

    def upload(self, local_path: str, key: str) -> str:
        import shutil
        from pathlib import Path

        dest = Path(self.cfg.data_dir) / "_local_blob" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"local://{dest}"

    def download(self, uri: str, local_path: str) -> None:
        import shutil
        from pathlib import Path

        src = uri.removeprefix("local://")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def push_image(self, local_tag: str) -> str:
        raise NotImplementedError()
