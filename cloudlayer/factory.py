"""Adapter selection. The only place that maps CLOUD_PROVIDER to an implementation."""
from __future__ import annotations

from cloudlayer.base import CloudAdapter, LocalAdapter


def get_adapter(cfg) -> CloudAdapter:
    provider = (cfg.provider or "local").lower()
    if provider == "local":
        return LocalAdapter(cfg)
    if provider == "hf":
        from cloudlayer.hf import HuggingFaceAdapter
        return HuggingFaceAdapter(cfg)
    raise ValueError(f"Unknown CLOUD_PROVIDER={provider!r}. Use hf or local.")
