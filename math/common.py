"""Shared constants for the math-pretraining Modal apps."""

from __future__ import annotations

import modal

HF_DATASET_ID = "nvidia/Nemotron-CC-Math-v1"

DATA_VOLUME_NAME = "nemotron-cc-math-v1"
DOLMINO_VOLUME_NAME = "dolma3-dolmino-mix-100B-1125"
TOKENIZED_VOLUME_NAME = "math-pretraining-tokenized"
UNTRAINED_VOLUME_NAME = "math-pretraining-untrained"
CHECKPOINT_VOLUME_NAME = "olmo-core-checkpoints-v2"
CACHE_VOLUME_NAME = "olmo-core-cache"

DATA_MOUNT = "/data"
DOLMINO_MOUNT = "/dolma3"
TOKENIZED_MOUNT = "/tokenized"
UNTRAINED_MOUNT = "/untrained"
CHECKPOINT_MOUNT = "/checkpoints"
CACHE_MOUNT = "/cache"

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
dolmino_volume = modal.Volume.from_name(DOLMINO_VOLUME_NAME, create_if_missing=True)
tokenized_volume = modal.Volume.from_name(TOKENIZED_VOLUME_NAME, create_if_missing=True)
untrained_volume = modal.Volume.from_name(UNTRAINED_VOLUME_NAME, create_if_missing=True)


def hf_image_base() -> modal.Image:
    """Pip-installs only. Callers extend then call .add_local_python_source('common') LAST."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "huggingface_hub>=0.26",
            "hf_transfer>=0.1.8",
            "pyarrow>=16",
            "tokenizers>=0.20",
            "numpy>=1.26",
            "tqdm>=4.66",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )


def hf_image() -> modal.Image:
    """Lightweight image for HF-based download. Includes common module."""
    return hf_image_base().add_local_python_source("common")
