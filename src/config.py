"""
src/config.py — Load config.yaml into a typed dataclass.

Usage anywhere in the project:
    from src.config import CFG
    print(CFG.model.hidden_size)   # → 64
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ── Root of the project (two levels up from this file) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml() -> dict:
    """Load config.yaml from the project root."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Nested dataclasses mirror the YAML structure ────────────────────────────


@dataclass
class DataConfig:
    raw_path: str
    artifacts_dir: str


@dataclass
class PreprocessingConfig:
    sequence_length: int
    features: list[str]
    target: str
    train_ratio: float
    val_ratio: float


@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    seed: int


@dataclass
class ArtifactsConfig:
    model_path: str
    scaler_path: str
    metrics_path: str


@dataclass
class Config:
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig

    # Convenience: return absolute paths regardless of cwd
    def abs(self, relative_path: str) -> Path:
        return PROJECT_ROOT / relative_path


def _build_config() -> Config:
    raw = _load_yaml()
    return Config(
        data=DataConfig(**raw["data"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        artifacts=ArtifactsConfig(**raw["artifacts"]),
    )


# ── Single importable instance ───────────────────────────────────────────────
CFG = _build_config()
