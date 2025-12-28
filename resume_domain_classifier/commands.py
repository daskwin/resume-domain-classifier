from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def _load_cfg(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"
    overrides = overrides or []
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides)


def train(**kwargs: Any) -> None:
    overrides = [f"{k}={v}" for k, v in kwargs.items()]
    cfg = _load_cfg("train", overrides=overrides)
    from resume_domain_classifier.training.train import train_from_config

    train_from_config(cfg)


def infer(text: str, **kwargs: Any) -> None:
    overrides = [f"{k}={v}" for k, v in kwargs.items()]
    cfg = _load_cfg("infer", overrides=overrides)
    from resume_domain_classifier.inference.predict import predict_text

    result = predict_text(cfg, text=text)
    print(result["predicted_label"])
    print(result["top"])


def main() -> None:
    fire.Fire({"train": train, "infer": infer})
