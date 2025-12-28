from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from resume_domain_classifier.data.datasets import load_raw_csv
from resume_domain_classifier.data.download_data import dvc_pull
from resume_domain_classifier.models.lightning_module import TfidfLinearLightning
from resume_domain_classifier.preprocessing.text_cleaning import clean_text
from resume_domain_classifier.utils.logging import get_git_commit_id, setup_logging

LOGGER = logging.getLogger(__name__)


class NumpyDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.int64)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.features[idx]), torch.tensor(
            self.targets[idx], dtype=torch.long
        )


@dataclass(frozen=True)
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    input_dim: int
    num_classes: int
    labels: list[str]
    vectorizer: TfidfVectorizer


def _prepare(cfg: DictConfig) -> PreparedData:
    repo_root = Path(__file__).resolve().parents[2]

    if (repo_root / ".dvc").exists():
        LOGGER.info("DVC detected, trying to pull data...")
        dvc_pull(repo_root)
    else:
        LOGGER.info("DVC not initialized yet; using local files if present.")

    raw_path = repo_root / cfg.data.raw_path
    data = load_raw_csv(
        raw_path,
        text_col=cfg.data.text_col,
        label_col=cfg.data.label_col,
        sep=cfg.data.get("sep", ","),
        encoding=cfg.data.get("encoding", "utf-8"),
    )

    cleaned = [
        clean_text(
            t,
            lowercase=cfg.preprocessing.lowercase,
            strip=cfg.preprocessing.strip,
            remove_urls=cfg.preprocessing.remove_urls,
            remove_emails=cfg.preprocessing.remove_emails,
            remove_numbers=cfg.preprocessing.remove_numbers,
            keep_only_basic_punct=cfg.preprocessing.keep_only_basic_punct,
            max_length=cfg.preprocessing.max_length,
        )
        for t in data.texts
    ]

    labels = sorted(set(data.labels))
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    targets = np.array([label_to_id[label] for label in data.labels], dtype=np.int64)

    x_train_text, x_tmp_text, y_train, y_tmp = train_test_split(
        cleaned,
        targets,
        test_size=0.30,
        random_state=int(cfg.seed),
        stratify=targets,
    )
    x_val_text, x_test_text, y_val, y_test = train_test_split(
        x_tmp_text,
        y_tmp,
        test_size=0.50,
        random_state=int(cfg.seed),
        stratify=y_tmp,
    )

    vectorizer = TfidfVectorizer(
        max_features=int(cfg.model.tfidf.max_features),
        ngram_range=tuple(cfg.model.tfidf.ngram_range),
        min_df=int(cfg.model.tfidf.min_df),
        max_df=float(cfg.model.tfidf.max_df),
    )
    x_train = vectorizer.fit_transform(x_train_text).toarray()
    x_val = vectorizer.transform(x_val_text).toarray()
    x_test = vectorizer.transform(x_test_text).toarray()

    return PreparedData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        input_dim=int(x_train.shape[1]),
        num_classes=int(len(labels)),
        labels=labels,
        vectorizer=vectorizer,
    )


def _export_artifacts(cfg: DictConfig, prepared: PreparedData, model: TfidfLinearLightning) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    export_dir = repo_root / cfg.model.export.dir
    export_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(prepared.vectorizer, export_dir / "vectorizer.joblib")
    torch.save(model.state_dict(), export_dir / "model.pt")
    (export_dir / "labels.json").write_text(
        json.dumps(prepared.labels, ensure_ascii=False, indent=2)
    )

    meta = {
        "input_dim": prepared.input_dim,
        "num_classes": prepared.num_classes,
        "labels_count": len(prepared.labels),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (export_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    LOGGER.info("Exported artifacts to: %s", export_dir)
    return export_dir


def _eval_on_test(
    prepared: PreparedData, model: TfidfLinearLightning
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(prepared.x_test.astype(np.float32)))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    acc = float(accuracy_score(prepared.y_test, preds))
    f1m = float(f1_score(prepared.y_test, preds, average="macro"))
    avg_conf = float(np.max(probs, axis=1).mean())
    return acc, f1m, avg_conf


def train_from_config(cfg: DictConfig) -> None:
    setup_logging()
    seed_everything(int(cfg.seed), workers=True)

    prepared = _prepare(cfg)

    train_loader = DataLoader(
        NumpyDataset(prepared.x_train, prepared.y_train),
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
    )
    val_loader = DataLoader(
        NumpyDataset(prepared.x_val, prepared.y_val),
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
    )

    commit_id = get_git_commit_id()
    run_name = cfg.logging.run_name or commit_id or "local-run"

    mlflow_logger = MLFlowLogger(
        tracking_uri=str(cfg.logging.tracking_uri),
        experiment_name=str(cfg.logging.experiment_name),
        run_name=str(run_name),
    )

    model = TfidfLinearLightning(
        input_dim=prepared.input_dim,
        num_classes=prepared.num_classes,
        lr=float(cfg.model.optimizer.lr),
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=int(cfg.train.max_epochs),
        logger=mlflow_logger,
        callbacks=[lr_monitor],
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    mlflow_logger.log_hyperparams(
        {
            "seed": int(cfg.seed),
            "text_col": str(cfg.data.text_col),
            "label_col": str(cfg.data.label_col),
            "tfidf_max_features": int(cfg.model.tfidf.max_features),
            "tfidf_ngram_range": list(cfg.model.tfidf.ngram_range),
            "tfidf_min_df": int(cfg.model.tfidf.min_df),
            "tfidf_max_df": float(cfg.model.tfidf.max_df),
            "lr": float(cfg.model.optimizer.lr),
            "batch_size": int(cfg.train.batch_size),
            "max_epochs": int(cfg.train.max_epochs),
            "git_commit_id": commit_id or "unknown",
        }
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_acc, test_f1m, test_avg_conf = _eval_on_test(prepared, model)
    mlflow_logger.log_metrics(
        {"test_acc": test_acc, "test_f1_macro": test_f1m, "test_avg_conf": test_avg_conf}
    )

    _export_artifacts(cfg, prepared, model)
