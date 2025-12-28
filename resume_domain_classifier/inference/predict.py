from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from omegaconf import DictConfig

from resume_domain_classifier.models.lightning_module import TfidfLinearLightning
from resume_domain_classifier.preprocessing.text_cleaning import clean_text


def predict_text(cfg: DictConfig, *, text: str) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    artifact_dir = repo_root / str(cfg.artifact_dir)

    labels = json.loads((artifact_dir / "labels.json").read_text())
    vectorizer = joblib.load(artifact_dir / "vectorizer.joblib")

    meta = json.loads((artifact_dir / "meta.json").read_text())
    input_dim = int(meta["input_dim"])
    num_classes = int(meta["num_classes"])

    cleaned = clean_text(
        text,
        lowercase=cfg.preprocessing.lowercase,
        strip=cfg.preprocessing.strip,
        remove_urls=cfg.preprocessing.remove_urls,
        remove_emails=cfg.preprocessing.remove_emails,
        remove_numbers=cfg.preprocessing.remove_numbers,
        keep_only_basic_punct=cfg.preprocessing.keep_only_basic_punct,
        max_length=cfg.preprocessing.max_length,
    )

    features = vectorizer.transform([cleaned]).toarray().astype(np.float32)
    features_t = torch.from_numpy(features)

    model = TfidfLinearLightning(input_dim=input_dim, num_classes=num_classes, lr=1e-3)
    state = torch.load(artifact_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(features_t)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_id = int(probs.argmax())
    pred_label = labels[pred_id]

    top_k = int(cfg.top_k)
    top_idx = probs.argsort()[::-1][:top_k]
    top = [{"label": labels[int(i)], "prob": float(probs[int(i)])} for i in top_idx]

    return {
        "predicted_label": pred_label,
        "top": top,
        "probs": {labels[i]: float(p) for i, p in enumerate(probs)},
    }
