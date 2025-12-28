from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TextClassificationData:
    texts: list[str]
    labels: list[str]


def load_raw_csv(
    path: Path,
    text_col: str,
    label_col: str,
    sep: str = ",",
    encoding: str = "utf-8",
) -> TextClassificationData:
    dataframe = pd.read_csv(path, sep=sep, encoding=encoding)
    if text_col not in dataframe.columns:
        raise ValueError(f"Text column '{text_col}' not found. Columns: {list(dataframe.columns)}")
    if label_col not in dataframe.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. Columns: {list(dataframe.columns)}"
        )

    texts = dataframe[text_col].astype(str).tolist()
    labels = dataframe[label_col].astype(str).tolist()
    return TextClassificationData(texts=texts, labels=labels)
