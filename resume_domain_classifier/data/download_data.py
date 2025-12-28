from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def dvc_pull(repo_root: Path) -> None:
    try:
        from dvc.repo import Repo
    except Exception as exc:
        raise RuntimeError("DVC is not installed in the current environment.") from exc

    try:
        repo = Repo(str(repo_root))
        repo.pull()
        LOGGER.info("DVC pull completed.")
    except Exception as exc:
        raise RuntimeError(
            "Failed to run `dvc pull`. Initialize DVC and configure a remote, "
            "or place dataset locally under data/raw/ according to configs."
        ) from exc
