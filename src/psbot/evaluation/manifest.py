"""Immutable metadata attached to every model and tournament artifact."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from psbot.constants import SCHEMA_VERSION, SHOWDOWN_COMMIT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


class RunManifest(BaseModel):
    """Reproduction fields required by the project plan."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = SCHEMA_VERSION
    git_sha: str
    showdown_commit: str = SHOWDOWN_COMMIT
    lock_sha256: str
    config_sha256: str
    seed: int
    model_sha256: str | None = None
