"""Shared JSON file-writing utilities for typed result records."""

import json
from collections.abc import Mapping
from pathlib import Path


def write_json_record(
    payload: Mapping[str, object],
    destination: Path,
) -> Path:
    """Write a mapping as indented UTF-8 JSON and return its path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
