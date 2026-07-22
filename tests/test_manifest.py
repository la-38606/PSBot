import hashlib
from pathlib import Path

from psbot.evaluation.manifest import sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"psbot")
    assert sha256_file(artifact) == hashlib.sha256(b"psbot").hexdigest()
