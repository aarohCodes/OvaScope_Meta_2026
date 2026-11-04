"""Repository hygiene: every source file is tracked, and no data or images are."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)  # noqa: S603
    return [line for line in out.stdout.splitlines() if line]


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout"
)


def test_no_python_source_is_ignored():
    # Regression: an unanchored `data/` rule once excluded ml/ovascope_ml/data/ from the repo.
    ignored = _git(
        "ls-files", "--others", "--ignored", "--exclude-standard", "--", "ml", "services", "tests", "scripts"
    )
    source = [f for f in ignored if f.endswith(".py") and "__pycache__" not in f]
    assert source == [], f"source files excluded by .gitignore: {source}"


def test_no_dataset_or_image_files_tracked():
    tracked = _git("ls-files")
    bad = [
        f
        for f in tracked
        if f.startswith(("data/", "artifacts/"))
        or (
            f.lower().endswith((".jpg", ".jpeg", ".png", ".dcm", ".npz", ".npy"))
            and not f.startswith("tests/fixtures/geometric/")
        )
    ]
    assert bad == []
