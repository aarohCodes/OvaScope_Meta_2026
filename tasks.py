"""Cross-platform task runner (the Makefile delegates here; `make` is not required).

python tasks.py setup          create .venv, install Python + web dependencies
python tasks.py dev            API (inline worker) on :8000 + web on :3000
python tasks.py api | web | worker
python tasks.py test           pytest + web unit tests
python tasks.py e2e            Playwright end-to-end tests (starts API + web)
python tasks.py lint           ruff + mypy + eslint + tsc
python tasks.py data-audit     manifest, validation, shortcut audit
python tasks.py fetch-models   pinned DINOv2 + SAM 2.1 weights
python tasks.py train [CONFIG] / evaluate [CONFIG] / bundle [CONFIG] / report [CONFIG]
python tasks.py check-no-data  fail if images/data are staged for commit
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
NPM = "npm.cmd" if WIN else "npm"
WEB = ROOT / "apps" / "web"
DEFAULT_CONFIG = "configs/mmotu_v1.yaml"


def run(cmd: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if r.returncode != 0:
        sys.exit(r.returncode)


def ml(*args: str) -> None:
    run([PY, "-m", "ovascope_ml", *args])


def setup() -> None:
    if not VENV_PY.exists():
        run([sys.executable, "-m", "venv", ".venv"])
    py = str(VENV_PY)
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    # Hash-pinned universal lock (uv pip compile); CPU torch wheels come from the PyTorch index.
    run(
        [
            py,
            "-m",
            "pip",
            "install",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "-r",
            "requirements/lock.txt",
        ]
    )
    run([py, "-m", "pip", "install", "-e", ".", "--no-deps"])
    run([NPM, "ci"], cwd=WEB)


def api() -> None:
    run([PY, "-m", "uvicorn", "ovascope_api.main:get_app", "--factory", "--host", "127.0.0.1", "--port", "8000"])


def worker() -> None:
    run([PY, "-m", "ovascope_worker.main"], env={"OVASCOPE_INLINE_WORKER": "false"})


def web() -> None:
    run([NPM, "run", "dev"], cwd=WEB)


def dev() -> None:
    procs = [
        subprocess.Popen(
            [PY, "-m", "uvicorn", "ovascope_api.main:get_app", "--factory", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT,
        ),
        subprocess.Popen([NPM, "run", "dev"], cwd=WEB),
    ]
    print("API http://127.0.0.1:8000  |  Web http://localhost:3000  (Ctrl+C to stop)")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


def test() -> None:
    run([PY, "-m", "pytest", "-q"])
    run([NPM, "run", "test"], cwd=WEB)


def e2e() -> None:
    run([NPM, "run", "e2e"], cwd=WEB)


def lint() -> None:
    run([PY, "-m", "ruff", "check", "ml", "services", "tests", "scripts", "tasks.py"])
    run([PY, "-m", "mypy"])
    run([NPM, "run", "lint"], cwd=WEB)
    run([NPM, "run", "typecheck"], cwd=WEB)


def check_no_data() -> None:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    out += subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    bad = [
        f
        for f in set(out)
        if f.startswith(("data/", "artifacts/"))
        or (
            f.lower().endswith((".jpg", ".jpeg", ".png", ".dcm"))
            and not f.startswith(("tests/fixtures/geometric/", "apps/web/public/"))
        )
    ]
    if bad:
        print("Refusing: data or image files are tracked/staged:\n  " + "\n  ".join(sorted(bad)))
        sys.exit(1)
    print("ok: no dataset images or derived data tracked")


TASKS = {
    "setup": setup,
    "dev": dev,
    "api": api,
    "web": web,
    "worker": worker,
    "test": test,
    "e2e": e2e,
    "lint": lint,
    "check-no-data": check_no_data,
    "data-audit": lambda: ml("data-audit"),
    "verify-mirror": lambda: ml("verify-mirror"),
    "fetch-models": lambda: ml("fetch-models"),
    "provider-status": lambda: ml("provider-status"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    name, rest = sys.argv[1], sys.argv[2:]
    cfg = rest[0] if rest else DEFAULT_CONFIG
    if name in ("train", "evaluate", "report"):
        ml(name, "--config", cfg)
    elif name == "bundle":
        ml("export-bundle", "--config", cfg)
    elif name in TASKS:
        TASKS[name]()
    else:
        print(f"unknown task {name!r}\n{__doc__}")
        sys.exit(2)


if __name__ == "__main__":
    main()
