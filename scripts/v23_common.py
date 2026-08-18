from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
WEIGHTS = ROOT / "weights"
REPORTS.mkdir(parents=True, exist_ok=True)
WEIGHTS.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "steel lattice transmission tower structure",
    "crossarm of an electricity transmission tower",
    "insulator string on an electricity transmission tower",
    "earth wire peak at the top of an electricity transmission tower",
]
DISPLAY = ["steelwork", "crossarm", "insulator", "earthwire peak"]
SHOWCASE = ROOT / "data/images/test/POS_2326530.jpg"
DATA_YAML = ROOT / "data/data.yaml"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted((ROOT / "data").rglob("*")):
        if p.is_file():
            out[str(p.relative_to(ROOT))] = sha256(p)
    return out


def git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "not-a-git-checkout"


def runtime_env() -> dict[str, Any]:
    import torch
    try:
        import ultralytics
        uv = ultralytics.__version__
    except Exception:
        uv = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": uv,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "github_sha": git_sha(),
        "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", "local"),
    }


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
