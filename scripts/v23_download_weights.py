from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

from v23_common import ROOT, WEIGHTS, dataset_manifest, runtime_env, sha256, write_json


def capture_weight(candidate: str, dest: Path) -> None:
    src = ROOT / candidate
    if not src.exists():
        src = Path(candidate)
    if not src.exists():
        raise FileNotFoundError(f"Ultralytics did not materialize expected weight: {candidate}")
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)


def main() -> None:
    from ultralytics import YOLO, YOLOE

    yolo26 = os.getenv("YOLO26_WEIGHT", "yolo26n.pt")
    yoloe26 = os.getenv("YOLOE26_WEIGHT", "yoloe-26n-seg.pt")

    model = YOLO(yolo26)
    del model
    gc.collect()
    capture_weight(yolo26, WEIGHTS / Path(yolo26).name)

    model = YOLOE(yoloe26)
    del model
    gc.collect()
    capture_weight(yoloe26, WEIGHTS / Path(yoloe26).name)

    manifest = {
        "evidence_type": "online-checkpoint-acquisition",
        "performance_claim": False,
        "runtime": runtime_env(),
        "dataset_manifest": dataset_manifest(),
        "weights": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(WEIGHTS.glob("*.pt"))
        },
    }
    write_json(ROOT / "reports/v2_3_online_checkpoint_manifest.json", manifest)
    print(manifest["weights"])


if __name__ == "__main__":
    main()
