"""Frozen crop comparison for the existing supervised material head, on Roihu only."""
import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from material_head_common import decide

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def extent(box, width, height, padding):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    return [max(0, int(x0 - dx)), max(0, int(y0 - dy)),
            min(width, int(x1 + dx + 0.999999)), min(height, int(y1 + dy + 0.999999))]


def main(config_path):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Requires a Roihu gputest allocation; no local fallback")
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config_path = ROOT / config_path
    cfg = json.loads(config_path.read_text())
    image_path, freeze_path = ROOT / cfg["image"], ROOT / cfg["freeze"]
    head_path, encoder_path = ROOT / cfg["head"], ROOT / cfg["encoder"]
    for path, expected in [(image_path, cfg["image_sha256"]), (freeze_path, cfg["freeze_sha256"]),
                           (head_path, cfg["head_sha256"]), (encoder_path / "model.safetensors", cfg["encoder_sha256"])]:
        if sha(path) != expected:
            raise ValueError(f"Hash mismatch: {path}")
    freeze = json.loads(freeze_path.read_text())
    if freeze["references"][0]["xyxy"] != cfg["crops"][0]["xyxy"]:
        raise ValueError("Complete assembly does not match frozen reference")
    out = ROOT / cfg["run"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "code").mkdir()
    for path in [config_path, Path(__file__), ROOT / "scripts/material_head_common.py"]:
        shutil.copy2(path, out / "code" / path.name)
    result = {
        "status": "ENCODING", "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(config_path), "image_sha256": sha(image_path),
        "freeze_sha256": sha(freeze_path), "head_sha256": sha(head_path),
        "encoder_sha256": sha(encoder_path / "model.safetensors"), "classes": cfg["classes"],
        "encoder_gradient_steps": 0, "head_gradient_steps": 0, "comparisons": [],
        "runtime": {"job_id": os.environ["SLURM_JOB_ID"], "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__, "transformers": transformers.__version__},
        "claim_boundary": cfg["claim_boundary"]
    }
    write(out / "results.json", result)
    start = time.perf_counter()
    try:
        image = Image.open(image_path).convert("RGB")
        processor = AutoProcessor.from_pretrained(encoder_path, local_files_only=True, trust_remote_code=False)
        encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True, trust_remote_code=False,
                                             use_safetensors=True).to("cuda").eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        inputs, index = [], []
        for crop in cfg["crops"]:
            for view, padding in [("tight", 0), ("context", cfg["context_padding"])]:
                box = extent(crop["xyxy"], image.width, image.height, padding)
                inputs.append(image.crop(box))
                index.append((crop["id"], view, box))
        batch = processor(images=inputs, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            features = encoder.get_image_features(**batch)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = F.normalize(features.float(), dim=-1).cpu().numpy()
        np.savez_compressed(out / "features.npz", embeddings=features)
        head = np.load(head_path)
        logits = features @ head["weight"].T + head["bias"]
        np.savez_compressed(out / "logits.npz", logits=logits)
        by_crop = {}
        for i, (crop_id, view, box) in enumerate(index):
            by_crop.setdefault(crop_id, {})[view] = {"crop_xyxy": box, "logits": logits[i].tolist()}
        for crop in cfg["crops"]:
            views = by_crop[crop["id"]]
            decision = decide(views["tight"]["logits"], views["context"]["logits"], crop["xyxy"], cfg)
            result["comparisons"].append({**crop, "views": views, "decision": decision})
        result.update(status="COMPLETE", elapsed_seconds=time.perf_counter() - start,
                      features_sha256=sha(out / "features.npz"), logits_sha256=sha(out / "logits.npz"))
        write(out / "results.json", result)
        print(json.dumps({"status": "COMPLETE", "decisions": [r["decision"]["material"] for r in result["comparisons"]]}), flush=True)
    except BaseException as exc:
        result.update(status="FAILED", error=str(exc), traceback=traceback.format_exc())
        write(out / "results.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uk_two_stage_porcelain_v1.json")
    main(parser.parse_args().config)
