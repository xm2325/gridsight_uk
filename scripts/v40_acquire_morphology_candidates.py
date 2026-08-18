from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v4_0_morphology_candidates"
OUT.mkdir(parents=True, exist_ok=True)
UA = "GridSight-UK-v4.0-morphology-acquisition/1.0 (+https://github.com/xm2325/gridsight_uk)"
RETIRED_HOLDOUTS = {"3437435", "7561805"}

# These four records were already present in the pre-model v2.9 geographic acquisition queue.
# They were not selected into the v2.9 frozen final holdout and are therefore eligible for a new development cycle.
CANDIDATES = [
    {"photo_id": "7072688", "country": "Scotland", "photographer": "Richard Sutcliffe", "url": "https://s0.geograph.org.uk/geophotos/07/07/26/7072688_5aae390b.jpg", "licence": "CC BY-SA 2.0"},
    {"photo_id": "7478407", "country": "Scotland", "photographer": "Richard Sutcliffe", "url": "https://s0.geograph.org.uk/geophotos/07/47/84/7478407_cba3b805.jpg", "licence": "CC BY-SA 2.0"},
    {"photo_id": "6610209", "country": "Wales", "photographer": "Alan Hughes", "url": "https://s0.geograph.org.uk/geophotos/06/61/02/6610209_b56889e7.jpg", "licence": "CC BY-SA 2.0"},
    {"photo_id": "8091164", "country": "England", "photographer": "Daniel Beardsmore", "url": "https://s0.geograph.org.uk/geophotos/08/09/11/8091164_0e5e19aa.jpg", "licence": "CC BY-SA 2.0"},
]


def fetch(url: str) -> bytes:
    errors = []
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "image/jpeg,image/*;q=0.8,*/*;q=0.1"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"download failed for {url}: {' | '.join(errors)}")


def make_contact_sheet(rows):
    thumb_w, thumb_h = 640, 480
    label_h = 64
    canvas = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * 2), "white")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        image = Image.open(ROOT / row["path"]).convert("RGB")
        fitted = ImageOps.contain(image, (thumb_w, thumb_h))
        x = (i % 2) * thumb_w + (thumb_w - fitted.width) // 2
        y0 = (i // 2) * (thumb_h + label_h)
        y = y0 + (thumb_h - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        text = f"{row['record_id']} | {row['country']} | {row['width_px']}x{row['height_px']}\n{row['photographer']} | candidate only - no model inference"
        draw.text(((i % 2) * thumb_w + 8, y0 + thumb_h + 6), text, fill="black", font=font)
    path = ROOT / "reports/v4_0_morphology_contact_sheet.jpg"
    canvas.save(path, quality=94)
    return str(path.relative_to(ROOT))


def main():
    if any(c["photo_id"] in RETIRED_HOLDOUTS for c in CANDIDATES):
        raise RuntimeError("Retired v3.8 holdout leaked into v4.0 candidate queue")
    rows = []
    for candidate in CANDIDATES:
        payload = fetch(candidate["url"])
        path = OUT / f"POS_{candidate['photo_id']}.jpg"
        path.write_bytes(payload)
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format
        if fmt != "JPEG":
            raise RuntimeError(f"Expected JPEG for {path.name}, got {fmt}")
        rows.append({
            **candidate,
            "record_id": f"POS_{candidate['photo_id']}",
            "path": str(path.relative_to(ROOT)),
            "photo_page_url": f"https://www.geograph.org.uk/photo/{candidate['photo_id']}",
            "width_px": width,
            "height_px": height,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sha1": hashlib.sha1(payload).hexdigest(),
            "ground_truth_status": "candidate_only_visual_review_required",
            "model_inference_allowed": False,
            "eligible_use": "new-v4-development-cycle-after-pixel-review; never old-v3.8-holdout tuning",
        })
    contact = make_contact_sheet(rows)
    report = {
        "status": "PASS",
        "version": "v4.0-morphology-diverse-acquisition",
        "selection_basis": "Four records retained from the pre-model v2.9 geographic acquisition queue after excluding both retired v3.8 final-holdout records.",
        "retired_holdouts_excluded": sorted(RETIRED_HOLDOUTS),
        "count": len(rows),
        "candidates": rows,
        "contact_sheet": contact,
        "model_inference_performed": False,
        "next_gate": "pixel-review morphology/annotatability before any component labels or model inference",
    }
    (ROOT / "reports/v4_0_morphology_candidate_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
