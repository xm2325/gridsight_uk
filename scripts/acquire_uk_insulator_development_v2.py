"""Acquire one source-preserved 33 kV development image after v3 was frozen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from acquire_uk_material_sources import ROOT, digest, extract, fetch, jpeg_size

OUT = ROOT / "data/external/uk_insulator_development_v2"
PHOTO_ID = "7989492"
EXPECTED_SHA = "6db9a6764119b3b71dacccb2b1ac5abbae2513cc549ef8d05807354adc86096b"
BOXES = [[162,147,201,215],[287,38,327,113],[428,189,470,261]]
V3_MANIFEST = ROOT / "data/external/uk_insulator_localisation_v3/manifest.json"
V3_SHA = "d74f206e506c9c61303cdf20c092c44c107332cc3931ccf0f6a8079e68ac50ac"


def main(cache=None):
    if OUT.exists():
        raise FileExistsError(f"Existing development source: {OUT}")
    if digest(V3_MANIFEST.read_bytes()) != V3_SHA:
        raise ValueError("The untouched v3 acceptance set was not frozen first")
    v3 = json.loads(V3_MANIFEST.read_text())
    accepted_hashes = {row["image_sha256"] for row in v3["records"] if row["role"] != "excluded"}
    accepted_ids = {row["photo_id"] for row in v3["records"] if row["role"] != "excluded"}
    if EXPECTED_SHA in accepted_hashes or PHOTO_ID in accepted_ids:
        raise ValueError("Development image crosses the v3 acceptance boundary")
    page_url = f"https://www.geograph.org.uk/photo/{PHOTO_ID}"
    page = (cache / "pages" / f"{PHOTO_ID}.html").read_bytes() if cache else fetch(page_url)
    image_url, licence_url, title = extract(page, PHOTO_ID)
    image = (cache / "images" / f"{PHOTO_ID}.jpg").read_bytes() if cache else fetch(image_url)
    if digest(image) != EXPECTED_SHA:
        raise ValueError("Development image bytes changed")
    width, height = jpeg_size(image)
    for box in BOXES:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Invalid box: {box}")
    (OUT / "images").mkdir(parents=True)
    (OUT / "pages").mkdir()
    image_path = OUT / "images" / f"uk_development_{PHOTO_ID}.jpg"
    page_path = OUT / "pages" / f"{PHOTO_ID}.html"
    image_path.write_bytes(image)
    page_path.write_bytes(page)
    record = {
        "record_id": f"uk_development_{PHOTO_ID}", "photo_id": PHOTO_ID,
        "title": title, "author": "Morgan Will", "asset_group": "tattenhall_33kv_202503",
        "role": "development_only", "boxes": BOXES, "photo_page_url": page_url,
        "image_url": image_url, "licence": "CC BY-SA 2.0", "licence_url": licence_url,
        "image_file": str(image_path.relative_to(ROOT)), "image_sha256": digest(image),
        "page_file": str(page_path.relative_to(ROOT)), "page_sha256": digest(page),
        "width": width, "height": height, "bytes": len(image),
        "reference_status": "analyst visible-object box; not expert reviewed",
        "selection_note": "Selected for development after the v2 33 kV miss; same asset-group label as consumed v2 photo 7989489.",
        "model_inference_performed_on_this_image_before_freeze": False,
    }
    manifest = {
        "version": "uk-insulator-development-v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "v3_acceptance_manifest_sha256": V3_SHA, "v3_acceptance_read_for_training": False,
        "records": [record], "claim_boundary": "Development-only analyst reference from a consumed asset cluster; never independent evaluation truth.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": digest((OUT / "manifest.json").read_bytes()),
                      "images": 1, "boxes": len(BOXES)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-cache", type=Path)
    main(parser.parse_args().verified_cache)
