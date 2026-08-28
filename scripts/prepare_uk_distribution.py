#!/usr/bin/env python3
"""Download source-audited UK photographs for qualitative transfer, never as GT."""
from __future__ import annotations
import hashlib
import html
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/target_sources/commons_uk_poles_1.json"
OUT = ROOT / "data/external/uk_distribution_pilot_v1"
UA = "GridSight-UK-research/1.0 (+https://github.com/xm2325/gridsight_uk)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def main():
    from PIL import Image, ImageOps, ImageDraw
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "selection_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["metadata_sha256"] != sha(SOURCE):
            raise ValueError("Source metadata changed")
    else:
        candidates = json.loads(SOURCE.read_text())["query"]["pages"]
        previous = set()
        # Exclude every Geograph ID already recorded in the repository's source
        # and holdout files, without opening any frozen holdout pixels.
        for folder in (ROOT / "data", ROOT / "reports"):
            for p in folder.rglob("*.json"):
                if "external" in p.parts:
                    continue
                previous.update(re.findall(r"(?:POS_|photo/)(\d+)", p.read_text(errors="replace")))
        rows = []
        for c in sorted(candidates, key=lambda r: r["pageid"]):
            info = c["imageinfo"][0]
            meta = info["extmetadata"]
            get = lambda k: meta.get(k, {}).get("value", "")
            if min(info["width"], info["height"]) < 600 or get("LicenseShortName") != "CC BY-SA 2.0":
                continue
            matches = re.findall(r"geograph(?:\.org\.uk\s*-\s*|\s+)(\d+)", c["title"], re.I)
            if len(matches) != 1 or matches[0] in previous:
                continue
            # These place categories and coordinates were reviewed before model
            # inference. The narrow range excludes Ireland; no asset IDs inferred.
            if not (49.8 < float(get("GPSLatitude")) < 59 and -5.8 < float(get("GPSLongitude")) < 2):
                continue
            rows.append({"image_id": f"uk_geograph_{matches[0]}", "commons_page_id": c["pageid"],
                         "title": c["title"].removeprefix("File:"), "geograph_id": matches[0],
                         "url": info["url"], "source_page": info["descriptionurl"],
                         "geograph_page": f"https://www.geograph.org.uk/photo/{matches[0]}",
                         "publisher_sha1": info["sha1"], "published_bytes": info["size"],
                         "width": info["width"], "height": info["height"],
                         "author": clean(get("Artist")), "attribution": clean(get("Attribution")),
                         "license": get("LicenseShortName"), "license_url": get("LicenseUrl"),
                         "capture_date": get("DateTimeOriginal"), "source_categories": get("Categories"),
                         "split": "qualitative_only", "ground_truth_status": "NONE",
                         "material_reference": None})
        plan = {"id": "uk-distribution-qualitative-v1", "metadata_sha256": sha(SOURCE),
                "query": '"electricity pole" "geograph" in Commons file namespace; first 50 API results',
                "selection": "Original minimum edge >=600px, explicit CC BY-SA 2.0, GB place metadata, all eligible candidates; selected before inference",
                "previous_geograph_ids_excluded": sorted(previous), "images": rows,
                "limitations": ["Not a random UK asset sample", "No human box or material ground truth",
                                "No training, checkpoint selection, threshold tuning or quantitative UK claims",
                                "Ground-level photos differ from aerial inspection; distant assets may be too small"]}
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    results = []
    old = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else None
    receipts = {r["image_id"]: r for r in old["images"]} if old else {}
    for r in plan["images"]:
        path = OUT / "images" / f"{r['image_id']}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            data = path.read_bytes()
        else:
            for attempt in range(4):
                try:
                    req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        data = response.read()
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429:
                        delay = max(120, int(exc.headers.get("Retry-After", "120")))
                        print(json.dumps({"event": "PUBLISHER_RATE_LIMIT", "wait_seconds": delay}), flush=True)
                        if attempt == 3:
                            raise
                        time.sleep(delay)
                    else:
                        raise
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            time.sleep(10)
        if hashlib.sha1(data).hexdigest() != r["publisher_sha1"] or len(data) != r["published_bytes"]:
            raise ValueError("Original photo bytes differ from Commons metadata")
        with Image.open(io.BytesIO(data)) as im:
            if im.size != (r["width"], r["height"]):
                raise ValueError("Original dimensions changed")
            im.verify()
        if not path.exists():
            path.write_bytes(data)
        row = {**r, "image_file": str(path.relative_to(OUT)), "sha256": sha(path)}
        if r["image_id"] in receipts and receipts[r["image_id"]] != row:
            raise ValueError("Existing photo receipt changed")
        results.append(row)
        print(json.dumps({"event": "UK_PHOTO_VERIFIED", "image_id": r["image_id"]}), flush=True)
    manifest = {**plan, "selection_plan_sha256": sha(plan_path), "images": results, "count": len(results), "status": "VERIFIED_ORIGINAL_PHOTOGRAPHS_NO_GROUND_TRUTH"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for offset in range(0, len(results), 12):
        group = results[offset:offset+12]
        sheet = Image.new("RGB", (1600, ((len(group)+3)//4)*330), "#f4f5f7")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(group):
            with Image.open(OUT / r["image_file"]) as im:
                im = ImageOps.contain(im.convert("RGB"), (400, 288))
                x, y = (j % 4)*400, (j//4)*330
                sheet.paste(im, (x+(400-im.width)//2, y+(288-im.height)//2))
                draw.text((x+7, y+292), r["image_id"] + " | NO GT", fill="#162838")
        sheet.save(OUT / f"contact_{offset//12+1}.jpg", quality=91)
    print(json.dumps({"event": "UK_PILOT_VERIFIED", "count": len(results), "manifest_sha256": sha(OUT/"manifest.json")}))


if __name__ == "__main__":
    main()
