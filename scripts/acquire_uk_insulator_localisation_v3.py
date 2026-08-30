"""Acquire the third untouched UK localisation cohort before v2 adaptation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from acquire_uk_material_sources import ROOT, digest, extract, fetch, jpeg_size

OUT = ROOT / "data/external/uk_insulator_localisation_v3"


def source(photo_id, author, asset_group, role, sha256, boxes=(), exclusion_reason=None,
           negative_evidence=None):
    return {"photo_id": str(photo_id), "author": author, "asset_group": asset_group,
            "role": role, "expected_image_sha256": sha256, "boxes": list(boxes),
            "exclusion_reason": exclusion_reason, "negative_evidence": negative_evidence}


SOURCES = [
    source("3193950", "Peter Facey", "graces_farm_201209", "prospective_test",
           "487b618c64a5664f0ad5ae0b911e341e3a4edd4ae9af20bfc1ea438090e8af4d",
           [[136,78,165,109],[148,36,177,102],[188,40,215,105],[226,45,252,106],[273,87,299,119]]),
    source("3866162", "Peter Facey", "otterbourne_11kv_201402", "prospective_test",
           "58b33ef95c36ec18a84bdba98b704499cebadde3158333098c9e074d687fdeb7",
           [[134,161,149,183],[149,161,163,183],[162,161,177,184]]),
    source("5298351", "David Howard", "higham_gobion_201702", "prospective_test",
           "dea4e226b6f6db7e2a0a529b46caae7c3d9bace713551a81c9c14c5f4110d25a",
           [[251,43,274,65],[283,20,309,43],[320,38,344,63],
            [251,69,274,92],[285,77,310,100],[321,69,345,92]]),
    source("6512993", "Tiger", "waresley_junction_202006", "prospective_test",
           "eace933935b024dee4a536460d1b37ddba06a188ebac075727396aa2cb320b03",
           [[218,123,239,149],[216,148,238,177],[173,192,197,220],[249,188,275,218]]),
    source("6941195", "Bob Harvey", "market_overton_202108", "prospective_test",
           "0fd29940cc4e3c686110d087073909cb47e168de51072e4121c3b5d036ef8b3a",
           [[74,92,93,119],[158,69,178,102],[234,51,255,84],
            [273,185,303,226],[332,183,361,226],[393,181,423,225]]),
    source("7223669", "Bob Harvey", "grimsthorpe_feed_202108", "prospective_test",
           "4027879d038741b017b90f21bf870126ecb61797bbe25f7170104b85df39f551",
           [[236,77,265,107],[288,91,317,122],[336,86,365,117]]),
    source("7797894", "Bob Harvey", "dunsby_fen_transformer_202404", "prospective_test",
           "b7199b32051ae237fda4a1b5a573eb035a8081424ef8dabf78f11e622b1dd2f7",
           [[148,90,187,127],[191,151,230,184],[362,132,398,160]]),
    source("7545070", "DS Pugh", "newland_park_telecom_202307", "hard_negative",
           "012463bbf390d771d0cf0c416de93fc4cc5ce4a494925c56a228ac20acb8df94", [],
           negative_evidence="The publisher identifies a BT telephone pole; cable terminals are outside the electricity-distribution target."),
    source("4246106", "Keith Evans", "peasenhall_telephone_201410", "hard_negative",
           "25cb4ce7759e9914a134d429d3ca0eae9983646ee24721926fe81281408a0cc8", [],
           negative_evidence="The publisher explicitly identifies old telephone-line insulators; they are outside the electricity-distribution target."),
    source("6941220", "Bob Harvey", "market_overton_202108", "excluded",
           "1c700aedd903a538c6cc02b4c71bb99c6aef24d9abd55bab510d07917423b5cd",
           exclusion_reason="Same asset group as accepted photo 6941195 and contains transformer-bushing/unit ambiguity."),
]


def prior_records():
    found_hashes, found_ids, found_groups = {}, {}, {}
    for path in (ROOT / "data/external").glob("*/manifest.json"):
        if path.parent == OUT:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("records", payload.get("images", [])):
            image_hash = row.get("image_sha256") or row.get("sha256")
            photo_id = str(row.get("photo_id") or row.get("geograph_id") or "")
            asset_group = row.get("asset_group")
            if image_hash:
                found_hashes.setdefault(image_hash, []).append(str(path.relative_to(ROOT)))
            if photo_id:
                found_ids.setdefault(photo_id, []).append(str(path.relative_to(ROOT)))
            if asset_group:
                found_groups.setdefault(asset_group, []).append(str(path.relative_to(ROOT)))
    return found_hashes, found_ids, found_groups


def main(cache=None):
    if OUT.exists():
        raise FileExistsError(f"Existing frozen pool: {OUT}")
    (OUT / "images").mkdir(parents=True)
    (OUT / "pages").mkdir()
    prior_hashes, prior_ids, prior_groups = prior_records()
    accepted_sources = [row for row in SOURCES if row["role"] != "excluded"]
    overlap = {
        "image_hashes": {row["expected_image_sha256"]: prior_hashes[row["expected_image_sha256"]]
                         for row in accepted_sources if row["expected_image_sha256"] in prior_hashes},
        "photo_ids": {row["photo_id"]: prior_ids[row["photo_id"]]
                      for row in accepted_sources if row["photo_id"] in prior_ids},
        "asset_groups": {row["asset_group"]: prior_groups[row["asset_group"]]
                         for row in accepted_sources if row["asset_group"] in prior_groups},
    }
    if any(overlap.values()):
        raise ValueError(f"Acceptance overlaps earlier datasets: {overlap}")
    records = []
    for item in SOURCES:
        photo_id = item["photo_id"]
        page_url = f"https://www.geograph.org.uk/photo/{photo_id}"
        page = (cache / "pages" / f"{photo_id}.html").read_bytes() if cache else fetch(page_url)
        image_url, licence_url, title = extract(page, photo_id)
        image = (cache / "images" / f"{photo_id}.jpg").read_bytes() if cache else fetch(image_url)
        if digest(image) != item["expected_image_sha256"]:
            raise ValueError(f"Image bytes changed: {photo_id}")
        width, height = jpeg_size(image)
        for box in item["boxes"]:
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"Invalid box: {photo_id} {box}")
        image_path = OUT / "images" / f"uk_localisation_v3_{photo_id}.jpg"
        page_path = OUT / "pages" / f"{photo_id}.html"
        image_path.write_bytes(image)
        page_path.write_bytes(page)
        row = {key: value for key, value in item.items() if key != "expected_image_sha256"}
        row.update(record_id=f"uk_localisation_v3_{photo_id}", title=title,
                   photo_page_url=page_url, image_url=image_url,
                   licence="CC BY-SA 2.0", licence_url=licence_url,
                   image_file=str(image_path.relative_to(ROOT)), image_sha256=digest(image),
                   page_file=str(page_path.relative_to(ROOT)), page_sha256=digest(page),
                   width=width, height=height, bytes=len(image),
                   reference_status=("source-evidenced non-target plus analyst no-target decision"
                                     if item["role"] == "hard_negative"
                                     else "analyst visible-object box; not expert reviewed"),
                   model_inference_performed_before_freeze=False)
        records.append(row)
    accepted = [row for row in records if row["role"] in {"prospective_test", "hard_negative"}]
    manifest = {
        "version": "uk-insulator-localisation-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_v2_adapted_model_inference": True,
        "model_inference_performed_before_freeze": False,
        "target_definition": "Visible electricity-distribution-line insulator body or assembly that attaches or constrains a conductor.",
        "exclusions_from_target": ["transformer or recloser bushings", "telecom cable terminals",
                                   "telephone-line insulators", "guy-wire strain units", "railway equipment"],
        "records": records,
        "counts": {role: sum(row["role"] == role for row in records)
                   for role in ("prospective_test", "hard_negative", "excluded")},
        "acceptance_images": len(accepted),
        "positive_reference_boxes": sum(len(row["boxes"]) for row in accepted),
        "acceptance_asset_groups": sorted({row["asset_group"] for row in accepted}),
        "prior_overlap": overlap,
        "claim_boundary": "Prospective small-object technique check with analyst references and source-evidenced telecom negatives; not expert inspection truth or UK population accuracy.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": digest((OUT / "manifest.json").read_bytes()),
                      "counts": manifest["counts"], "acceptance_images": manifest["acceptance_images"],
                      "positive_reference_boxes": manifest["positive_reference_boxes"],
                      "asset_groups": len(manifest["acceptance_asset_groups"])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-cache", type=Path)
    main(parser.parse_args().verified_cache)
