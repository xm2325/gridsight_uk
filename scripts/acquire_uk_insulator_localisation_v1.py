"""Acquire a frozen UK distribution-insulator localisation diagnostic.

The rectangles are analyst-drawn visible-object references, not expert-reviewed
inspection ground truth. No model output was viewed before this selection froze.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from acquire_uk_material_sources import ROOT, digest, extract, fetch, jpeg_size

OUT = ROOT / "data/external/uk_insulator_localisation_v1"


def source(photo_id, author, asset_group, role, sha256, boxes=(), exclusion_reason=None):
    return {"photo_id": str(photo_id), "author": author, "asset_group": asset_group,
            "role": role, "expected_image_sha256": sha256, "boxes": list(boxes),
            "exclusion_reason": exclusion_reason}


SOURCES = [
    source("554343", "Dave Hitchborne", "skendleby_hundleby_200710_sequence", "prospective_test",
           "c1e40b822248039438223b9db033412bb3661fc81ae8edf4030acafc30d68c2c",
           [[165,5,198,44],[77,95,129,143],[225,33,271,79],[351,149,402,198],
            [215,214,260,264],[270,241,318,293],[67,223,119,276],[480,238,533,294],
            [105,342,158,397],[485,344,539,400],[292,151,322,191],
            [395,228,430,276],[432,248,469,295]]),
    source("554843", "Dave Hitchborne", "skendleby_hundleby_200710_sequence", "prospective_test",
           "cbbf758f9b0e58baf6d4f46e76374ca07bded47001914bcf5fa896b8841348e5",
           [[235,125,335,237],[395,235,475,328]]),
    source("2489239", "Stephen Craven", "naddle_201106", "prospective_test",
           "496d0d1ec2967c4855b996f897a8192b4ff6062d8bacbd5f3c698621a72bd87a",
           [[101,126,120,163],[126,117,145,159],[151,123,171,161]]),
    source("6941217", "Bob Harvey", "eleven_kv_assembly_202107", "prospective_test",
           "032d59ab4b8b0fd241a4d25b2e3cb08815aeea51f997a4de7c24271332245b2e",
           [[178,84,201,120],[232,104,255,131],[283,115,306,144],[204,54,221,92],
            [258,68,277,106],[307,81,325,118],[74,220,89,246],[105,209,122,240],
            [153,208,170,238]]),
    source("6812792", "Bob Harvey", "deeping_st_nicholas_202103", "prospective_test",
           "e5ebf3deb9cbaf232ae7778815ea276b1367ea14370827035ada558a82f1c6b6",
           [[219,59,259,103],[272,79,311,120],[324,105,364,145]]),
    source("2257820", "Robin Stott", "pathlow_201105", "prospective_test",
           "2fd8492af44b06ea6f419e111ccfcaecaff6f008b8b60b892713bd9f725fc729",
           [[14,37,37,64],[48,40,72,68],[82,42,106,70],[117,42,143,71],
            [8,82,34,113],[47,80,74,114],[88,86,116,120],[133,87,163,123]]),
    source("4701966", "Trevor Littlewood", "middle_rigg_201506", "prospective_test",
           "ff6513daaa37148f2edaa65ddf0cb02b873bfa3e1c8f4a6043ad17d0593e9e88",
           [[500,133,525,174],[534,145,561,184]]),
    source("3479839", "Bob Harvey", "pole_transformer_201305", "hard_negative",
           "650c759ab84cbdf3da921ff16709033fb79808cf8423fb74795a16560f0f5dc9", []),
    source("3999343", "Derek Harper", "combeinteignhead_201309", "excluded",
           "5d2624d86d7511e567d1e4c5a52f4d87507814107ee899798f235476c808ef0e",
           exclusion_reason="Target is too small and partially obscured for a stable visible-object reference."),
    source("4334892", "Stephen Burton", "transmission_tower_201408", "excluded",
           "e7435bb7973f10593b55b5a67d75c50afab0dd3de4cf5a7ff9c482508d406448",
           exclusion_reason="Transmission-tower strings are outside this distribution-pole acceptance scope."),
    source("3288920", "Albert Bridge", "shaws_bridge_pylon_201210", "excluded",
           "3efc219e9179cafdbcb509796e949801c40a246a3d6ce1440e00067317614e19",
           exclusion_reason="Transmission-tower strings are outside this distribution-pole acceptance scope."),
    source("7730730", "Stephen Craven", "northallerton_services_202402", "excluded",
           "7fcc3af27dc20469fed085dd1b98f28c032c3965e081c1efd6b109ff14b422ca",
           exclusion_reason="Building-mounted legacy pot insulators create a taxonomy ambiguity."),
    source("3937904", "David Lally", "railway_pot_array_201308", "excluded",
           "7d4f0a9297649bf2cd37127a6dc7b92dfecf5d7bc34aecd2d5a33968b961afdb",
           exclusion_reason="Railway equipment is outside this distribution-pole acceptance scope."),
]


def prior_image_hashes():
    found = {}
    for path in (ROOT / "data/external").glob("*/manifest.json"):
        if path.parent == OUT:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for record in payload.get("records", []):
            value = record.get("image_sha256")
            if value:
                found.setdefault(value, []).append(str(path.relative_to(ROOT)))
    return found


def main(cache=None):
    if OUT.exists():
        raise FileExistsError(f"Existing frozen pool: {OUT}")
    (OUT / "images").mkdir(parents=True)
    (OUT / "pages").mkdir()
    prior = prior_image_hashes()
    accepted_hashes = {s["expected_image_sha256"] for s in SOURCES if s["role"] != "excluded"}
    overlap = {value: prior[value] for value in accepted_hashes if value in prior}
    if overlap:
        raise ValueError(f"Acceptance images already occur in an earlier external manifest: {overlap}")
    records = []
    for item in SOURCES:
        photo_id = item["photo_id"]
        page_url = f"https://www.geograph.org.uk/photo/{photo_id}"
        page = (cache / f"{photo_id}.html").read_bytes() if cache else fetch(page_url)
        image_url, licence_url, title = extract(page, photo_id)
        image = (cache / f"{photo_id}.jpg").read_bytes() if cache else fetch(image_url)
        if digest(image) != item["expected_image_sha256"]:
            raise ValueError(f"Image bytes changed for {photo_id}")
        width, height = jpeg_size(image)
        for box in item["boxes"]:
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"Invalid box for {photo_id}: {box} vs {width}x{height}")
        image_path = OUT / "images" / f"uk_localisation_{photo_id}.jpg"
        page_path = OUT / "pages" / f"{photo_id}.html"
        image_path.write_bytes(image); page_path.write_bytes(page)
        record = {k: v for k, v in item.items() if k != "expected_image_sha256"}
        record.update({"record_id": f"uk_localisation_{photo_id}", "title": title,
                       "photo_page_url": page_url, "image_url": image_url,
                       "licence": "CC BY-SA 2.0", "licence_url": licence_url,
                       "image_file": str(image_path.relative_to(ROOT)), "image_sha256": digest(image),
                       "page_file": str(page_path.relative_to(ROOT)), "page_sha256": digest(page),
                       "width": width, "height": height, "bytes": len(image),
                       "reference_status": "analyst visible-object box; not expert reviewed",
                       "model_inference_performed_before_freeze": False})
        records.append(record)
    accepted = [r for r in records if r["role"] in {"prospective_test", "hard_negative"}]
    manifest = {
        "version": "uk-insulator-localisation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_frozen_before_model_inference": True,
        "model_inference_performed_before_freeze": False,
        "target_definition": "Visible distribution-line insulator body or assembly that attaches or constrains a conductor.",
        "exclusions_from_target": ["transformer bushings and service enclosures", "guy-wire strain apples",
                                    "transmission-tower strings", "railway and building-mounted legacy equipment"],
        "records": records,
        "counts": {role: sum(r["role"] == role for r in records)
                   for role in ("prospective_test", "hard_negative", "excluded")},
        "acceptance_images": len(accepted),
        "positive_reference_boxes": sum(len(r["boxes"]) for r in accepted),
        "acceptance_asset_groups": sorted({r["asset_group"] for r in accepted}),
        "prior_manifest_image_hash_overlap": overlap,
        "claim_boundary": "Analyst visible-object references support a prospective technique diagnostic, not expert-reviewed inspection accuracy or UK population performance."
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": digest((OUT / "manifest.json").read_bytes()),
                      "counts": manifest["counts"], "acceptance_images": len(accepted),
                      "positive_reference_boxes": manifest["positive_reference_boxes"],
                      "asset_groups": len(manifest["acceptance_asset_groups"])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-cache", type=__import__("pathlib").Path,
                        help="Use previously downloaded page/image bytes; hashes are still enforced")
    main(parser.parse_args().verified_cache)
