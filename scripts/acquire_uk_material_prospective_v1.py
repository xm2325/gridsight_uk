"""Acquire and freeze a source-evidenced UK material transfer pool.

The rectangles are analyst-selected oracle regions tied to publisher text. They
are not expert inspection labels and are never generated from a model.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from acquire_uk_material_sources import ROOT, digest, extract, fetch, jpeg_size

OUT = ROOT / "data/external/uk_material_prospective_v1"


def row(photo_id, author, asset_group, material, role, expected_sha, evidence, regions=(),
        evidence_page_id=None, exclusion_reason=None):
    return {"photo_id": str(photo_id), "author": author, "asset_group": asset_group,
            "material": material, "role": role, "expected_image_sha256": expected_sha,
            "evidence_excerpt": evidence, "evidence_page_id": str(evidence_page_id or photo_id),
            "regions": list(regions), "exclusion_reason": exclusion_reason}


SOURCES = [
    row("3209028", "Peter Facey", "mayles_lane_4ye030_20120524", "glass", "adaptation",
        "2a891658aed1f96a6279cdacfc02548b97cfdb09d45ec87260622db39cf24361",
        "These insulators are made of toughened glass.", [[205, 65, 265, 355]]),
    row("3208894", "Peter Facey", "mayles_lane_4ye030_20120524", "porcelain_ceramic", "adaptation",
        "0661fc7fd65b8a1aae54cff0d6a4917e73fed51010946891cdd8c9149341d33b",
        "There are 22 porcelain insulators in each of the two strings.", [[105, 105, 195, 500]]),
    row("770248", "Peter Facey", "boyatt_wood_pylon_damage_20080408", "glass", "adaptation",
        "346c724e755ba01fc7563c9089101e97da4ea4b537aee587a0ff948fb321f2cf",
        "The glass insulators are vulnerable to catapults or air guns.",
        [[165, 110, 225, 240], [295, 250, 335, 350], [140, 360, 195, 515], [312, 500, 360, 630]]),
    row("770272", "Peter Facey", "boyatt_wood_pylon_damage_20080408", "porcelain_ceramic", "adaptation",
        "d105a16068e99024433b9e875febcfdff8c16d6e7127c6dafd8ce641658feff7",
        "Porcelain insulators are used for the replacement", [[258, 185, 325, 395]]),
    row("7175466", "Bob Harvey", "swayfield_refurbishment_20220515", "glass", "adaptation",
        "49c44665fe44c8f1d6e3a1f6bc7965abf6fccc515dcb8c736fcede86fe76991c",
        "The new ones, as seen in [[7175466]] are a single glass insulator", [[205, 140, 285, 385]],
        evidence_page_id="7175472"),
    row("7175472", "Bob Harvey", "swayfield_refurbishment_20220515", "porcelain_ceramic", "adaptation",
        "d0ea64624930bcdb21397674e1913a7d7b165c173a1aea056e086c1c1d907f48",
        "4 wires hung by a doubled ceramic insulator", [[175, 150, 270, 465]]),
    row("7402738", "Alan Murray-Rust", "glint_sulators_20230208", "glass", "prospective_test",
        "27ff406d8a2148ba53711c670544ff63909a08649db52a62c5e95987319956c5",
        "The clear glass insulators catch the sun", [[105, 130, 180, 200], [340, 195, 415, 260],
        [40, 205, 105, 285], [255, 245, 330, 325], [5, 295, 70, 370], [350, 365, 430, 430]]),
    row("2566302", "Ben Harris", "odd_one_out_20110821", "glass", "prospective_test",
        "cb59deadd512b8f8fc096b866f3d7285d137b7c4b4a223b6b2863f0191d54691",
        "All the other insulators on this pylon used clear glass throughout", [[195, 185, 280, 490]]),
    row("3809215", "Peter Facey", "park_hills_wood_20140110", "glass", "prospective_test",
        "3e596bb5c8101626c33d57a9b03f0d3fe765d847d5e03bc6414ed4577437442d",
        "showing up the glass insulators", [[165, 65, 200, 145], [285, 65, 325, 145],
        [170, 175, 205, 260], [310, 180, 350, 265], [165, 285, 205, 375], [320, 285, 360, 375]]),
    row("2159569", "Walter Baxter", "selkirk_a707_20101202", "porcelain_ceramic", "prospective_test",
        "53f002834b81cfda21a944dc63ac4ad5098142e9eec9ea0ffb2b5969947be925",
        "These brown glaze finished porcelain pin-type insulators support the conductors",
        [[95, 235, 165, 295], [205, 145, 280, 220], [350, 80, 425, 155]]),
    row("6812816", "Bob Harvey", "deeping_st_nicholas_cross_tree_20210301", "porcelain_ceramic", "prospective_test",
        "c92ce86e109ea62e7a2e6f98e72b31e8a766e85772354d1ffed61b4f7ffec6fb",
        "the three phases are only insulated by a single ceramic disk", [[335, 60, 415, 140], [440, 120, 520, 200]]),
    row("6714446", "Peter Facey", "chilcomb_11kv_substation_20201225", "porcelain_ceramic", "excluded",
        "6981e3921991223d308b55f199507c7c5337aec5235b00f09c61d516b45cfe19",
        "four separate wires strung on ceramic insulators", exclusion_reason="Caption refers to distant wires beyond the blue car; no unambiguous target region."),
    row("3328344", "Ashley Dace", "gladstone_pottery_museum_20130627", "porcelain_ceramic", "excluded",
        "695f4facebf248fa13c9222506430dcf1b806c5a192e5cfbfb35a578b3ae5339",
        "Ceramics are excellent insulators", exclusion_reason="Museum material exemplar, not an in-service overhead-line asset."),
    row("4472918", "Bikeboy", "potters_bar_line_work_20150512", "glass", "excluded",
        "852c71a39073256337b37fce52fac68e3f63dfa49961ff15a353922ccf0be5bb",
        "crawling over the glass insulators", evidence_page_id="4472909",
        exclusion_reason="Cross-page evidence is compatible, but individual strings are too small for the frozen ROI diagnostic."),
    row("2595323", "Ed Lloyd-Hughes", "chester_zoo_line_work_20110717", None, "excluded",
        "f6e4fc3681e3fefd8521ec3aeb5820745f4934d06babc1c4cc22c25e8afc5e66",
        "lowering the old pot insulators balanced by the new glass set going up",
        exclusion_reason="Mixed old/new materials are visible without a sufficiently unambiguous region association."),
]


def main():
    if OUT.exists():
        raise FileExistsError(f"Existing frozen pool: {OUT}")
    (OUT / "images").mkdir(parents=True)
    (OUT / "pages").mkdir()
    records = []
    page_cache = {}
    for source in SOURCES:
        photo_id = source["photo_id"]
        page_url = f"https://www.geograph.org.uk/photo/{photo_id}"
        page = fetch(page_url)
        image_url, licence_url, title = extract(page, photo_id)
        image = fetch(image_url)
        if digest(image) != source["expected_image_sha256"]:
            raise ValueError(f"Image bytes changed for {photo_id}")
        width, height = jpeg_size(image)
        for box in source["regions"]:
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"Invalid region for {photo_id}: {box} vs {width}x{height}")
        evidence_id = source["evidence_page_id"]
        evidence_page = page if evidence_id == photo_id else page_cache.get(evidence_id)
        if evidence_page is None:
            evidence_page = fetch(f"https://www.geograph.org.uk/photo/{evidence_id}")
            page_cache[evidence_id] = evidence_page
        searchable = html.unescape(evidence_page.decode("latin-1")).lower()
        if source["evidence_excerpt"].lower() not in searchable:
            raise ValueError(f"Evidence excerpt missing for {photo_id} on {evidence_id}")
        image_path = OUT / "images" / f"uk_material_{photo_id}.jpg"
        page_path = OUT / "pages" / f"{photo_id}.html"
        image_path.write_bytes(image)
        page_path.write_bytes(page)
        record = {k: v for k, v in source.items() if k != "expected_image_sha256"}
        record.update({"record_id": f"uk_material_{photo_id}", "title": title,
                       "photo_page_url": page_url, "image_url": image_url,
                       "licence": "CC BY-SA 2.0", "licence_url": licence_url,
                       "image_file": str(image_path.relative_to(ROOT)), "image_sha256": digest(image),
                       "page_file": str(page_path.relative_to(ROOT)), "page_sha256": digest(page),
                       "evidence_page_url": f"https://www.geograph.org.uk/photo/{evidence_id}",
                       "evidence_page_sha256": digest(evidence_page), "width": width, "height": height,
                       "bytes": len(image), "region_status": "source-assisted analyst oracle region; not expert reviewed",
                       "model_inference_performed_before_freeze": False})
        records.append(record)
    adaptation_groups = {r["asset_group"] for r in records if r["role"] == "adaptation"}
    test_groups = {r["asset_group"] for r in records if r["role"] == "prospective_test"}
    if adaptation_groups & test_groups:
        raise ValueError("Asset group leakage")
    manifest = {"version": "uk-material-prospective-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "selection_frozen_before_model_inference": True, "records": records,
                "counts": {role: sum(r["role"] == role for r in records)
                           for role in ("adaptation", "prospective_test", "excluded")},
                "region_counts": {role: sum(len(r["regions"]) for r in records if r["role"] == role)
                                  for role in ("adaptation", "prospective_test")},
                "adaptation_asset_groups": sorted(adaptation_groups), "test_asset_groups": sorted(test_groups),
                "asset_group_overlap": False, "polymer_test_targets": 0,
                "claim_boundary": "Publisher text supports material identity; rectangles are analyst-selected oracle regions, not expert inspection labels. The prospective split supports a source-evidenced UK diagnostic, not population-level UK accuracy."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest_sha256": digest((OUT / "manifest.json").read_bytes()),
                      "counts": manifest["counts"], "region_counts": manifest["region_counts"]}, indent=2))


if __name__ == "__main__":
    main()
