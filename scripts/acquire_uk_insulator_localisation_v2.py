"""Acquire a second untouched UK localisation cohort before adapted-model inference."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from acquire_uk_material_sources import ROOT, digest, extract, fetch, jpeg_size

OUT = ROOT / "data/external/uk_insulator_localisation_v2"


def source(photo_id, author, asset_group, role, sha256, boxes=(), exclusion_reason=None):
    return {"photo_id": str(photo_id), "author": author, "asset_group": asset_group,
            "role": role, "expected_image_sha256": sha256, "boxes": list(boxes),
            "exclusion_reason": exclusion_reason}


SOURCES = [
    source("380261", "Richard Mudhar", "ufford_200703", "prospective_test",
           "812563f1a6e14a60d5fbb024008ef34e15d4a6e62deac23440b476acee4979f1",
           [[257,61,282,88],[273,38,299,66],[365,45,390,72],[388,66,414,94]]),
    source("3772586", "Derek Harper", "tredis_201311", "prospective_test",
           "7952b6a9271e3638f9f509b4ab72d11918129e191e516195f94394e0dbe5dcb1",
           [[131,83,154,109],[178,70,201,101],[238,62,263,96],
            [194,157,227,199],[239,155,271,198]]),
    source("5922372", "Andrew Tryon", "kirkmichael_recloser_201809", "prospective_test",
           "93ef7bfb4d6cc59cc0fd10c91c3f2a04bbf157f1133f158f4a35615b8040219e",
           [[171,113,200,148],[200,98,230,138]]),
    source("7989489", "Morgan Will", "tattenhall_33kv_202503", "prospective_test",
           "b14521f9d6eb6188997455dc171e6a7a4579bce1dc1830903dd753c350a9445f",
           [[250,18,270,46],[293,15,314,45],[333,17,354,46]]),
    source("482081", "D-G-Seamon", "acol_200706", "hard_negative",
           "d1bb85c9eb97469dcbf499f392f647db7d8d15997ea924c1540eaaa49a355a5a", []),
    source("7828514", "P Harris", "anstiebury_202407", "excluded",
           "1e3dcd4c2bff9cb6a68a1a1fd9548172fa7531abcba6a1652467c1d0662424c3",
           exclusion_reason="Multiple targets are too small and blurred for stable visible-object boxes."),
    source("8351149", "Anne Burgess", "lochuisge_202606", "excluded",
           "59000016d5caf15af2191d7d3e22e1aaaf1ed5cb1436317989fbd226acdfa4e7",
           exclusion_reason="The distant pole-top target spans only a few source pixels."),
]


def prior_hashes():
    found = {}
    for path in (ROOT/"data/external").glob("*/manifest.json"):
        if path.parent == OUT: continue
        try: payload=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError): continue
        for row in payload.get("records",payload.get("images",[])):
            value=row.get("image_sha256") or row.get("sha256")
            if value: found.setdefault(value,[]).append(str(path.relative_to(ROOT)))
    return found


def main(cache=None):
    if OUT.exists(): raise FileExistsError(f"Existing frozen pool: {OUT}")
    (OUT/"images").mkdir(parents=True);(OUT/"pages").mkdir()
    previous=prior_hashes(); accepted={r["expected_image_sha256"] for r in SOURCES if r["role"]!="excluded"}
    overlap={value:previous[value] for value in accepted if value in previous}
    if overlap: raise ValueError(f"Acceptance images overlap prior manifests: {overlap}")
    records=[]
    for item in SOURCES:
        pid=item["photo_id"];page_url=f"https://www.geograph.org.uk/photo/{pid}"
        page=(cache/f"{pid}.html").read_bytes() if cache else fetch(page_url)
        image_url,licence_url,title=extract(page,pid)
        image=(cache/f"{pid}.jpg").read_bytes() if cache else fetch(image_url)
        if digest(image)!=item["expected_image_sha256"]: raise ValueError(f"Image bytes changed: {pid}")
        width,height=jpeg_size(image)
        for box in item["boxes"]:
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=width and 0<=y0<y1<=height): raise ValueError(f"Invalid box: {pid} {box}")
        image_path=OUT/"images"/f"uk_localisation_v2_{pid}.jpg";page_path=OUT/"pages"/f"{pid}.html"
        image_path.write_bytes(image);page_path.write_bytes(page)
        row={k:v for k,v in item.items() if k!="expected_image_sha256"}
        row.update(record_id=f"uk_localisation_v2_{pid}",title=title,photo_page_url=page_url,
                   image_url=image_url,licence="CC BY-SA 2.0",licence_url=licence_url,
                   image_file=str(image_path.relative_to(ROOT)),image_sha256=digest(image),
                   page_file=str(page_path.relative_to(ROOT)),page_sha256=digest(page),
                   width=width,height=height,bytes=len(image),
                   reference_status="analyst visible-object box; not expert reviewed",
                   model_inference_performed_before_freeze=False)
        records.append(row)
    accepted_rows=[r for r in records if r["role"] in {"prospective_test","hard_negative"}]
    manifest={"version":"uk-insulator-localisation-v2","created_at":datetime.now(timezone.utc).isoformat(),
              "selection_frozen_before_adapted_model_inference":True,
              "model_inference_performed_before_freeze":False,
              "target_definition":"Visible distribution-line insulator body or assembly that attaches or constrains a conductor.",
              "exclusions_from_target":["transformer or recloser bushings","service enclosures","guy-wire strain units","railway equipment"],
              "records":records,"counts":{role:sum(r["role"]==role for r in records)
                                            for role in ("prospective_test","hard_negative","excluded")},
              "acceptance_images":len(accepted_rows),
              "positive_reference_boxes":sum(len(r["boxes"]) for r in accepted_rows),
              "acceptance_asset_groups":sorted({r["asset_group"] for r in accepted_rows}),
              "prior_manifest_image_hash_overlap":overlap,
              "claim_boundary":"Small prospective technique check with analyst references; not expert inspection truth or UK population accuracy."}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"manifest_sha256":digest((OUT/"manifest.json").read_bytes()),
                      "counts":manifest["counts"],"acceptance_images":manifest["acceptance_images"],
                      "positive_reference_boxes":manifest["positive_reference_boxes"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verified-cache",type=Path)
    main(parser.parse_args().verified_cache)
