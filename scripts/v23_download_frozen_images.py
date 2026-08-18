#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, pathlib, urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"data/image_sources.json").read_text())
rows=[]
for item in manifest["images"]:
    out=ROOT/"data/images"/item["split"]/item["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    req=urllib.request.Request(item["url"], headers={"User-Agent":"GridSight-UK-v2.3-Actions/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data=r.read()
    sha256=hashlib.sha256(data).hexdigest()
    sha1=hashlib.sha1(data).hexdigest()
    if sha256 != item["expected_sha256"]:
        raise SystemExit(f"SHA256 mismatch for {item['record_id']}: {sha256} != {item['expected_sha256']}")
    if sha1 != item["expected_sha1"]:
        raise SystemExit(f"SHA1 mismatch for {item['record_id']}: {sha1} != {item['expected_sha1']}")
    out.write_bytes(data)
    rows.append({"record_id":item["record_id"],"path":str(out.relative_to(ROOT)),"bytes":len(data),"sha256":sha256,"sha1":sha1,"url":item["url"]})
report={"status":"PASS","count":len(rows),"images":rows}
(ROOT/"reports").mkdir(exist_ok=True)
(ROOT/"reports/v2_3_frozen_image_download.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
