#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/images/test/POS_2326530.jpg"
REPORT = ROOT / "reports/v2_3_showcase_image_download.json"

SOURCES = [
    {
        "kind": "geograph_direct_runtime_derivative",
        "url": "https://s0.geograph.org.uk/geophotos/02/32/65/2326530_5fe1bca3.jpg",
    },
    {
        "kind": "commons_original_fallback",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/37/Electricity_Pylon_-_geograph.org.uk_-_2326530.jpg",
    },
]


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "GridSight-UK-v2.3-GitHub-Actions/1.0 (+https://github.com/xm2325/gridsight_uk)",
                    "Accept": "image/jpeg,image/*;q=0.8,*/*;q=0.1",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    assert last is not None
    raise last


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for source in SOURCES:
        try:
            data = fetch(source["url"])
            if not data.startswith(b"\xff\xd8"):
                raise RuntimeError("downloaded bytes are not a JPEG")
            OUT.write_bytes(data)
            record = {
                "status": "PASS",
                "record_id": "POS_2326530",
                "source_kind": source["kind"],
                "source_url": source["url"],
                "runtime_sha256": hashlib.sha256(data).hexdigest(),
                "runtime_sha1": hashlib.sha1(data).hexdigest(),
                "bytes": len(data),
                "identity_note": "Runtime image identity is recorded from downloaded bytes. It is not asserted byte-identical to the Commons original unless hashes match independently.",
                "ground_truth_note": "This download does not create or modify labels.",
            }
            REPORT.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record, indent=2))
            return
        except Exception as exc:
            errors.append({"source": source, "error": repr(exc)})
    failure = {"status": "FAIL", "record_id": "POS_2326530", "errors": errors}
    REPORT.write_text(json.dumps(failure, indent=2) + "\n")
    print(json.dumps(failure, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
