#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import struct
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data/image_sources.json").read_text())
UA = "GridSight-UK-v2.4-GitHub-Actions/1.0 (+https://github.com/xm2325/gridsight_uk)"
KNOWN_DIRECT = {
    "POS_190181": "https://s0.geograph.org.uk/photos/19/01/190181_09b95889.jpg",
    "POS_5442616": "https://s0.geograph.org.uk/geophotos/05/44/26/5442616_afefc9f4.jpg",
    "POS_2326530": "https://s0.geograph.org.uk/geophotos/02/32/65/2326530_5fe1bca3.jpg",
}


def fetch(url: str, accept: str = "*/*") -> bytes:
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    assert last is not None
    raise last


def discover_geograph_direct(record_id: str) -> tuple[str, str]:
    photo_id = record_id.removeprefix("POS_")
    page_url = f"https://www.geograph.org.uk/photo/{photo_id}"
    html = fetch(page_url, "text/html,*/*;q=0.1").decode("utf-8", "replace")
    matches = re.findall(r"https://s\d+\.geograph\.org\.uk/(?:photos|geophotos)/[^\"'<> ]+?\.jpg", html)
    if matches:
        # Prefer the first concrete Geograph image URL surfaced by the canonical photo page.
        return matches[0].replace("&amp;", "&"), page_url
    if record_id in KNOWN_DIRECT:
        return KNOWN_DIRECT[record_id], page_url
    raise RuntimeError(f"No direct Geograph JPEG found on {page_url}")


def jpeg_size(data: bytes) -> tuple[int, int]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG")
    i = 2
    sof = {0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF}
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        marker = data[i]; i += 1
        if marker in {0xD8,0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(data):
            break
        seglen = struct.unpack(">H", data[i:i+2])[0]
        if marker in sof:
            if i + 7 > len(data):
                break
            height = struct.unpack(">H", data[i+3:i+5])[0]
            width = struct.unpack(">H", data[i+5:i+7])[0]
            return width, height
        i += seglen
    raise ValueError("JPEG dimensions not found")


def main() -> None:
    rows = []
    for item in MANIFEST["images"]:
        record_id = item["record_id"]
        out = ROOT / "data/images" / item["split"] / item["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        attempts = []
        data = None
        used_url = None
        source_kind = None
        page_url = None

        # Preferred path: canonical Geograph page -> its own direct JPEG derivative.
        try:
            direct, page_url = discover_geograph_direct(record_id)
            data = fetch(direct, "image/jpeg,image/*;q=0.8,*/*;q=0.1")
            used_url = direct
            source_kind = "geograph_page_discovered_runtime_derivative"
        except Exception as exc:
            attempts.append({"kind": "geograph", "error": repr(exc)})

        # Fallback: exact Commons original. This may be rate-limited on hosted runners.
        if data is None:
            try:
                data = fetch(item["url"], "image/jpeg,image/*;q=0.8,*/*;q=0.1")
                used_url = item["url"]
                source_kind = "commons_original"
            except Exception as exc:
                attempts.append({"kind": "commons", "error": repr(exc)})

        if data is None:
            raise SystemExit(f"Could not hydrate {record_id}: {attempts}")
        if not data.startswith(b"\xff\xd8"):
            raise SystemExit(f"Hydrated bytes for {record_id} are not JPEG")

        width, height = jpeg_size(data)
        expected_ratio = item["expected_width_px"] / item["expected_height_px"]
        runtime_ratio = width / height
        ratio_error = abs(runtime_ratio / expected_ratio - 1.0)
        if ratio_error > 0.015:
            raise SystemExit(
                f"Aspect-ratio mismatch for {record_id}: runtime {width}x{height} vs expected "
                f"{item['expected_width_px']}x{item['expected_height_px']} (relative error {ratio_error:.4f})"
            )

        out.write_bytes(data)
        sha256 = hashlib.sha256(data).hexdigest()
        sha1 = hashlib.sha1(data).hexdigest()
        commons_exact = sha256 == item["expected_sha256"] and sha1 == item["expected_sha1"]
        rows.append({
            "record_id": record_id,
            "path": str(out.relative_to(ROOT)),
            "bytes": len(data),
            "width_px": width,
            "height_px": height,
            "runtime_sha256": sha256,
            "runtime_sha1": sha1,
            "source_kind": source_kind,
            "source_url": used_url,
            "geograph_page_url": page_url,
            "commons_original_byte_exact": commons_exact,
            "aspect_ratio_relative_error": ratio_error,
            "attempt_failures": attempts,
            "label_geometry_note": "YOLO labels are normalised; this runtime derivative is accepted only when the image aspect ratio matches the canonical source within 1.5%.",
        })

    report = {
        "status": "PASS",
        "count": len(rows),
        "evidence_type": "runtime-image-hydration",
        "ground_truth_note": "Hydration does not create or modify labels.",
        "images": rows,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/v2_3_frozen_image_download.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
