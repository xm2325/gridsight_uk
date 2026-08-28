#!/usr/bin/env python3
"""Acquire a circuit-separated EPRI pilot using verified public ZIP byte ranges.

No remote code is executed. Original CSV, polygons, ZIP metadata, original JPEG
bytes, and hashes are retained. A plan is frozen before any model inference.
"""
from __future__ import annotations

import argparse
import ast
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "runtime/target_sources"
OUT = ROOT / "data/external/epri_components_v1"
BASE = "https://publicstorageaccnt.blob.core.windows.net/drone-distribution-inspection-imagery"
CSV_SHA = "a2b75c6d6aa08e2e7620ca007eb9b6b52546b4ee119f2105ec1ac689bedd3a52"
SEED = "gridsight-epri-components-20260827-v1"
NAMES = ["pole", "crossarm", "insulator"]
SPLITS = {"train": {1: 80, 2: 80, 3: 80, 6: 80}, "dev": {4: 80}, "eval": {5: 50, 7: 50}}
UA = "GridSight-UK-research/1.0 (+https://github.com/xm2325/gridsight_uk)"


def digest(data):
    if isinstance(data, (str, Path)):
        with open(data, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def blobs():
    result = {}
    for b in ET.parse(SOURCES / "epri_blob_listing.xml").findall(".//Blob"):
        name = b.findtext("Name")
        result[name] = {"name": name, "url": f"{BASE}/{name}",
                        "size": int(b.findtext("Properties/Content-Length")),
                        "etag": b.findtext("Properties/Etag"),
                        "last_modified": b.findtext("Properties/Last-Modified"),
                        "publisher_md5_base64": b.findtext("Properties/Content-MD5")}
    return result


def get_range(blob, start, length):
    if length <= 0:
        return b""
    end = start + length - 1
    if start < 0 or end >= blob["size"]:
        raise ValueError("Byte range outside the published blob")
    headers = {"User-Agent": UA, "Range": f"bytes={start}-{end}",
               "If-Match": '"' + blob["etag"].strip('"') + '"', "Accept-Encoding": "identity"}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(blob["url"], headers=headers), timeout=90) as r:
                if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{blob['size']}":
                    raise ValueError("Server did not return the exact requested partial content")
                if r.headers.get("ETag", "").strip('"') != blob["etag"].strip('"'):
                    raise ValueError("Publisher blob changed since the inventory was frozen")
                payload = r.read(length + 1)
            if len(payload) != length:
                raise ValueError("Short or oversized byte-range response")
            return payload
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class RangeReader(io.RawIOBase):
    """Seekable reader with an on-disk cache; used for ZIP central directories only."""
    def __init__(self, blob):
        self.blob, self.position = blob, 0
        self.cache = OUT / "archive_index" / blob["name"]
        self.cache.mkdir(parents=True, exist_ok=True)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else (self.position if whence == 1 else self.blob["size"]) + offset
        if pos < 0:
            raise ValueError("Negative seek")
        self.position = pos
        return pos

    def read(self, size=-1):
        size = self.blob["size"] - self.position if size < 0 else min(size, self.blob["size"] - self.position)
        if size <= 0:
            return b""
        if size > 16 * 1024 * 1024:
            raise ValueError("Refusing an oversized archive-index request")
        p = self.cache / f"{self.position}-{size}.bin"
        receipt = p.with_suffix(".json")
        if p.exists():
            meta = json.loads(receipt.read_text())
            data = p.read_bytes()
            if meta["etag"] != self.blob["etag"] or meta["sha256"] != digest(data) or len(data) != size:
                raise ValueError("Corrupt cached ZIP index")
        else:
            data = get_range(self.blob, self.position, size)
            p.write_bytes(data)
            write_json(receipt, {"etag": self.blob["etag"], "sha256": digest(data), "start": self.position, "size": size})
        self.position += size
        return data


def index_archives():
    catalog = blobs()
    for circuit in sorted({c for split in SPLITS.values() for c in split}):
        blob = catalog[f"Circuit{circuit}.zip"]
        target = OUT / "archive_index" / f"circuit_{circuit}.json"
        if target.exists():
            if json.loads(target.read_text())["blob"] != blob:
                raise ValueError("Existing index does not match publisher inventory")
            continue
        with zipfile.ZipFile(RangeReader(blob)) as z:
            entries = [{"member": i.filename, "file_size": i.file_size, "compress_size": i.compress_size,
                        "compress_type": i.compress_type, "header_offset": i.header_offset,
                        "crc32": i.CRC, "flags": i.flag_bits}
                       for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith((".jpg", ".jpeg"))]
        write_json(target, {"blob": blob, "entries": entries})
        print(json.dumps({"event": "INDEXED", "circuit": circuit, "images": len(entries),
                          "median_jpeg_bytes": sorted(e["file_size"] for e in entries)[len(entries)//2]}), flush=True)


def labels():
    path = SOURCES / "Overhead-Distribution-Labels.csv"
    if digest(path) != CSV_SHA:
        raise ValueError("Official label CSV checksum changed")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        key = row["External ID"].lower()
        if key in result:
            raise ValueError("Duplicate official image label key")
        # Official CSV stores Python literal dictionaries, not executable code.
        result[key] = ast.literal_eval(row["Label"])
    return result


def make_plan():
    target = OUT / "selection_plan.json"
    if target.exists():
        p = json.loads(target.read_text())
        print(json.dumps({"event": "PLAN_ALREADY_FROZEN", "sha256": digest(target), "images": len(p["images"])}))
        return
    source_labels = labels()
    selected, unavailable = [], []
    for split, circuits in SPLITS.items():
        for circuit, count in circuits.items():
            index_path = OUT / "archive_index" / f"circuit_{circuit}.json"
            index = json.loads(index_path.read_text())
            eligible = []
            for entry in index["entries"]:
                name = PurePosixPath(entry["member"].replace("\\", "/")).name
                if not re.fullmatch(rf"{circuit} \(\d+\)\.jpe?g", name, re.I):
                    continue
                if name.lower() not in source_labels:
                    unavailable.append({"circuit": circuit, "file_name": name, "reason": "no official CSV row"})
                    continue
                eligible.append((name, entry))
            eligible.sort(key=lambda x: digest((SEED + "/" + str(circuit) + "/" + x[0].lower()).encode()))
            if len(eligible) < count:
                raise ValueError("Insufficient files in the fixed circuit")
            for name, entry in eligible[:count]:
                selected.append({"image_id": f"epri_c{circuit}_{re.search(r'\((\d+)\)', name).group(1)}",
                                 "file_name": name, "split": split, "circuit": circuit,
                                 "blob": index["blob"], "zip_entry": entry,
                                 "source_labels": source_labels[name.lower()]})
    compressed = sum(r["zip_entry"]["compress_size"] for r in selected)
    if compressed > 3_500_000_000:
        raise ValueError(f"Fixed pilot exceeds 3.5 GB transfer budget: {compressed}")
    counts = {s: collections.Counter(o.get("value") for r in selected if r["split"] == s
                                    for o in r["source_labels"].get("objects", [])) for s in SPLITS}
    plan = {"id": SEED, "status": "FROZEN_BEFORE_IMAGE_DOWNLOAD_AND_MODEL_INFERENCE", "seed": SEED,
            "source_csv_sha256": CSV_SHA, "publisher": "EPRI, P. Kulkarni, D. Lewis",
            "dataset_title": "Drone-based Distribution Inspection Imagery 1.0",
            "doi": "10.34740/kaggle/dsv/3803175", "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "source_page": "https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery",
            "access": "Public unsigned Azure blob URLs, no credentials, forms, or private access used",
            "classes": NAMES, "split_circuits": SPLITS,
            "selection": "Fixed SHA256 order within publisher circuit; no score- or appearance-based selection",
            "compressed_bytes_budget": 3_500_000_000, "planned_compressed_bytes": compressed,
            "all_source_class_counts": counts, "unlabelled_files_excluded": unavailable,
            "limitations": ["Circuit IDs are publisher filename groups, not independently verified asset IDs",
                            "Crossarm material, insulator material and pole-top are not annotated",
                            "Only three polygon object classes are scored; wire polylines are not converted to boxes",
                            "Pretraining overlap is unknown; no UK performance claim"], "images": selected}
    write_json(target, plan)
    print(json.dumps({"event": "PLAN_FROZEN", "sha256": digest(target), "images": len(selected),
                      "compressed_bytes": compressed, "counts": counts}), flush=True)


def polygon_box(points, width, height):
    if len(points) < 3:
        raise ValueError("Polygon has fewer than three points")
    xs, ys = [float(p["x"]) for p in points], [float(p["y"]) for p in points]
    if not all(math.isfinite(x) for x in xs + ys):
        raise ValueError("Non-finite polygon coordinate")
    raw = [min(xs), min(ys), max(xs), max(ys)]
    box = [max(0., raw[0]), max(0., raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Empty clipped bounding box")
    return box, raw != box


def download_one(row):
    from PIL import Image
    p = OUT / "images" / row["split"] / (row["image_id"] + ".jpg")
    receipt = OUT / "receipts" / (row["image_id"] + ".json")
    if p.exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["sha256"] != digest(p) or previous["selection_plan_sha256"] != digest(OUT / "selection_plan.json"):
            raise ValueError("Previously downloaded image differs from its receipt")
        return previous
    entry, blob = row["zip_entry"], row["blob"]
    if entry["flags"] & 1:
        raise ValueError("Encrypted ZIP member is not allowed")
    header = get_range(blob, entry["header_offset"], 30)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50 or fields[3] != entry["compress_type"]:
        raise ValueError("Invalid local ZIP member header")
    extra_len = fields[-2] + fields[-1]
    packed = get_range(blob, entry["header_offset"] + 30, extra_len + entry["compress_size"])
    local_name = packed[:fields[-2]].decode("utf-8" if fields[2] & 0x800 else "cp437")
    if local_name != entry["member"]:
        raise ValueError("Local member name differs from central directory")
    compressed = packed[extra_len:]
    if entry["compress_type"] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif entry["compress_type"] == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError("Unsupported compression method")
    if len(data) != entry["file_size"] or zlib.crc32(data) != entry["crc32"]:
        raise ValueError("ZIP CRC or image size mismatch")
    with Image.open(io.BytesIO(data)) as im:
        width, height = im.size
        if im.format != "JPEG":
            raise ValueError("Expected an original JPEG")
        im.verify()
    references = []
    for object_index, obj in enumerate(row["source_labels"].get("objects", [])):
        if obj.get("value") not in NAMES:
            continue
        if "polygon" not in obj:
            raise ValueError("Target object lacks a polygon")
        box, clipped = polygon_box(obj["polygon"], width, height)
        references.append({"annotation_id": f"{row['image_id']}_{object_index}",
                           "class_id": NAMES.index(obj["value"]), "class_name": obj["value"],
                           "box": box, "polygon": obj["polygon"], "clipped_to_image": clipped,
                           "classifications": obj.get("classifications", [])})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    result = {"image_id": row["image_id"], "file_name": row["file_name"], "split": row["split"],
              "circuit": row["circuit"], "image_file": str(p.relative_to(OUT)), "width": width, "height": height,
              "sha256": digest(data), "bytes": len(data), "zip_crc32": entry["crc32"],
              "source_archive_url": blob["url"], "source_archive_etag": blob["etag"],
              "zip_member": entry["member"], "raw_annotation_provenance": "selection_plan.json/source_labels",
              "selection_plan_sha256": digest(OUT / "selection_plan.json"), "references": references}
    write_json(receipt, result)
    return result


def make_labels(rows):
    for r in rows:
        p = OUT / "labels" / r["split"] / (r["image_id"] + ".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for ref in r["references"]:
            x1, y1, x2, y2 = ref["box"]
            values = [(x1+x2)/2/r["width"], (y1+y2)/2/r["height"], (x2-x1)/r["width"], (y2-y1)/r["height"]]
            lines.append(str(ref["class_id"]) + " " + " ".join(f"{v:.10f}" for v in values))
        p.write_text("\n".join(lines) + ("\n" if lines else ""))
        r["label_file"] = str(p.relative_to(OUT))
        r["label_sha256"] = digest(p)
    for location, path in (("local", OUT), ("roihu", Path("/scratch/project_2012997/keen_ai/data/external/epri_components_v1"))):
        (OUT / f"train_{location}.yaml").write_text(
            f"path: {path}\ntrain: images/train\nval: images/dev\nnames:\n" +
            "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES)))


def download_all():
    plan = json.loads((OUT / "selection_plan.json").read_text())
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(download_one, plan["images"]):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(json.dumps({"event": "IMAGES_VERIFIED", "count": len(rows), "total": len(plan["images"])}), flush=True)
    hashes = collections.defaultdict(list)
    for r in rows:
        hashes[r["sha256"]].append(r["image_id"])
    duplicates = [v for v in hashes.values() if len(v) > 1]
    if duplicates:
        write_json(OUT / "duplicates.json", duplicates)
        raise ValueError("Exact duplicate images found; resolve before any model run")
    make_labels(rows)
    summary = {s: {"images": sum(r["split"] == s for r in rows),
                   "class_instances": dict(collections.Counter(ref["class_name"] for r in rows if r["split"] == s for ref in r["references"])),
                   "negative_images": sum(r["split"] == s and not r["references"] for r in rows)} for s in SPLITS}
    manifest = {k: v for k, v in plan.items() if k not in ("images", "unlabelled_files_excluded", "status")}
    manifest.update(status="VERIFIED_ORIGINAL_IMAGES_AND_PUBLISHER_POLYGONS", selection_plan_sha256=digest(OUT/"selection_plan.json"),
                    image_count=len(rows), original_bytes=sum(r["bytes"] for r in rows), summary=summary,
                    source_archive_whole_sha256_verified=False, integrity="Pinned Azure ETag + exact Content-Range + member CRC32 + local SHA256",
                    images=rows)
    write_json(OUT / "manifest.json", manifest)
    verify()


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    if manifest["selection_plan_sha256"] != digest(OUT / "selection_plan.json"):
        raise ValueError("Frozen selection plan changed")
    groups, hashes = {}, set()
    for row in manifest["images"]:
        if digest(OUT / row["image_file"]) != row["sha256"] or digest(OUT / row["label_file"]) != row["label_sha256"]:
            raise ValueError("Image or derived YOLO label changed")
        if row["sha256"] in hashes:
            raise ValueError("Duplicate image")
        hashes.add(row["sha256"])
        if row["circuit"] in groups and groups[row["circuit"]] != row["split"]:
            raise ValueError("Circuit leakage")
        groups[row["circuit"]] = row["split"]
        lines = (OUT / row["label_file"]).read_text().splitlines()
        if len(lines) != len(row["references"]):
            raise ValueError("Label count mismatch")
        if len(lines) != len(set(lines)):
            raise ValueError("Duplicate training boxes need annotation review")
        for line, ref in zip(lines, row["references"]):
            values = list(map(float, line.split()))
            x1,y1,x2,y2 = ref["box"]
            expected = [ref["class_id"], (x1+x2)/2/row["width"], (y1+y2)/2/row["height"],
                        (x2-x1)/row["width"], (y2-y1)/row["height"]]
            if len(values) != 5 or any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-9) for a,b in zip(values,expected)):
                raise ValueError("Derived YOLO labels do not reconstruct the publisher polygons")
    print(json.dumps({"event": "DATASET_VERIFIED", "manifest_sha256": digest(OUT/"manifest.json"),
                      "images": len(manifest["images"]), "summary": manifest["summary"]}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["index", "plan", "download", "verify"])
    args = parser.parse_args()
    {"index": index_archives, "plan": make_plan, "download": download_all, "verify": verify}[args.stage]()
