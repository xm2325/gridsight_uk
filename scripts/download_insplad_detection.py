#!/usr/bin/env python3
"""Retrieve the detection member of the official InsPLAD release, checking CRC.

The publisher nests its detection ZIP in a deflated ZIP64 archive, so individual
photographs cannot be fetched by byte range. Only the 4.36 GB detection member is
transferred, not the unrelated fault datasets. No images are extracted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://data.mendeley.com/datasets/5n3fjgvfyz/1"
DOWNLOAD = "https://data.mendeley.com/public-files/datasets/5n3fjgvfyz/files/96707044-99bb-40b2-bf23-6fa1b41ab9b0/file_downloaded"
# Observed public redirect; not an authenticated or signed URL.
OBJECT_URL = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4665fbe5-596f-4d1a-98cc-8a856b6e256e"
TOTAL_ARCHIVE_BYTES = 6400493732
DATA_START = 97
COMPRESSED_BYTES = 4361848806
DECOMPRESSED_BYTES = 4361421001
EXPECTED_CRC32 = 277744259


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/external/insplad_cache/InsPLAD-det.zip")
    args = parser.parse_args()
    destination = args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace cached archive: {destination}")
    temporary = destination.with_suffix(".zip.part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete download exists; inspect before retrying: {temporary}")
    end = DATA_START + COMPRESSED_BYTES - 1
    request = urllib.request.Request(OBJECT_URL, headers={
        "Range": f"bytes={DATA_START}-{end}",
        "User-Agent": "GridSight-UK/InsPLAD-research-evaluation",
        "Accept-Encoding": "identity",
    })
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    sha = hashlib.sha256()
    crc = compressed = expanded = 0
    started = last_log = time.monotonic()
    with urllib.request.urlopen(request, timeout=90) as response:
        expected_range = f"bytes {DATA_START}-{end}/{TOTAL_ARCHIVE_BYTES}"
        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
            raise RuntimeError("Server did not honor the exact, bounded archive-member range")
        response_meta = {key: response.headers.get(key) for key in ("ETag", "Last-Modified", "Content-Range")}
        with temporary.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                compressed += len(chunk)
                if compressed > COMPRESSED_BYTES:
                    raise RuntimeError("Download exceeded the advertised member length")
                data = inflater.decompress(chunk)
                output.write(data)
                expanded += len(data)
                sha.update(data)
                crc = zlib.crc32(data, crc)
                if expanded > DECOMPRESSED_BYTES:
                    raise RuntimeError("Expanded member exceeded expected size")
                if time.monotonic() - last_log >= 20:
                    print(json.dumps({"downloaded_bytes": compressed, "total_bytes": COMPRESSED_BYTES,
                                      "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                    last_log = time.monotonic()
            tail = inflater.flush()
            output.write(tail)
            expanded += len(tail)
            sha.update(tail)
            crc = zlib.crc32(tail, crc)
    if (compressed, expanded, crc) != (COMPRESSED_BYTES, DECOMPRESSED_BYTES, EXPECTED_CRC32):
        raise RuntimeError(f"Archive member verification failed: {compressed}, {expanded}, {crc}")
    if not inflater.eof or inflater.unused_data:
        raise RuntimeError("Unexpected deflate stream boundary")
    with zipfile.ZipFile(temporary) as archive:
        members = len(archive.infolist())
    temporary.replace(destination)
    metadata = {"source_page": SOURCE, "official_download_url": DOWNLOAD, "object_url": OBJECT_URL,
                "license_from_publisher": "CC BY-NC 3.0", "usage": "non-commercial research evaluation",
                "member": "InsPLAD-det.zip", "sha256": sha.hexdigest(), "crc32": crc,
                "bytes": expanded, "compressed_bytes_transferred": compressed, "zip_members": members,
                "response": response_meta, "whole_outer_archive_sha256_verified": False,
                "verification": "ZIP member CRC32 and size verified against official ZIP64 central directory",
                "elapsed_seconds": round(time.monotonic() - started, 2)}
    destination.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "VERIFIED", **metadata}), flush=True)


if __name__ == "__main__":
    main()
