#!/usr/bin/env python3
"""Download only pinned official weights and verify GitHub release SHA-256."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    config = json.loads((ROOT / "configs/roihu_benchmark_weights.json").read_text())
    for asset in config["assets"]:
        path = ROOT / asset["name"] if asset["name"].endswith(".ts") else ROOT / "weights" / asset["name"]
        expected_sha = asset["digest"].removeprefix("sha256:")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            part = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(asset["url"], timeout=90) as response, part.open("xb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if part.stat().st_size != asset["size"] or file_sha(part) != expected_sha:
                raise RuntimeError(f"Downloaded weight does not match release metadata: {asset['name']}")
            part.replace(path)
        if path.stat().st_size != asset["size"] or file_sha(path) != expected_sha:
            raise RuntimeError(f"Existing weight does not match release metadata: {path}")
        print(json.dumps({"status": "VERIFIED", "name": asset["name"], "sha256": expected_sha}), flush=True)


if __name__ == "__main__":
    main()
