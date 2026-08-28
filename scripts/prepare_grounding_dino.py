#!/usr/bin/env python3
"""Fetch a pinned public safetensors release, verifying publisher file hashes."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from prepare_keen_components import ROOT, digest, write_json

MODEL = "IDEA-Research/grounding-dino-base"
REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
OUT = ROOT / "weights/grounding-dino-base-12bdfa3"


def verify_file(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Wrong byte count: {path.name}")
    if "lfs" in entry:
        if digest(path) != entry["lfs"]["sha256"]:
            raise ValueError(f"Wrong publisher SHA256: {path.name}")
    else:
        payload=path.read_bytes()
        if hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()!=entry["blobId"]:
            raise ValueError(f"Wrong publisher Git blob hash: {path.name}")


def download_release(model,revision,out,meta_path):
    meta=json.loads(meta_path.read_text())
    if meta["sha"]!=revision or meta["id"]!=model or meta["private"] or meta["gated"]:
        raise ValueError("Model release or public access status differs")
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for entry in meta["siblings"]:
        name=entry["rfilename"]
        if name in [".gitattributes","pytorch_model.bin"]:continue
        if Path(name).name!=name:raise ValueError("Unexpected nested release path")
        target=out/name
        url=f"https://huggingface.co/{model}/resolve/{revision}/{name}"
        if not target.exists():
            part=target.with_suffix(target.suffix+".partial")
            for attempt in range(4):
                try:
                    offset=part.stat().st_size if part.exists() else 0
                    if offset==entry["size"]:break
                    if offset>entry["size"]:raise ValueError("Oversized partial download")
                    headers={"User-Agent":"GridSight-UK-research/1.0","Accept-Encoding":"identity"}
                    if offset:headers["Range"]=f"bytes={offset}-"
                    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                        if offset and response.status==206:
                            expected=f"bytes {offset}-{entry['size']-1}/{entry['size']}"
                            if response.headers.get("Content-Range")!=expected:raise ValueError("Unexpected resumed range")
                            mode="ab"
                        elif response.status==200:
                            mode="wb"
                        else:raise ValueError("Unexpected download status")
                        with part.open(mode) as f:
                            while chunk:=response.read(1024*1024):f.write(chunk)
                    verify_file(part,entry)
                    break
                except Exception:
                    if attempt==3:raise
                    time.sleep(2**attempt)
            verify_file(part,entry);part.replace(target)
        verify_file(target,entry)
        records.append({"file":name,"sha256":digest(target),"bytes":target.stat().st_size,"source":url,
                        "publisher_lfs_sha256":entry.get("lfs",{}).get("sha256"),"publisher_blob_id":entry["blobId"]})
        print(json.dumps({"event":"MODEL_FILE_VERIFIED","file":name,"bytes":target.stat().st_size}),flush=True)
    write_json(out/"verified_manifest.json",{"model_id":model,"revision":revision,"metadata_sha256":digest(meta_path),
        "license":"Apache-2.0","format":"safetensors only; no remote code","files":records})


def main():
    download_release(MODEL,REVISION,OUT,ROOT/"runtime/target_sources/grounding_dino_base_api.json")


if __name__=="__main__":main()
