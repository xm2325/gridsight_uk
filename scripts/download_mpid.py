"""Resumable, hash-checked MPID v1 downloader for the Roihu data cache."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/external/mpid_v1'
RECORD='https://zenodo.org/records/14604384'
FILES=[
 ('composite_MPID-20250106T113900Z-001.zip','7f973026e3f1ff4c5843a668900a039f'),
 ('glass_MPID-20250106T113616Z-001.zip','240282587adee484a680521aba4b8f05'),
 ('porcelain_MPID-20250106T111149Z-001.zip','650a8ed4f55168150a8cf6c890983581'),
]
UA='GridSight-UK-MPID-audit/1.0 (+https://github.com/xm2325/gridsight_uk)'

def hashes(path:Path)->tuple[str,str]:
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):md5.update(block);sha.update(block)
    return md5.hexdigest(),sha.hexdigest()

def download(name:str,expected_md5:str)->dict:
    target=OUT/name;part=target.with_suffix(target.suffix+'.part')
    if target.exists():
        md5,sha=hashes(target)
        if md5!=expected_md5:raise RuntimeError(f'Existing file MD5 mismatch: {name}')
        return {'name':name,'bytes':target.stat().st_size,'md5':md5,'sha256':sha,'status':'reused_verified'}
    url=f'{RECORD}/files/{name}?download=1'
    for attempt in range(6):
        offset=part.stat().st_size if part.exists() else 0
        headers={'User-Agent':UA};mode='wb'
        if offset:headers['Range']=f'bytes={offset}-';mode='ab'
        try:
            request=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(request,timeout=120) as response:
                if offset and response.status!=206:
                    part.unlink();offset=0;mode='wb'
                with part.open(mode) as output:
                    while True:
                        block=response.read(8*1024*1024)
                        if not block:break
                        output.write(block)
            md5,sha=hashes(part)
            if md5!=expected_md5:raise RuntimeError(f'Downloaded MD5 mismatch: {name}: {md5}')
            part.replace(target)
            return {'name':name,'bytes':target.stat().st_size,'md5':md5,'sha256':sha,'status':'downloaded_verified'}
        except Exception:
            if attempt==5:raise
            time.sleep(min(30,2**attempt))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,md5 in FILES:
        print(json.dumps({'event':'download_start','name':name}),flush=True)
        row=download(name,md5);rows.append(row);print(json.dumps({'event':'download_verified',**row}),flush=True)
    manifest={'version':'MPID-v1-Zenodo-14604384','record_url':RECORD,'doi':'10.5281/zenodo.14604384',
      'publisher_license':'CC BY 4.0','retrieved_at':datetime.now(timezone.utc).isoformat(),'files':rows,
      'archive_bytes':sum(row['bytes'] for row in rows),'extracted':False,
      'training_authorized':False,'next_gate':'inventory labels, units, upstream groups and duplicates before a bounded protocol'}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'event':'mpid_download_complete','archive_bytes':manifest['archive_bytes']}),flush=True)

if __name__=='__main__':main()
