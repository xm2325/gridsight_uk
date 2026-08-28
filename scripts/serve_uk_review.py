#!/usr/bin/env python3
"""Loopback-only review server: immutable proposals and separately versioned drafts."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from functools import partial
import hmac
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit,parse_qs

from prepare_keen_components import ROOT,write_json
from uk_review_common import validate_review_image,validate_review_v3


class RevisionConflict(ValueError):pass


class DraftStore:
    def __init__(self,folder,sources,validator=validate_review_image,schema='gridsight-uk-review-v2'):
        self.folder=Path(folder);self.sources={r["image_id"]:r for r in sources};self.lock=threading.RLock()
        self.validator=validator;self.schema=schema
        self.path=self.folder/"draft.json"

    def load(self):
        with self.lock:
            if self.path.exists():
                data=json.loads(self.path.read_text())
                if data.get("training_approved") is not False or data.get("expert_validated") is not False:
                    raise ValueError("Unreviewed draft cannot claim training approval or expert validation")
                if set(data["images"])!=set(self.sources):raise ValueError("Draft source set changed")
                if data['schema']!=self.schema:raise ValueError('Draft schema changed; explicit migration required')
                for key,image in data["images"].items():self.validator(image,self.sources[key])
                return data
            return {"schema":self.schema,"training_approved":False,"expert_validated":False,
                    "images":{key:{"image_sha256":r["sha256"],"status":"unreviewed","reviewer":"","notes":"",
                                   "objects":[],"revision":0} for key,r in self.sources.items()}}

    def save_image(self,image_id,value,revision):
        if image_id not in self.sources:raise ValueError("Unknown image")
        self.validator(value,self.sources[image_id])
        with self.lock:
            data=self.load();current=data["images"][image_id]
            if type(revision) is not int or revision!=current["revision"]:raise RevisionConflict("Newer edit exists; reload before saving")
            saved=deepcopy(value);saved["revision"]=revision+1;saved["saved_at"]=datetime.now(timezone.utc).isoformat()
            data["images"][image_id]=saved
            self.folder.mkdir(parents=True,exist_ok=True)
            with (self.folder/"events.jsonl").open("a") as f:
                f.write(json.dumps({"event":"DRAFT_SAVED_NOT_TRAINING_APPROVED","image_id":image_id,"draft":saved},ensure_ascii=False)+"\n")
            write_json(self.path,data)
            return saved

    def export(self):
        data=self.load();data["exported_at"]=datetime.now(timezone.utc).isoformat()
        if self.schema=='gridsight-uk-review-v3':
            data['class_names']=['pole','crossarm','insulator','steelwork','pole-top']
            data['region_class_ids']=[4]
            data['steelwork_status']='Draft structural-metal hypothesis; composition needs evidence'
        data["sources"]={key:{k:r[k] for k in ["image_id","sha256","width","height","source_page","credit","license","license_url"]}
                         for key,r in self.sources.items()}
        data["warning"]="Editable draft annotations only. Ready for second review does not authorize training or establish expert ground truth."
        return data


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self,*args,root,stores,token,**kwargs):
        self.stores=stores;self.token=token
        super().__init__(*args,directory=str(root),**kwargs)

    def api_json(self,value,status=200,download=False):
        body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
        if download:self.send_header("Content-Disposition",'attachment; filename="gridsight_uk_review_draft.json"')
        self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)

    def store(self):
        qa=parse_qs(urlsplit(self.path).query).get("qa")==["1"]
        return self.stores["qa" if qa else "main"],qa

    def do_GET(self):
        path=urlsplit(self.path).path
        if path in ["/api/state","/api/export"]:
            store,qa=self.store()
            data=store.export() if path=="/api/export" else store.load()
            self.api_json({"data":data,"qa_mode":qa,"csrf_token":self.token} if path=="/api/state" else data,
                          download=path=="/api/export")
        else:super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path!="/api/save":self.api_json({"error":"Unknown endpoint"},404);return
        expected=f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Origin")!=expected or not hmac.compare_digest(self.headers.get("X-Review-Token",""),self.token):
            self.api_json({"error":"Same-origin review request required"},403);return
        try:
            if self.headers.get("Content-Type","").split(";")[0]!="application/json":raise ValueError("JSON required")
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=2_000_000:raise ValueError("Invalid request size")
            request=json.loads(self.rfile.read(length))
            if not isinstance(request,dict):raise ValueError("Object required")
            store,_=self.store()
            saved=store.save_image(request["image_id"],request["draft"],request["revision"])
            self.api_json({"saved":saved})
        except RevisionConflict as exc:self.api_json({"error":str(exc)},409)
        except (ValueError,KeyError,TypeError) as exc:self.api_json({"error":str(exc)},400)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=8772)
    p.add_argument("--run",type=Path,default=ROOT/"runs/uk_component_review/v2_20260827")
    args=p.parse_args()
    bundle=json.loads((args.run/"report/data.json").read_text())
    v3=bundle['schema']=='gridsight-uk-review-v3';version='v3' if v3 else 'v2'
    validator=validate_review_v3 if v3 else validate_review_image
    stores={"main":DraftStore(ROOT/f"data/annotations/uk_component_review_{version}",bundle["images"],validator,bundle['schema']),
            "qa":DraftStore(ROOT/f"runtime/review_qa_{version}",bundle["images"],validator,bundle['schema'])}
    handler=partial(ReviewHandler,root=args.run,stores=stores,token=secrets.token_urlsafe(32))
    with ThreadingHTTPServer(("127.0.0.1",args.port),handler) as server:
        print(f"Review: http://127.0.0.1:{args.port}/report/index.html (loopback only)",flush=True)
        server.serve_forever()


if __name__=="__main__":main()
