"""Prepare a bounded single-class EPRI+UK insulator adaptation dataset.

UK boxes are analyst visible-object references on already-consumed development
images. They are not expert-reviewed labels and never enter a future acceptance
set. The untouched v2 acceptance cohort is only hash-pinned here, not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/"data/external/uk_distribution_pilot_v1"
EPRI=ROOT/"data/external/epri_components_v1"
OUT=ROOT/"data/external/uk_insulator_adaptation_v1"
PILOT_SHA="fcb4c41d8379bc3c4eab0e7e6c7a099af1d2ec6b533bcc5953d15816ed59d171"
EPRI_SHA="56e0517fcbf864f6c60aa1e2b0869cf9061138a32eb2b2acd40ad37efcb8cffa"
ACCEPTANCE=ROOT/"data/external/uk_insulator_localisation_v2/manifest.json"
ACCEPTANCE_SHA="2fde93a4332e4499cb047a4a684808c798b2e13c387375fcb6ef98395697ffdf"

POSITIVES={
 "7106830":{"split":"train","boxes":[[245,125,320,235],[505,220,580,325],[765,345,840,455]]},
 "5321865":{"split":"train","boxes":[[455,106,491,139],[444,134,488,165],[500,149,538,181],[535,162,571,193],[578,128,616,161]]},
 "2485029":{"split":"train","boxes":[[1135,48,1185,105],[1190,210,1240,260],[1235,335,1290,390]]},
 "3996644":{"split":"train","boxes":[[360,168,405,205],[493,158,540,195]]},
 "6414068":{"split":"dev","boxes":[[755,105,792,138],[846,50,884,82],[908,57,947,88]]},
 "6337870":{"split":"dev","boxes":[[748,145,782,176],[822,127,857,161]]},
 "5722811":{"split":"dev","boxes":[[965,312,1010,360],[1040,386,1095,432],[1138,414,1198,462]]},
}
NEGATIVES={"1802882":"train","2032831":"train","4279330":"dev"}


def sha(path):
 value=hashlib.sha256()
 with open(path,"rb") as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b""):value.update(block)
 return value.hexdigest()


def stable_key(seed,value):return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def yolo_lines(boxes,width,height):
 lines=[]
 for x0,y0,x1,y1 in boxes:
  lines.append("0 "+" ".join(f"{v:.10f}" for v in ((x0+x1)/2/width,(y0+y1)/2/height,(x1-x0)/width,(y1-y0)/height)))
 return lines


def square_crop(boxes,width,height,scale):
 x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes);x1=max(b[2] for b in boxes);y1=max(b[3] for b in boxes)
 side=min(max(width,height),max(256,int(max(x1-x0,y1-y0)*scale+.5)))
 cx=(x0+x1)/2;cy=(y0+y1)/2
 left=max(0,min(width-side,int(cx-side/2+.5)));top=max(0,min(height-side,int(cy-side/2+.5)))
 right=min(width,left+side);bottom=min(height,top+side)
 left=max(0,right-side);top=max(0,bottom-side)
 return [left,top,right,bottom]


def verify_definitions():
 if sha(PILOT/"manifest.json")!=PILOT_SHA or sha(EPRI/"manifest.json")!=EPRI_SHA or sha(ACCEPTANCE)!=ACCEPTANCE_SHA:
  raise ValueError("Pinned source or acceptance manifest changed")
 pilot=json.loads((PILOT/"manifest.json").read_text());by_id={r["geograph_id"]:r for r in pilot["images"]}
 unknown=(set(POSITIVES) | set(NEGATIVES)) - set(by_id)
 if unknown:raise ValueError(f"Unknown UK development source: {sorted(unknown)}")
 acceptance=json.loads(ACCEPTANCE.read_text())
 used_hashes={by_id[k]["sha256"] for k in set(POSITIVES)|set(NEGATIVES)}
 acceptance_hashes={r["image_sha256"] for r in acceptance["records"] if r["role"]!="excluded"}
 if used_hashes&acceptance_hashes:raise ValueError("UK development/acceptance pixel leakage")
 for pid,item in POSITIVES.items():
  row=by_id[pid];path=PILOT/row["image_file"]
  if sha(path)!=row["sha256"]:raise ValueError(f"UK image hash mismatch: {pid}")
  for box in item["boxes"]:
   x0,y0,x1,y1=box
   if not(0<=x0<x1<=row["width"] and 0<=y0<y1<=row["height"]):raise ValueError(f"Invalid UK box: {pid} {box}")
 for pid in NEGATIVES:
  row=by_id[pid]
  if sha(PILOT/row["image_file"])!=row["sha256"]:raise ValueError(f"UK negative hash mismatch: {pid}")
 return by_id


def link_or_copy(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 try:os.link(source,target)
 except OSError:shutil.copy2(source,target)


def add_sample(source,split,name,boxes,crop,records,origin,origin_id,origin_sha):
 if sha(source)!=origin_sha:raise ValueError(f"Source image hash mismatch: {origin_id}")
 with Image.open(source) as image:
  if crop:
   x0,y0,x1,y1=crop;canvas=image.convert("RGB").crop(crop);adjusted=[[b[0]-x0,b[1]-y0,b[2]-x0,b[3]-y0] for b in boxes]
   image_path=OUT/"images"/split/f"{name}.jpg";image_path.parent.mkdir(parents=True,exist_ok=True);canvas.save(image_path,quality=95)
   width,height=canvas.size
  else:
   width,height=image.size;adjusted=boxes;image_path=OUT/"images"/split/f"{name}.jpg";link_or_copy(source,image_path)
 for box in adjusted:
  x0,y0,x1,y1=box
  if not(0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError(f"Prepared box outside image: {name} {box}")
 label_path=OUT/"labels"/split/f"{name}.txt";label_path.parent.mkdir(parents=True,exist_ok=True)
 label_path.write_text("\n".join(yolo_lines(adjusted,width,height))+("\n" if adjusted else ""))
 records.append({"sample_id":name,"split":split,"origin":origin,"origin_id":origin_id,"origin_sha256":origin_sha,
                 "image_file":str(image_path.relative_to(OUT)),"image_sha256":sha(image_path),
                 "label_file":str(label_path.relative_to(OUT)),"label_sha256":sha(label_path),
                 "width":width,"height":height,"boxes":adjusted,"crop_xyxy":crop,
                 "reference_status":"publisher polygon" if origin=="EPRI" else "analyst visible-object box; not expert reviewed"})


def main(verify_only=False):
 by_id=verify_definitions()
 if verify_only:
  print(json.dumps({"status":"DEFINITIONS_VERIFIED","positive_assets":len(POSITIVES),
                    "positive_boxes":sum(len(v["boxes"]) for v in POSITIVES.values()),
                    "negative_assets":len(NEGATIVES),"acceptance_read_for_training":False},indent=2));return
 if OUT.exists():raise FileExistsError(f"Existing prepared dataset: {OUT}")
 records=[];epri=json.loads((EPRI/"manifest.json").read_text())
 for split,positive_n,negative_n in (("train",100,10),("dev",30,5)):
  rows=[r for r in epri["images"] if r["split"]==split]
  positive=[r for r in rows if any(ref["class_name"]=="insulator" for ref in r["references"])]
  negative=[r for r in rows if not any(ref["class_name"]=="insulator" for ref in r["references"])]
  selected=sorted(positive,key=lambda r:stable_key(73,r["image_id"]))[:positive_n]+sorted(negative,key=lambda r:stable_key(79,r["image_id"]))[:negative_n]
  for row in selected:
   boxes=[ref["box"] for ref in row["references"] if ref["class_name"]=="insulator"]
   add_sample(EPRI/row["image_file"],split,row["image_id"],boxes,None,records,"EPRI",row["image_id"],row["sha256"])
 for pid,item in POSITIVES.items():
  row=by_id[pid];source=PILOT/row["image_file"];split=item["split"]
  add_sample(source,split,f"uk_{pid}_full",item["boxes"],None,records,"UK_DEVELOPMENT",pid,row["sha256"])
  seen=set()
  for index,scale in enumerate((1.5,2.5,4.0)):
   crop=square_crop(item["boxes"],row["width"],row["height"],scale)
   if tuple(crop) in seen:continue
   seen.add(tuple(crop));add_sample(source,split,f"uk_{pid}_crop{index+1}",item["boxes"],crop,records,"UK_DEVELOPMENT",pid,row["sha256"])
 for pid,split in NEGATIVES.items():
  row=by_id[pid];add_sample(PILOT/row["image_file"],split,f"uk_{pid}_negative",[],None,records,"UK_DEVELOPMENT",pid,row["sha256"])
 for location,path in (("local",OUT),("roihu",Path("/scratch/project_2012997/keen_ai")/OUT.relative_to(ROOT))):
  (OUT/f"dataset_{location}.yaml").write_text(f"path: {path}\ntrain: images/train\nval: images/dev\nnames:\n  0: insulator\n")
 groups={split:{r["origin_id"] for r in records if r["split"]==split and r["origin"]=="UK_DEVELOPMENT"} for split in ("train","dev")}
 if groups["train"]&groups["dev"]:raise ValueError("UK asset leakage")
 manifest={"version":"uk-insulator-adaptation-v1","selection_frozen_before_training":True,
           "pilot_manifest_sha256":PILOT_SHA,"epri_manifest_sha256":EPRI_SHA,
           "untouched_acceptance_manifest_sha256":ACCEPTANCE_SHA,"acceptance_images_read_for_training":False,
           "uk_reference_status":"analyst visible-object boxes; not expert reviewed",
           "uk_asset_groups":{k:sorted(v) for k,v in groups.items()},"uk_asset_group_overlap":False,
           "records":records,"counts":{split:{"samples":sum(r["split"]==split for r in records),
                                               "boxes":sum(len(r["boxes"]) for r in records if r["split"]==split),
                                               "uk_samples":sum(r["split"]==split and r["origin"]=="UK_DEVELOPMENT" for r in records)}
                                      for split in ("train","dev")},
           "claim_boundary":"Development adaptation data only. Repeated crops are not independent assets; UK references are analyst boxes, not expert truth."}
 (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print(json.dumps({"manifest_sha256":sha(OUT/"manifest.json"),"counts":manifest["counts"],
                   "uk_groups":manifest["uk_asset_groups"]},indent=2))


if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--verify-definitions",action="store_true")
 main(parser.parse_args().verify_definitions)
