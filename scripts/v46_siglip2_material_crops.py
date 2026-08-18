from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v23_common import REPORTS, ROOT, runtime_env, sha256, write_json

PROTOCOL = ROOT / "configs/v46_siglip2_material_crop_protocol.json"
ANNOTATIONS = ROOT / "data/v4_annotations/material_reference_v44.json"
REF_ID = "POS_8090535"
DEV_ID = "POS_2952166"
CLASSES = ["glass", "ceramic_family"]


def load_json(path: Path):
    return json.loads(path.read_text())


def rec(data, rid):
    return next(r for r in data["records"] if r["record_id"] == rid)


def image_path(rid):
    p = ROOT / f"reports/v4_4_angle_material_candidates/{rid}.jpg"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def crop_box(image, box, frac, min_pad):
    x1,y1,x2,y2=map(float,box);w=max(1.0,x2-x1);h=max(1.0,y2-y1)
    px=max(float(min_pad),w*frac);py=max(float(min_pad),h*frac)
    X1=max(0,int(math.floor(x1-px)));Y1=max(0,int(math.floor(y1-py)));X2=min(image.width,int(math.ceil(x2+px)));Y2=min(image.height,int(math.ceil(y2+py)))
    return image.crop((X1,Y1,X2,Y2)),[X1,Y1,X2,Y2]


def pooled(output):
    # Transformers 4.57.x SigLIP2 get_text_features()/get_image_features() return
    # a pooled torch.Tensor directly. Older/other compatible APIs may expose
    # BaseModelOutput-like or tuple outputs, so keep those fallbacks too.
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output,"pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output,(tuple,list)) and len(output)>1:
        return output[1]
    raise RuntimeError(f"Cannot find pooled embedding in {type(output)}")


def image_features(model, processor, images):
    inputs=processor(images=images,return_tensors="pt")
    allowed={k:v for k,v in inputs.items() if k in {"pixel_values","pixel_attention_mask","spatial_shapes"}}
    with torch.no_grad(): out=model.get_image_features(**allowed)
    return F.normalize(pooled(out).float(),dim=-1)


def text_features(model, processor, texts):
    inputs=processor(text=texts,padding="max_length",return_tensors="pt")
    allowed={k:v for k,v in inputs.items() if k in {"input_ids","attention_mask","position_ids"}}
    with torch.no_grad(): out=model.get_text_features(**allowed)
    return F.normalize(pooled(out).float(),dim=-1)


def normalised_mean(x):
    return F.normalize(x.mean(dim=0,keepdim=True),dim=-1)[0]


def class_prototypes_from_text(model,processor,prompts):
    protos=[]
    detail={}
    for name in CLASSES:
        feats=text_features(model,processor,prompts[name]);proto=normalised_mean(feats);protos.append(proto);detail[name]={"n_prompts":len(prompts[name])}
    return torch.stack(protos),detail


def class_prototypes_from_images(model,processor,ref_crops,ref_labels):
    feats=image_features(model,processor,ref_crops);protos=[];detail={}
    for name in CLASSES:
        idx=[i for i,y in enumerate(ref_labels) if y==name]
        proto=normalised_mean(feats[idx]);protos.append(proto);detail[name]={"n_reference_crops":len(idx)}
    return torch.stack(protos),detail,feats


def z2(scores):
    # Row-wise z normalisation. With two classes this standardises modality scale without training parameters.
    mean=scores.mean(dim=1,keepdim=True);std=scores.std(dim=1,keepdim=True,unbiased=False).clamp_min(1e-8)
    return (scores-mean)/std


def softmax_relative(scores):
    return torch.softmax(scores,dim=-1)


def evaluate(score_matrix,labels,ids,method):
    pred=score_matrix.argmax(dim=1).cpu().tolist();rel=softmax_relative(score_matrix).cpu().tolist();rows=[]
    for i,(p,r) in enumerate(zip(pred,rel)):
        rows.append({"id":ids[i],"true_label":labels[i],"predicted_label":CLASSES[p],"correct":CLASSES[p]==labels[i],"relative_score":float(r[p]),"class_scores":{CLASSES[j]:float(score_matrix[i,j]) for j in range(2)},"relative_scores":{CLASSES[j]:float(r[j]) for j in range(2)}})
    accuracy=sum(x["correct"] for x in rows)/len(rows)
    recalls={}
    cm={a:{b:0 for b in CLASSES} for a in CLASSES}
    for x in rows:cm[x["true_label"]][x["predicted_label"]]+=1
    for c in CLASSES:
        members=[x for x in rows if x["true_label"]==c];recalls[c]=sum(x["correct"] for x in members)/len(members)
    return {"method":method,"accuracy":accuracy,"balanced_accuracy":sum(recalls.values())/len(recalls),"per_class_recall":recalls,"confusion_matrix":cm,"predictions":rows}


def render(dev_record, eval_result, output):
    from PIL import Image,ImageDraw,ImageFont
    image=Image.open(image_path(DEV_ID)).convert("RGB");draw=ImageDraw.Draw(image);font=ImageFont.load_default()
    draw.rectangle((0,0,image.width,22),fill=(255,255,255));draw.text((5,5),"MANUAL BOX | SIGLIP2 MATERIAL | RELATIVE UNCALIBRATED SCORE",fill=(0,0,0),font=font)
    by_id={x["id"]:x for x in eval_result["predictions"]}
    for b in dev_record["boxes"]:
        row=by_id[b["id"]];xy=tuple(int(x) for x in b["xyxy"]);color=(30,190,95) if row["predicted_label"]=="glass" else (190,95,35)
        draw.rectangle(xy,outline=color,width=3);display="glass" if row["predicted_label"]=="glass" else "ceramic"
        draw.text((xy[0],max(24,xy[1]-12)),f"{display} {row['relative_score']*100:.1f}%",fill=color,font=font)
    image.save(output,quality=95)


def main():
    from PIL import Image
    from transformers import AutoModel,AutoProcessor

    protocol=load_json(PROTOCOL);annotations=load_json(ANNOTATIONS);ref=rec(annotations,REF_ID);dev=rec(annotations,DEV_ID)
    if protocol["adaptive_development_source"]!=DEV_ID:raise RuntimeError("Protocol/source mismatch")
    if sha256(image_path(REF_ID))!=ref["source_sha256"] or sha256(image_path(DEV_ID))!=dev["source_sha256"]:raise RuntimeError("Source image hash mismatch")
    frac=float(protocol["crop"]["box_padding_fraction_each_side"]);min_pad=int(protocol["crop"]["minimum_padding_px"])
    ref_image=Image.open(image_path(REF_ID)).convert("RGB");dev_image=Image.open(image_path(DEV_ID)).convert("RGB")
    ref_crops=[];ref_labels=[];ref_ids=[];ref_crop_boxes=[]
    for b in ref["boxes"]:
        crop,cb=crop_box(ref_image,b["xyxy"],frac,min_pad);ref_crops.append(crop);ref_labels.append(b["material_task_label"]);ref_ids.append(b["id"]);ref_crop_boxes.append(cb)
    dev_crops=[];dev_labels=[];dev_ids=[];dev_crop_boxes=[]
    for b in dev["boxes"]:
        crop,cb=crop_box(dev_image,b["xyxy"],frac,min_pad);dev_crops.append(crop);dev_labels.append(b["material_task_label"]);dev_ids.append(b["id"]);dev_crop_boxes.append(cb)

    model_id=protocol["model"]["id"]
    processor=AutoProcessor.from_pretrained(model_id)
    model=AutoModel.from_pretrained(model_id)
    model.eval()

    text_proto,text_meta=class_prototypes_from_text(model,processor,protocol["zero_shot_prompt_ensemble"])
    image_proto,image_meta,ref_feats=class_prototypes_from_images(model,processor,ref_crops,ref_labels)
    dev_feats=image_features(model,processor,dev_crops)
    text_scores=dev_feats@text_proto.T
    image_scores=dev_feats@image_proto.T
    hybrid_scores=(z2(text_scores)+z2(image_scores))/2.0
    results={
        "siglip2_text_prompt_prototype":evaluate(text_scores,dev_labels,dev_ids,"siglip2_text_prompt_prototype"),
        "siglip2_image_reference_prototype":evaluate(image_scores,dev_labels,dev_ids,"siglip2_image_reference_prototype"),
        "siglip2_equal_weight_text_image_hybrid":evaluate(hybrid_scores,dev_labels,dev_ids,"siglip2_equal_weight_text_image_hybrid"),
    }
    for name,res in results.items():
        render(dev,res,REPORTS/f"v4_6_{name}_{DEV_ID}.jpg")
    champion=max(results.values(),key=lambda x:(x["balanced_accuracy"],x["accuracy"],x["method"]=="siglip2_equal_weight_text_image_hybrid"))
    report={
        "version":"v4.6-siglip2-material-crop-development",
        "evidence_type":"oracle-box cross-source material crop classification development experiment",
        "evaluation_semantics":protocol["evaluation_semantics"],
        "model_id":model_id,
        "protocol":{"path":str(PROTOCOL.relative_to(ROOT)),"sha256":sha256(PROTOCOL)},
        "annotation_sha256":sha256(ANNOTATIONS),
        "reference":{"record_id":REF_ID,"n_crops":len(ref_crops),"labels":ref_labels,"expanded_crop_boxes":ref_crop_boxes,"prototype_meta":image_meta},
        "adaptive_development":{"record_id":DEV_ID,"n_crops":len(dev_crops),"labels":dev_labels,"expanded_crop_boxes":dev_crop_boxes},
        "text_meta":text_meta,
        "methods":results,
        "development_selected_champion":champion["method"],
        "champion_selection_is_not_independent_evidence":True,
        "score_semantics":"cosine-derived two-class relative score; display score is softmax-normalised for readability and is not calibrated probability",
        "localisation_semantics":"oracle/manual component boxes only; v4.6 does not claim automatic localisation",
        "claim_boundary":protocol["claim_boundary"],
        "runtime":runtime_env(),"retired_v3_8_holdout_used":False,
    }
    write_json(REPORTS/"v4_6_siglip2_material_crop_results.json",report)
    print(json.dumps({"methods":{k:{"accuracy":v["accuracy"],"balanced_accuracy":v["balanced_accuracy"],"confusion_matrix":v["confusion_matrix"]} for k,v in results.items()},"development_selected_champion":champion["method"]},indent=2))


if __name__=="__main__":main()
