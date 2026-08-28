"""Pure geometry and review validation helpers; proposals never become gold labels."""
import math
from roihu_demo_ablation import nms, iou


def pole_regions(predictions,width,height,cfg):
    regions=[]
    poles=[p for p in nms(predictions) if p["class_id"]==0 and p["score"]>=cfg["pole_score"]
           and p["box"][3]-p["box"][1]>=cfg["min_pole_height"]]
    for p in poles:
        x1,y1,x2,y2=p["box"];pw=x2-x1;ph=y2-y1
        side=max(cfg["min_side"],cfg["side_height_fraction"]*ph,cfg["side_width_factor"]*pw)
        cx=(x1+x2)/2;top=y1-cfg["above_pole_fraction"]*ph
        region=[max(0,int(math.floor(cx-side/2))),max(0,int(math.floor(top))),
                min(width,int(math.ceil(cx+side/2))),min(height,int(math.ceil(top+side)))]
        if region==[0,0,width,height]:continue
        if region[2]-region[0]<16 or region[3]-region[1]<16:continue
        if any(iou(region,r["box"])>cfg["duplicates_iou"] for r in regions):continue
        regions.append({"box":region,"source_pole_box":p["box"],"source_pole_score":p["score"],
                        "manual":False,"purpose":"component search ROI; not a pole-top detection"})
        if len(regions)==cfg["max_poles"]:break
    return regions


def map_crop_predictions(predictions,region,width,height):
    out=[]
    for p in predictions:
        b=p["box"];x,y=region[:2]
        box=[max(0.,min(float(width),b[0]+x)),max(0.,min(float(height),b[1]+y)),
             max(0.,min(float(width),b[2]+x)),max(0.,min(float(height),b[3]+y))]
        if box[2]>box[0] and box[3]>box[1]:out.append({**p,"box":box})
    return out


def validate_review_image(value,source,class_count=3):
    """Fail closed on invalid draft geometry, provenance or unsupported claims."""
    if not isinstance(value,dict) or value.get("image_sha256")!=source["sha256"]:raise ValueError("Image fingerprint mismatch")
    if value.get("training_approved",False) or value.get("expert_validated",False):raise ValueError("Draft cannot approve training or expert validation")
    if value.get("status") not in ["unreviewed","draft","ready_for_second_review"]:raise ValueError("Unsupported review state")
    reviewer=value.get("reviewer","")
    if not isinstance(reviewer,str) or len(reviewer)>120:raise ValueError("Invalid reviewer")
    if value["status"]=="ready_for_second_review" and not reviewer.strip():raise ValueError("Reviewer required")
    if not isinstance(value.get("notes",""),str) or len(value.get("notes",""))>4000:raise ValueError("Invalid notes")
    objects=value.get("objects")
    if not isinstance(objects,list) or len(objects)>200:raise ValueError("Invalid object list")
    ids=set()
    for o in objects:
        if not isinstance(o,dict) or not isinstance(o.get("id"),str) or len(o["id"])>120 or o["id"] in ids:raise ValueError("Invalid object ID")
        ids.add(o["id"])
        if type(o.get("class_id")) is not int or o["class_id"] not in range(class_count):raise ValueError("Invalid component class")
        b=o.get("box")
        if not isinstance(b,list) or len(b)!=4 or not all(type(n) in (int,float) and math.isfinite(n) for n in b):raise ValueError("Invalid box numbers")
        if not (0<=b[0]<b[2]<=source["width"] and 0<=b[1]<b[3]<=source["height"]):raise ValueError("Box outside original image")
        material=o.get("material")
        if material not in [None,"unknown","glass","porcelain","polymer"]:raise ValueError("Unsupported material")
        if o["class_id"]!=2 and material is not None:raise ValueError("Material schema applies only to insulators")
        evidence=o.get("material_evidence","")
        if not isinstance(evidence,str) or len(evidence)>1000:raise ValueError("Invalid material evidence")
        if material in ["glass","porcelain","polymer"] and not evidence.strip():raise ValueError("Material evidence required; use unknown otherwise")
        origins=["manual_draft","machine_proposal"]+(["derived_geometry"] if class_count==5 else [])
        if o.get("origin") not in origins:raise ValueError("Unknown annotation origin")
        if o.get("training_approved",False) or o.get("expert_validated",False):raise ValueError("This tool cannot approve training or expert ground truth")
    return value


def validate_review_v3(value,source):
    validate_review_image(value,source,class_count=5)
    by_id={o['id']:o for o in value['objects']}
    for o in value['objects']:
        expected='inspection_region' if o['class_id']==4 else 'component'
        if o.get('entity_kind')!=expected:raise ValueError('Component and inspection-region semantics must remain separate')
        if o['class_id']==4:
            if o.get('origin')=='machine_proposal' or any(k in o for k in ['score','confidence','proposal_source']):raise ValueError('Inspection regions cannot carry detector confidence')
        evidence=o.get('steelwork_evidence','')
        if not isinstance(evidence,str) or len(evidence)>1000:raise ValueError('Invalid steelwork evidence')
        if o['class_id']==3 and value['status']=='ready_for_second_review' and not evidence.strip():raise ValueError('Steelwork requires material and assembly evidence before second review')
        if o.get('origin')=='derived_geometry' and o['class_id']!=4:raise ValueError('Derived geometry is only an inspection region')
        parent=o.get('parent_pole_id')
        if parent is not None and (parent not in by_id or by_id[parent]['class_id']!=0 or parent==o['id']):raise ValueError('Parent must be a different pole draft in this image')
        for name in ['occluded','truncated']:
            if name in o and type(o[name]) is not bool:raise ValueError('Visibility flags must be boolean')
    return value
