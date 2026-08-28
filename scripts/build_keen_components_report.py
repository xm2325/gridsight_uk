#!/usr/bin/env python3
"""Verify the completed EPRI experiment and build a local, original-image explorer."""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

from prepare_keen_components import ROOT, digest, write_json
from keen_component_metrics import summarize, match_image, geometric_confusion
from roihu_keen_components import verify_predictions

DEFAULT_RUN = ROOT / "runs/keen_components/epri_components_v1_20260827"
ARMS = ["open_vocabulary", "supervised"]
ARM_LABELS = {"open_vocabulary": "Original open vocabulary", "supervised": "Supervised adaptation"}
COLORS = ["#e5b52b", "#398cf6", "#d650dd"]


def check_completed(report, choices, protocol_sha):
    if report["status"] != "COMPLETED_MULTICOMPONENT_TRAINING_AND_FROZEN_EVALUATION":
        raise ValueError("Cannot report an incomplete run as finished")
    if report["training_progress"]["completed_epochs"] != report["config"]["training"]["epochs"]:
        raise ValueError("Incomplete fixed epoch budget")
    expected = {"protocol_sha256": protocol_sha, "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                "uk_manifest_sha256": report["uk_manifest_sha256"], "checkpoint_sha256": report["selected_checkpoint_sha256"],
                "baseline_checkpoint_sha256": report["config"]["checkpoint_sha256"],
                "prompts": report["config"]["baseline_prompts"], "evaluation": report["config"]["evaluation"],
                "eval_used_for_selection": False, "uk_used_for_selection": False}
    if choices != expected:
        raise ValueError("Frozen evaluation choices changed")


def annotated_image(path, predictions, names, caption, max_size=(1100,800), show_scores=True, score_threshold=.25):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    with Image.open(path) as source:
        w,h = source.size
        photo = ImageOps.contain(source.convert("RGB"), max_size)
    canvas = Image.new("RGB", (photo.width,photo.height+44), "#102135")
    canvas.paste(photo,(0,44))
    draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype("DejaVuSans.ttf",17)
    except OSError:
        font=ImageFont.load_default(size=17)
    draw.text((10,12),caption,fill="white",font=font)
    sx,sy=photo.width/w,photo.height/h
    for p in sorted(predictions,key=lambda p:p["score"]):
        if p["score"] < score_threshold:
            continue
        b=p["box"];b=[b[0]*sx,44+b[1]*sy,b[2]*sx,44+b[3]*sy]
        color=COLORS[p["class_id"]]
        draw.rectangle(b,outline=color,width=3)
        text=f"{names[p['class_id']]}  {p['score']:.2f}" if show_scores else f"{names[p['class_id']]} reference"
        width=draw.textlength(text,font=font)+10
        x=max(0,min(photo.width-width,b[0]));y=max(44,b[1]-23)
        draw.rectangle([x,y,x+width,y+23],fill=color)
        draw.text((x+5,y+2),text,fill="#091526",font=font)
    return canvas


def build_charts(output, bundle, run):
    cache=ROOT/"runtime/report_cache"
    (cache/"matplotlib").mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(cache/"matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME",str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=bundle["meta"]["classes"]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,metric,title in [(axes[0],"ap50","AP@0.50"),(axes[1],"ap50_95","AP@0.50:0.95")]:
        for arm,shift,color in [("open_vocabulary",-.19,"#9aaabd"),("supervised",.19,"#2474d7")]:
            values=[100*bundle["aggregate"]["eval"][arm]["summary"]["ap"][n][metric] for n in names]
            bars=ax.bar([i+shift for i in range(len(names))],values,.36,color=color,label=ARM_LABELS[arm])
            ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=10)
        ax.set_xticks(range(len(names)),names);ax.set_ylim(0,105);ax.set_ylabel("%")
        ax.set_title(title);ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(loc="upper left",fontsize=8)
    fig.suptitle("EPRI circuit-separated evaluation · 100 images · all three component classes",fontsize=13)
    fig.savefig(output/"class_ap.png",dpi=180);plt.close(fig)
    with (run/"training/results.csv").open() as f:
        training=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(f)]
    fig,axes=plt.subplots(1,2,figsize=(12,3.5),constrained_layout=True)
    epochs=[int(float(r["epoch"])) for r in training]
    for key in ["train/box_loss","train/cls_loss"]:
        axes[0].plot(epochs,[float(r[key]) for r in training],label=key)
    for key in ["metrics/mAP50(B)","metrics/mAP50-95(B)"]:
        axes[1].plot(epochs,[100*float(r[key]) for r in training],label=key)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False);ax.legend(fontsize=8);ax.grid(alpha=.15);ax.set_xlabel("Epoch")
    axes[0].set_title("Training losses");axes[1].set_title("Development only · library validation metrics")
    fig.savefig(output/"training.png",dpi=180);plt.close(fig)


def choose_cases(rows):
    def metrics(row,arm):return row["metrics"][arm]
    default=next((r for r in rows if len({x["class_id"] for x in r["references"]})==3),rows[0])
    gained=max(rows,key=lambda r:(metrics(r,"supervised")["tp"]-metrics(r,"open_vocabulary")["tp"],r["image_id"]))
    failure=max(rows,key=lambda r:(metrics(r,"supervised")["fn"],metrics(r,"supervised")["fp"],r["image_id"]))
    regression=min(rows,key=lambda r:(metrics(r,"supervised")["f1"]-metrics(r,"open_vocabulary")["f1"],r["image_id"]))
    return {"default":default["image_id"], "gain":gained["image_id"], "misses":failure["image_id"],
            "regression":regression["image_id"], "uk_closeup":"uk_geograph_7106830",
            "rule":"Default: first frozen eval image with three reference classes. Gain/misses/regression: deterministic outcome-based error audit, not population estimates. UK close-up chosen from source pixels before inference. Every image remains available."}


def output_counts(rows, arm, names, thresholds=(.05,.25,.5)):
    """Counts describe output coverage only; no accuracy without reference labels."""
    result={}
    for threshold in thresholds:
        by_class={name:0 for name in names};nonempty=0
        for row in rows:
            predictions=[p for p in row["predictions"][arm] if p["score"]>=threshold]
            nonempty+=bool(predictions)
            for p in predictions:by_class[names[p["class_id"]]]+=1
        result[f"{threshold:.2f}"]={"images_with_output":nonempty,"images_without_output":len(rows)-nonempty,
                                    "boxes_by_class":by_class,"not_accuracy":True}
    return result


def review_queue(rows, checkpoint_sha):
    """Include empty images so missed objects are not excluded from annotation."""
    tasks=[];materials=[]
    for row in rows:
        proposals=[{"proposal_id":f"{row['image_id']}_{i}","class_id":p["class_id"],"proposed_box":p["box"],
                    "detector_score":p["score"],"source_arm":"supervised","status":"MACHINE_PROPOSAL_NOT_GROUND_TRUTH"}
                   for i,p in enumerate(row["predictions"]["supervised"]) if p["score"]>=.25]
        task={k:row[k] for k in ["image_id","image_file","source_page","credit","license","license_url"]}
        task.update(image_sha256=row["sha256"],status="UNREVIEWED",machine_proposals=proposals,
                    reviewed_objects=[],reviewer=None,reviewed_at=None,
                    instruction="Review the full original image and add missing objects, including when the proposal list is empty")
        tasks.append(task)
        for p in proposals:
            if p["class_id"]==2:
                materials.append({**{k:task[k] for k in ["image_id","image_file","image_sha256","source_page","credit","license","license_url"]},
                                  **p,"is_insulator_reviewed":None,"object_extent_reviewed":False,"material_label":None,
                                  "reviewer":None,"reviewed_at":None})
    return {"status":"UNREVIEWED_ANNOTATION_DEVELOPMENT", "not_ground_truth":True,
            "role":"All images were inspected by models; none is a fresh untouched holdout",
            "source_model_sha256":checkpoint_sha,"proposal_threshold":.25,
            "material_options":["glass","porcelain","polymer","unknown"],
            "review_requirements":["Inspect all original images, including images with no machine output",
                "Add missed objects and remove false proposals", "Correct object extent and annotation unit",
                "Use unknown when pixels do not support a material label", "Record reviewer and source"],
            "image_tasks":tasks,"image_task_count":len(tasks),
            "proposals":materials,"count":len(materials),"count_scope":"Unreviewed insulator proposals only, not image tasks"}


def add_visual_prompt(run, output, bundle, vp_run):
    report=json.loads((vp_run/"results.json").read_text())
    protocol=report["protocol"]
    reference=protocol["reference"]["image_id"]
    if report["status"]!="COMPLETED_QUALITATIVE_VISUAL_PROMPT_DIAGNOSTIC" or report["gradient_steps"]!=0 or report["performance_metrics"] is not None:
        raise ValueError("Visual-prompt report must be a completed qualitative diagnostic")
    if protocol["prior_experiment_results_sha256"]!=digest(run/"results.json") or protocol["uk_manifest_sha256"]!=digest(run/"uk_manifest.json"):
        raise ValueError("Visual-prompt diagnostic has different source experiments")
    for name,sha in report["source_snapshots"].items():
        if digest(vp_run/"code"/name)!=sha:raise ValueError("Visual-prompt code snapshot changed")
    if digest(vp_run/"code/keen_uk_visual_prompt_v1.json")!=report["protocol_sha256"]:
        raise ValueError("Visual-prompt protocol changed")
    choices=json.loads((vp_run/"frozen_choices.json").read_text())
    if digest(vp_run/"frozen_choices.json")!=report["frozen_choices_sha256"] or digest(vp_run/"reference_conditioned_model.pt")!=report["conditioned_model_sha256"]:
        raise ValueError("Visual-prompt frozen choices or conditioned model changed")
    expected_rows=[r for r in json.loads((run/"uk_manifest.json").read_text())["images"] if r["image_id"]!=reference]
    if len(expected_rows)!=26 or json.loads((vp_run/"targets.json").read_text())!=expected_rows:
        raise ValueError("Reference must be excluded and all other UK images retained")
    if choices["target_ids"]!=[r["image_id"] for r in expected_rows] or choices["reference_in_targets"] or choices["target_gt_used"]:
        raise ValueError("Visual reference/target separation changed")
    if choices["reference_boxes"]!=protocol["reference"]["boxes"] or choices["conditioned_model_sha256"]!=report["conditioned_model_sha256"]:
        raise ValueError("Visual-prompt geometry or checkpoint receipt mismatch")
    arm=json.loads((vp_run/"visual_prompt.json").read_text())
    if arm["summary"] is not None or arm["has_ground_truth"]:
        raise ValueError("Cannot assign accuracy to unlabelled UK targets")
    verified=verify_predictions(vp_run,expected_rows,arm["records"],bundle["meta"]["classes"],.001)
    target=output/"visual_prompt"
    target.mkdir(exist_ok=True)
    for name in ["results.json","frozen_choices.json","reference.json","targets.json","visual_prompt.json"]:
        shutil.copyfile(vp_run/name,target/name)
    shutil.copytree(vp_run/"code",target/"code",dirs_exist_ok=True)
    records={r["image_id"]:r for r in arm["records"]}
    rows=[]
    for original in bundle["datasets"]["uk"]:
        if original["image_id"]==reference:continue
        row=deepcopy(original);row["split"]="uk_vp"
        record=records[row["image_id"]]
        raw=vp_run/record["prediction_file"]
        destination=target/record["prediction_file"]
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(raw,destination)
        row["predictions"]["visual_prompt"]=json.loads(raw.read_text())["predictions"]
        row["metrics"]["visual_prompt"]=None
        row["raw_files"]["visual_prompt"]="visual_prompt/"+record["prediction_file"]
        rows.append(row)
    counts=output_counts(rows,"visual_prompt",bundle["meta"]["classes"])
    if counts!=report["output_counts"] or verified!=report["verified_prediction_count"]:
        raise ValueError("Visual-prompt output counts do not reproduce")
    bundle["datasets"]["uk_vp"]=rows
    bundle["meta"]["visual_prompt"]={"reference_id":reference,"target_count":len(rows),"output_counts":counts,
        "runtime":report["runtime"],"elapsed_seconds":report["elapsed_seconds"],"gradient_steps":0,
        "conditioned_model_sha256":report["conditioned_model_sha256"],"raw_results_sha256":digest(vp_run/"results.json"),
        "qualitative_examples":protocol["qualitative_examples"],"not_accuracy":True}
    reference_row=next(r for r in bundle["datasets"]["uk"] if r["image_id"]==reference)
    annotated_image(output/reference_row["image_file"],[{**b,"score":1.} for b in protocol["reference"]["boxes"]],
        bundle["meta"]["classes"],"REFERENCE ONLY | provisional prompt boxes | excluded from targets",show_scores=False).save(target/"reference_preview.jpg",quality=94)
    case=next(r for r in rows if r["image_id"]==protocol["qualitative_examples"][0])
    annotated_image(output/case["image_file"],case["predictions"]["visual_prompt"],bundle["meta"]["classes"],
        f"{case['image_id']} | visual prompt | score >= 0.05 | NOT ground truth",score_threshold=.05).save(target/"low_threshold_example.jpg",quality=94)
    return verified


def build_report(run, vp_run=None):
    report=json.loads((run/"results.json").read_text())
    config=report["config"]
    manifest=json.loads((run/"dataset_manifest.json").read_text())
    uk=json.loads((run/"uk_manifest.json").read_text())
    choices=json.loads((run/"frozen_choices.json").read_text())
    protocol_sha=digest(run/"code/keen_components_v1.json")
    check_completed(report,choices,protocol_sha)
    if digest(run/"frozen_choices.json")!=report["frozen_choices_sha256"]:
        raise ValueError("Frozen choice receipt changed")
    if digest(run/"dataset_manifest.json")!=report["dataset_manifest_sha256"] or digest(run/"uk_manifest.json")!=report["uk_manifest_sha256"]:
        raise ValueError("Input snapshots changed")
    if digest(run/"training/weights/best.pt")!=report["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint does not match evaluated model")
    for rel,sha in report["source_snapshots"].items():
        if digest(run/"code"/Path(rel).name)!=sha:
            raise ValueError("Training code snapshot changed")
    output=run/"report"
    (output/"images").mkdir(parents=True,exist_ok=True)
    verified_images=verified_predictions=0
    per_image=[];metrics_csv=[]
    bundle={"meta":{"classes":config["classes"], "colors":COLORS, "runtime":report["runtime"],
                    "training":config["training"], "training_seconds":report["training_seconds"],
                    "elapsed_seconds":report["elapsed_seconds"], "dataset_summary":manifest["summary"],
                    "checkpoint_sha256":report["selected_checkpoint_sha256"], "protocol_sha256":protocol_sha,
                    "source_csv_sha256":config["source_csv_sha256"], "raw_results_sha256":digest(run/"results.json")},
            "aggregate":report["evaluations"],"datasets":{}}
    for split in ["eval","dev","uk"]:
        rows=uk["images"] if split=="uk" else [r for r in manifest["images"] if r["split"]==split]
        source=ROOT/(config["uk_pilot"] if split=="uk" else config["dataset"])
        payloads={}
        arm_records={}
        for arm in ARMS:
            arm_report=json.loads((run/split/f"{arm}.json").read_text())
            records=arm_report["records"]
            verified_predictions+=verify_predictions(run/split,rows,records,config["classes"],config["inference"]["confidence_floor"])
            arm_records[arm]={r["image_id"]:r for r in records}
            payloads[arm]={r["image_id"]:json.loads((run/split/r["prediction_file"]).read_text()) for r in records}
            if split!="uk":
                metric_rows=[{"image_id":r["image_id"],"references":r["references"],"predictions":payloads[arm][r["image_id"]]["predictions"]} for r in rows]
                expected=summarize(metric_rows,config["classes"],config["evaluation"]["score_thresholds"],config["evaluation"]["ap_ious"])
                if expected!=arm_report["summary"] or expected!=report["evaluations"][split][arm]["summary"]:
                    raise ValueError("Aggregate metrics do not reproduce from saved raw outputs")
                if geometric_confusion(metric_rows,config["classes"])!=arm_report["geometric_confusion"]:
                    raise ValueError("Confusion audit differs from raw predictions")
                for confidence,point in expected["operating_points"].items():
                    for name,values in point["per_class"].items():
                        metrics_csv.append({"split":split,"arm":arm,"class":name,"confidence":confidence,
                                            **values,"ap50":expected["ap"][name]["ap50"],"ap50_95":expected["ap"][name]["ap50_95"]})
            elif arm_report["summary"] is not None or arm_report["has_ground_truth"]:
                raise ValueError("UK pilot must not claim ground truth or accuracy")
        samples=[]
        for r in rows:
            src=source/r["image_file"]
            if digest(src)!=r["sha256"]:
                raise ValueError(f"Original source image changed: {src}")
            target=output/"images"/(r["image_id"]+".jpg")
            if target.exists():
                if digest(target)!=r["sha256"]:
                    raise ValueError("Existing report photo differs from the original")
            else:
                shutil.copyfile(src,target)
            verified_images+=1
            sample={"image_id":r["image_id"],"file_name":r.get("file_name",r.get("title")),
                    "title":r.get("title",r.get("file_name")),"width":r["width"],"height":r["height"],
                    "image_file":"images/"+target.name,"sha256":r["sha256"],"split":split,"circuit":r.get("circuit"),
                    "source_page":r.get("source_page",manifest["source_page"]),
                    "credit":r.get("attribution") or r.get("author") or manifest["publisher"],
                    "license":r.get("license",manifest["license"]),"license_url":r.get("license_url",manifest["license_url"]),
                    "references":[{k:a[k] for k in ["class_id","class_name","box"]} for a in r.get("references",[])],
                    "predictions":{},"metrics":{},"raw_files":{}}
            for arm in ARMS:
                record=arm_records[arm][r["image_id"]]
                sample["predictions"][arm]=payloads[arm][r["image_id"]]["predictions"]
                sample["metrics"][arm]=record["metrics_025"]
                sample["raw_files"][arm]=f"../{split}/{record['prediction_file']}"
                per_image.append({"split":split,"arm":arm,"image_id":r["image_id"],"inference_seconds":record["elapsed_seconds"],
                                  **({k:record["metrics_025"][k] for k in ["tp","fp","fn","precision","recall","f1"]} if split!="uk" else {})})
            samples.append(sample)
        bundle["datasets"][split]=samples
    bundle["meta"]["uk_output_counts"]={arm:output_counts(bundle["datasets"]["uk"],arm,config["classes"]) for arm in ARMS}
    if vp_run is not None:
        verified_predictions+=add_visual_prompt(run,output,bundle,vp_run)
    cases=choose_cases(bundle["datasets"]["eval"])
    bundle["cases"]=cases
    write_json(output/"case_selection.json",cases)
    queue=review_queue(bundle["datasets"]["uk"],report["selected_checkpoint_sha256"])
    write_json(output/"material_review_queue.json",queue)
    shutil.copyfile(ROOT/"UK_COMPONENT_ANNOTATION_GUIDE.md",output/"UK_COMPONENT_ANNOTATION_GUIDE.md")
    bundle["meta"]["material_review_proposals"]=queue["count"]
    bundle["meta"]["annotation_image_tasks"]=queue["image_task_count"]
    write_json(output/"data.json",bundle)
    for filename,rows in [("metrics.csv",metrics_csv),("per_image.csv",per_image)]:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (output/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    template=(ROOT/"templates/keen_components_report.html").read_text()
    # Escape '<' so third-party titles cannot terminate the inert JSON script tag.
    embedded=json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(",",":")).replace("<","\\u003c")
    (output/"index.html").write_text(template.replace("__DATA_JSON__",embedded))
    build_charts(output,bundle,run)
    from PIL import Image,ImageDraw
    ids=[cases[k] for k in ["default","gain","misses"]]
    source_rows={r["image_id"]:r for r in bundle["datasets"]["eval"]}
    sheet=Image.new("RGB",(1800,1010),"#edf2f7")
    for j,key in enumerate(ids):
        row=source_rows[key]
        for i,arm in enumerate(ARMS):
            m=row["metrics"][arm]
            cap=f"{key} | {arm} | TP {m['tp']} / FP {m['fp']} / FN {m['fn']}"
            tile=annotated_image(output/row["image_file"],row["predictions"][arm],config["classes"],cap,(600,445))
            sheet.paste(tile,(j*600,i*505))
    sheet.save(output/"examples.jpg",quality=93)
    default=source_rows[cases["default"]]
    annotated_image(output/default["image_file"],default["predictions"]["supervised"],config["classes"],
                    f"{default['image_id']} | real model outputs | score >= 0.25 | material NOT classified",(1500,1100)).save(output/"example_supervised.jpg",quality=94)
    audit=ROOT/"runtime/target_sources/uvinsdet_label_audit.json"
    if audit.exists():
        shutil.copyfile(audit,output/audit.name)
        shutil.copyfile(audit.with_name("uvinsdet_porcelain_audit.jpg"),output/"uvinsdet_porcelain_audit.jpg")
    baseline=report["evaluations"]["eval"]["open_vocabulary"]["summary"]
    trained=report["evaluations"]["eval"]["supervised"]["summary"]
    rows=["# GridSight 配电多部件实验\n",f"Slurm {report['runtime']['slurm_job_id']}；固定40轮；单张GH200。训练 {report['training_seconds']:.1f} 秒。\n",
          "500张EPRI原图：320训练、80开发、100按线路分离评估；另27张英国照片仅作定性迁移。\n",
          "| 指标（新100张评估图） | 原始开放词汇 | 监督适配 |\n|---|---:|---:|"]
    for key,title in [("map50","三类mAP@0.50"),("map50_95","三类mAP@0.50:0.95")]:
        rows.append(f"| {title} | {baseline[key]*100:.1f}% | {trained[key]*100:.1f}% |")
    for name in config["classes"]:
        a=baseline["operating_points"]["0.25"]["per_class"][name];b=trained["operating_points"]["0.25"]["per_class"][name]
        rows.append(f"| {name} F1（分数≥0.25，IoU≥0.5） | {a['f1']*100:.1f}% | {b['f1']*100:.1f}% |")
    uk_counts=bundle["meta"]["uk_output_counts"]["supervised"]["0.25"]
    rows += ["\n## 英国迁移仍弱\n",
             f"固定分数0.25：27张中{uk_counts['images_without_output']}张没有输出框；杆体{uk_counts['boxes_by_class']['pole']}个、横担{uk_counts['boxes_by_class']['crossarm']}个、绝缘子{uk_counts['boxes_by_class']['insulator']}个。无GT，不能把空输出直接计为漏检或把框数当准确率。\n",
             f"已导出全部{queue['image_task_count']}张的补标任务，包括空输出图；其中只有{queue['count']}个监督模型绝缘子候选。所有材质字段为空，必须整图补漏、人工复核。\n"]
    if "visual_prompt" in bundle["meta"]:
        vp=bundle["meta"]["visual_prompt"];vc=vp["output_counts"]["0.25"]
        rows += ["\n## 单参考视觉提示诊断\n",
                 f"Slurm {vp['runtime']['slurm_job_id']}，零梯度更新。用一张英国图的5个临时部件框提取视觉提示；只在其余26张图推理，参考图不计入目标。分数0.25时{vc['images_with_output']}/26张有输出，类别框数{vc['boxes_by_class']}。这批图已被模型检查过，不是新鲜holdout；框数不是准确率，也未训练材质。\n"]
    rows += ["\n## 边界\n","这些指标不是英国资产准确率，也不与旧InsPLAD单类F1直接比较。线路是发布者分组，未额外核实物理资产身份；训练/评估没有全背景负例，不能证明非杆图过滤能力。公开预训练数据重叠未知。\n",
             "当前只学习 pole、crossarm、insulator。crossarm未标材质，不能直接称steelwork；绝缘子材质与pole-top未验证。UVInsDet瓷标注只来自训练1图、测试1图，且逐盘/伞裙标注粒度与整串目标不一致，因此没有启动看似丰富、实际缺少独立来源的材质训练。\n",
             "AP为101点插值的全尺寸框指标，IoU=.50:.95，不是EPRI官方榜单成绩；固定阈值P/R/F1从原始保存预测逐类重算。分数不是校准概率。\n",
             "所有样本与失败例都可浏览；英国图无GT，页面不显示准确率。结果均为Roihu预计算，不是在线推理。仅生成本地报告，未发布GitHub。\n",
             "## 来源\n","[EPRI发布页](https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery)，EPRI / P. Kulkarni / D. Lewis，CC BY-SA 4.0，DOI 10.34740/kaggle/dsv/3803175。英国照片各自署名与CC BY-SA 2.0保留在页面。\n",
             f"核验 {verified_predictions} 个原始预测JSON、{verified_images} 张展示原图、最佳权重SHA与冻结边界。\n"]
    (output/"RESULTS.md").write_text("\n".join(rows))
    verification={"status":"VERIFIED_FROM_ORIGINAL_PREDICTIONS", "prediction_files":verified_predictions,
                  "original_display_images":verified_images, "selected_checkpoint_sha256":report["selected_checkpoint_sha256"],
                  "raw_results_sha256":digest(run/"results.json"), "frozen_choices_sha256":digest(run/"frozen_choices.json"),
                  "dataset_manifest_sha256":digest(run/"dataset_manifest.json"), "uk_manifest_sha256":digest(run/"uk_manifest.json"),
                  "report_script_sha256":digest(__file__), "template_sha256":digest(ROOT/"templates/keen_components_report.html"),
                  "html_sha256":digest(output/"index.html"), "metrics_recomputed":True, "manual_predictions_added":False,
                  "unreviewed_material_proposals":queue["count"],"annotation_image_tasks":queue["image_task_count"],
                  "visual_prompt":bundle["meta"].get("visual_prompt"),
                  "notes":"Displayed images are byte-identical originals. Figures with overlays are separate derivatives. Only the selected trained checkpoint was rehashed locally; baseline hashes were checked by the Roihu initializer."}
    write_json(output/"verification.json",verification)
    print(json.dumps({"event":"REPORT_VERIFIED",**verification,"output":str(output)}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,default=DEFAULT_RUN)
    p.add_argument("--visual-prompt-run",type=Path)
    args=p.parse_args();build_report(args.run,args.visual_prompt_run)
