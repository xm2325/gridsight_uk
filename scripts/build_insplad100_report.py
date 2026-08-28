#!/usr/bin/env python3
"""Verify all 400 stored predictions and render a local, offline result report."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from prepare_insplad100 import verify_dataset
from roihu_benchmark100 import operating_metrics, summarize
from roihu_demo_ablation import digest

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["n640", "n1280", "m1280", "m1280_tiles"]
LABELS = ["n / 640", "n / 1280", "m / 1280", "m / 1280 + tiles"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/external/insplad100")
    args = parser.parse_args()
    run = args.run.resolve()
    dataset = args.dataset.resolve()
    report = json.loads((run / "results.json").read_text())
    if report["status"] != "COMPLETED_100_IMAGE_DIAGNOSTIC" or len(report["results"]) != 400:
        raise ValueError("Refuse to present a partial run as a completed 100-image comparison")
    manifest = verify_dataset(dataset)
    if digest(dataset / "manifest.json") != report["dataset_manifest_sha256"]:
        raise ValueError("Dataset no longer matches inference")
    by_id = {row["image_id"]: {**row, "outputs": {}} for row in manifest["images"]}
    seen = set()
    for row in report["results"]:
        key = row["image_id"], row["arm"]
        if key in seen or row["arm"] not in ARMS:
            raise ValueError("Duplicate or unknown image/arm result")
        seen.add(key)
        image = by_id[row["image_id"]]
        if row["image_sha256"] != image["sha256"]:
            raise ValueError("Prediction image hash differs from manifest")
        prediction_path = (run / row["prediction_file"]).resolve()
        if not prediction_path.is_relative_to(run):
            raise ValueError("Unsafe prediction path")
        stored = json.loads(prediction_path.read_text())
        if (stored["image_id"], stored["arm"]) != key:
            raise ValueError("Prediction identity mismatch")
        predictions = [p for p in stored["merged_predictions"] if p["class_id"] == 2]
        recomputed = operating_metrics(predictions, [ref["box"] for ref in image["references"]], report["protocol"])
        if recomputed != row["metrics"]:
            raise ValueError(f"Metric recomputation failed for {key}")
        image["outputs"][row["arm"]] = {**row, "predictions": predictions}
    if any(set(row["outputs"]) != set(ARMS) for row in by_id.values()):
        raise ValueError("Every image must have every arm")
    if summarize(report["results"], ARMS, report["protocol"]) != report["summary"]:
        raise ValueError("Aggregate metric recomputation failed")
    output = run / "report"
    output.mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    for image in by_id.values():
        shutil.copyfile(dataset / image["image_file"], output / image["image_file"])
    data = {key: report[key] for key in ("summary", "runtime", "run_id", "dataset_manifest_sha256")}
    data["images"] = list(by_id.values())
    template = (ROOT / "templates/insplad100_report.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", payload))
    flat_rows = []
    for arm in ARMS:
        for key, metrics in report["summary"][arm]["metrics"].items():
            flat_rows.append({"arm": arm, "operating_point": key, **metrics,
                              "mean_seconds_per_image": report["summary"][arm]["mean_seconds_per_image"]})
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)
    per_image = []
    key = "conf_0.05_iou_0.50"
    for row in report["results"]:
        per_image.append({name: row[name] for name in ("arm", "image_id", "file_name", "capture_prefix", "reference_count", "elapsed_seconds")}
                         | row["metrics"][key])
    with (output / "per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image[0]))
        writer.writeheader(); writer.writerows(per_image)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={"width_ratios": [1.8, 1]})
    positions = np.arange(4)
    for offset, metric, color in zip((-0.25, 0, 0.25), ("precision", "recall", "f1"), ("#376d9a", "#69ad9b", "#d08045")):
        values = [report["summary"][arm]["metrics"][key][metric] for arm in ARMS]
        bars = left.bar(positions + offset, values, width=0.23, label=metric.title(), color=color)
        left.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=8)
    left.set_ylim(0, 1.08); left.set_xticks(positions, LABELS, rotation=12)
    left.set_ylabel("Fixed-threshold detection metric"); left.legend(frameon=False, ncol=3, loc="upper left")
    latency = [report["summary"][arm]["mean_seconds_per_image"] * 1000 for arm in ARMS]
    bars = right.barh(LABELS, latency, color="#376d9a")
    right.invert_yaxis(); right.set_xlabel("Mean synchronized inference ms / image")
    right.bar_label(bars, labels=[f"{v:.1f}" for v in latency], padding=4)
    right.set_xlim(0, max(latency) * 1.2)
    fig.suptitle("GridSight | 100 real UAV images · 73 annotated insulators", fontsize=16, x=0.03, ha="left")
    fig.text(0.03, 0.02, "InsPLAD validation subset · score >= 0.05 · IoU >= 0.50 · 45 filename groups · not UK validation / not COCO mAP", fontsize=9, color="#526678")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(output / "comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    lines = ["# InsPLAD100：真实 GPU 对照结果", "", f"Slurm 作业：{report['runtime']['slurm_job_id']}；GPU：{report['runtime']['gpu']}。", "",
             "100 张固定真实图像，73 个标注绝缘子，45 个拍摄文件名前缀；未进行模型微调。", "",
             "## 分数 ≥ 0.05，IoU ≥ 0.50", "", "| 配置 | TP | FP | FN | Precision | Recall | F1 | ms/图 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        summary = report["summary"][arm]; m = summary["metrics"][key]
        lines.append(f"| {arm} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} | {summary['mean_seconds_per_image']*1000:.1f} |")
    lines += ["", "四种模型的全部 400 条推理记录已在本地逐条复算；所有固定阈值指标和汇总均一致。", "",
              "- 数据：巴西 InsPLAD 官方验证子集，非英国泛化评估；45 个文件名前缀不等于已验证的45座独立塔。",
              "- 仅评估玻璃与聚合物绝缘子的合并定位，不评估材质分类或缺陷。",
              "- 阈值 0.05 / 0.25 和 IoU 0.30 / 0.50 在推理前规定；不是 COCO mAP。",
              "- 假阳性/漏检按发布方标注计算，标注未经过本项目独立专家复核。",
              "- 基础模型与公开数据的预训练重叠未知。",
              "- 时间不含磁盘解码、模型加载、预热与输出写入，不是生产吞吐或P95。", "",
              "[交互对照](index.html) · [全部固定阈值汇总](comparison.csv) · [逐图结果](per_image.csv)", "",
              "来源：[InsPLAD 官方数据](https://data.mendeley.com/datasets/5n3fjgvfyz/1)，CC BY-NC 3.0，非商业研究评估。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    verification = {"status": "ALL_400_PREDICTIONS_RECOMPUTED", "images": 100, "records": 400,
                    "dataset_manifest_sha256": report["dataset_manifest_sha256"], "report": str(output / "index.html")}
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
