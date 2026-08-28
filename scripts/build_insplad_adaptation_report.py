#!/usr/bin/env python3
"""Recompute every saved prediction before rendering adaptation results."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
from pathlib import Path

from analyze_insplad100_errors import match
from prepare_insplad100 import verify_dataset as verify_diagnostic
from prepare_insplad_adaptation import DEFAULT_DATASET, verify_dataset
from roihu_insplad_train import check_gate, check_prompt_selection
from insplad_adapt_common import ROOT, load_protocol, verify_records, digest, summarize, write_json

KEY = "conf_0.05_iou_0.50"
LABELS = {"original_prompt": "Original long prompt", "selected_prompt": "Selected prompt",
          "long_multi": "Original long prompt", "untrained_detector": "Same detector / zero steps",
          "supervised": "Supervised / 20 epochs", "short_multi": "Short / multiple classes",
          "long_single": "Long / single class", "short_single": "Short / single class", "material_names": "Material names"}


def image_rows(rows, record_sources, output, source_images, image_namespace):
    rendered = {row["image_id"]: {**row, "outputs": {}} for row in rows}
    destination = output / "images" / image_namespace
    destination.mkdir(parents=True, exist_ok=True)
    for row in rendered.values():
        original = source_images / row["image_file"]
        shutil.copyfile(original, destination / row["file_name"])
        row["image_file"] = f"images/{image_namespace}/{row['file_name']}"
    for records, base in record_sources:
        for record in records:
            stored = json.loads((base / record["prediction_file"]).read_text())
            rendered[record["image_id"]]["outputs"][record["arm"]] = {**record, "predictions": stored["target_predictions"]}
    return list(rendered.values())


def curves(path):
    with path.open() as stream:
        return [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def verify_frozen_choices(report, choices, protocol_sha256):
    expected = {
        "checkpoint_sha256": report["selected_checkpoint_sha256"],
        "untrained_control_sha256": report["untrained_control"]["sha256"],
        "control_addendum_sha256": report["control_addendum_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "prompt_selection": report["prompt_selection"],
        "protocol_sha256": protocol_sha256,
        "selection_uses_holdout": False,
    }
    if choices != expected:
        raise ValueError("Frozen choices disagree with the evaluated models or protocol")
    if report["untrained_control"]["optimizer_steps"] != 0 or report["untrained_control"]["saved_at"] != "on_train_start":
        raise ValueError("Control must precede every optimizer update")


def plots(data, output, overfit, adapt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    from PIL import Image
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    held = data["datasets"]["holdout"]
    arms = ["original_prompt", "untrained_detector", "supervised"]
    x = np.arange(len(arms))
    fig, axis = plt.subplots(figsize=(10, 4.6))
    for offset, metric, color in zip((-.25, 0, .25), ("precision", "recall", "f1"), ("#376d9a", "#69ad9b", "#d08045")):
        values = [held["summary"][arm]["metrics"][KEY][metric] for arm in arms]
        bars = axis.bar(x + offset, values, width=.23, label=metric.title(), color=color)
        axis.bar_label(bars, labels=[f"{value:.2%}" if 0 < value < .001 else f"{value:.1%}" for value in values], fontsize=9, padding=3)
    axis.set_xticks(x, [LABELS[a] for a in arms]); axis.set_ylim(0, 1.15)
    axis.set_ylabel("Fixed-threshold metric"); axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.set_title("100 newly heldout images | 10 filename families | no training-family overlap", loc="left", pad=16)
    fig.text(.03, .015, "Score >= 0.05 · IoU >= 0.50 · Brazilian InsPLAD · not UK validation / not verified physical-asset independence", fontsize=8, color="#607580")
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(output / "comparison.png", dpi=180); plt.close(fig)

    old, new = curves(overfit / "training/results.csv"), curves(adapt / "training/results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for axis, rows, title in ((axes[0], old, "Two-image reconstruction / no augmentation"),
                             (axes[1], new, "320-image supervised adaptation")):
        for metric, label in (("train/box_loss", "box"), ("train/cls_loss", "classification")):
            axis.plot([r["epoch"] for r in rows], [r[metric] for r in rows], label=label)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Epoch"); axis.set_ylabel("Training loss")
        axis.set_yscale("log"); axis.legend(frameon=False)
    axes[2].plot([r["epoch"] for r in new], [r["metrics/mAP50(B)"] for r in new], label="AP50")
    axes[2].plot([r["epoch"] for r in new], [r["metrics/mAP50-95(B)"] for r in new], label="AP50:95")
    axes[2].set_title("80-image development split / library AP", fontsize=10)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0, 1.05); axes[2].legend(frameon=False)
    fig.suptitle("Actual training curves | development AP is not heldout performance", fontsize=13)
    fig.tight_layout(); fig.savefig(output / "training.png", dpi=180); plt.close(fig)

    rows = held["images"]
    first = next(r for r in rows if r["references"])
    remaining = [r for r in rows if r["image_id"] != first["image_id"]]
    improved = max(remaining, key=lambda r: r["outputs"]["supervised"]["metrics"][KEY]["tp"] -
                   r["outputs"]["original_prompt"]["metrics"][KEY]["tp"])
    remaining = [r for r in remaining if r["image_id"] != improved["image_id"]]
    difficult = max(remaining, key=lambda r: r["outputs"]["supervised"]["metrics"][KEY]["fp"] +
                    r["outputs"]["supervised"]["metrics"][KEY]["fn"])
    selected = [(first, "First annotated sample"),
                (improved, "Largest TP gain / post-hoc"),
                (difficult, "Largest remaining FP + FN / post-hoc")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.2))
    for index, (row, reason) in enumerate(selected):
        with Image.open(output / row["image_file"]) as source:
            pixels = source.convert("RGB")
        for column, arm in enumerate(("original_prompt", "supervised")):
            axis = axes[index, column]; axis.imshow(pixels)
            predictions = [p for p in row["outputs"][arm]["predictions"] if p["score"] >= .05]
            correct, false, missing = match(predictions, row["references"])
            expected = row["outputs"][arm]["metrics"][KEY]
            if (len(correct), len(false), len(missing)) != (expected["tp"], expected["fp"], expected["fn"]):
                raise ValueError("Case overlay disagrees with recorded metrics")
            for reference in row["references"]:
                x1, y1, x2, y2 = reference["box"]
                axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#2563eb", linestyle="--", linewidth=1.5))
            for prediction, color in [(p, "#009e73") for p, _ in correct] + [(p, "#d55e00") for p in false]:
                x1, y1, x2, y2 = prediction["box"]
                axis.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.5))
            axis.set_xticks([]); axis.set_yticks([])
            axis.set_title(f"{LABELS[arm]} | TP {len(correct)} FP {len(false)} FN {len(missing)}", fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{reason}\n{row['file_name']}", fontsize=8)
    fig.suptitle("Actual heldout predictions | blue: reference, green: matched, orange: unmatched", fontsize=12)
    fig.text(.04, .01, "Score >= 0.05, IoU >= 0.50. Illustrative examples; complete 100 images remain in HTML.", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96)); fig.savefig(output / "cases.png", dpi=180); plt.close(fig)
    write_json(output / "case_selection.json", {
        "scope": "First positive in frozen order, then two post-hoc illustrative examples; not a representative sample",
        "images": [{"image_id": r["image_id"], "file_name": r["file_name"], "reason": reason} for r, reason in selected]})


def error_audit(dataset):
    output = {}
    for arm in dataset["arms"]:
        totals, matched_counts = collections.Counter(), collections.Counter()
        negative_fp = 0
        for row in dataset["images"]:
            predictions = [p for p in row["outputs"][arm]["predictions"] if p["score"] >= .05]
            correct, false, _ = match(predictions, row["references"])
            totals.update(ref["category"] for ref in row["references"])
            matched_counts.update(row["references"][index]["category"] for _, index in correct)
            if not row["references"]:
                negative_fp += len(false)
        output[arm] = {"negative_image_false_positives": negative_fp,
                       "recall_by_gt_material_not_material_classification": {
                           name: {"tp": matched_counts[name], "total": total,
                                  "recall": matched_counts[name] / total}
                           for name, total in totals.items()}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--overfit", type=Path, required=True)
    parser.add_argument("--adapt", type=Path, required=True)
    args = parser.parse_args()
    prompts, overfit, adapt = (p.resolve() for p in (args.prompts, args.overfit, args.adapt))
    protocol = load_protocol()
    manifest = verify_dataset(DEFAULT_DATASET)
    diagnostic_dir = ROOT / "data/external/insplad100"
    diagnostic = verify_diagnostic(diagnostic_dir)
    gate_report = check_gate(overfit / "results.json", manifest, protocol)
    selection = check_prompt_selection(prompts / "selection.json", protocol)
    prompt_report = json.loads((prompts / "results.json").read_text())
    report = json.loads((adapt / "results.json").read_text())
    if report["status"] != "COMPLETED_FIXED_ADAPTATION_AND_HELDOUT_EVALUATION":
        raise ValueError("Refuse to present incomplete adaptation or evaluation as completed")
    if report["dataset_manifest_sha256"] != digest(DEFAULT_DATASET / "manifest.json"):
        raise ValueError("Adaptation dataset changed")
    if report["training_progress"]["completed_epochs"] != protocol["adaptation"]["epochs"]:
        raise ValueError("Fixed training budget was not completed")
    if digest(adapt / "training/weights/best.pt") != report["selected_checkpoint_sha256"]:
        raise ValueError("Selected trained checkpoint hash mismatch")
    if digest(adapt / "untrained_detector.pt") != report["untrained_control"]["sha256"]:
        raise ValueError("Zero-step control checkpoint hash mismatch")
    if digest(adapt / "frozen_evaluation_choices.json") != report["evaluation_choices_sha256"]:
        raise ValueError("Evaluation choices changed after freezing")
    verify_frozen_choices(report, json.loads((adapt / "frozen_evaluation_choices.json").read_text()),
                          digest(ROOT / "configs/insplad_adapt_protocol.json"))
    if report["protocol"] != protocol or report["prompt_selection"] != selection:
        raise ValueError("Evaluated protocol or selected prompt does not match the frozen inputs")
    if report["gate_source_sha256"] != digest(overfit / "results.json"):
        raise ValueError("Adaptation used a different reconstruction gate")
    if report["prompt_selection_sha256"] != digest(prompts / "selection.json"):
        raise ValueError("Adaptation used a different prompt selection receipt")
    for filename, expected in report["code_snapshot_sha256"].items():
        if digest(adapt / "code" / filename) != expected:
            raise ValueError(f"Source snapshot mismatch: {filename}")
    actual_training = {Path(p).name for p in report["training_setup"]["actual_training_images"]}
    expected_training = {r["file_name"] for r in manifest["images"] if r["split"] == "train"}
    if actual_training != expected_training:
        raise ValueError("Actual training images differ from the frozen train split")
    training_rows = curves(adapt / "training/results.csv")
    if [r["epoch"] for r in training_rows] != list(range(1, protocol["adaptation"]["epochs"] + 1)):
        raise ValueError("Training curve does not contain exactly the fixed epochs")
    heldout = [row for row in manifest["images"] if row["split"] == "holdout"]
    verified = len(prompt_report["results"]) + len(gate_report["results"])
    if verified != 502:
        raise ValueError("Expected 500 prompt records and two reconstruction records")
    for name, rows, expected_arms in (
        ("diagnostic", diagnostic["images"], ["supervised", "untrained_detector"]),
        ("holdout", heldout, ["supervised", "untrained_detector", "original_prompt", "selected_prompt"])
    ):
        records = report["evaluations"][name]["results"]
        if {r["arm"] for r in records} != set(expected_arms):
            raise ValueError("Missing or unexpected evaluation arm")
        verified += verify_records(records, rows, adapt / name, protocol)
        summary = summarize(records, expected_arms, protocol)
        if summary != report["evaluations"][name]["summary"] or any(s["n_images"] != 100 for s in summary.values()):
            raise ValueError("Evaluation aggregate mismatch or missing images")
    output = adapt / "report"; output.mkdir(exist_ok=True)
    held_images = image_rows(heldout, [(report["evaluations"]["holdout"]["results"], adapt / "holdout")],
                            output, DEFAULT_DATASET, "holdout")
    diagnostic_images = image_rows(diagnostic["images"], [
        (prompt_report["results"], prompts), (report["evaluations"]["diagnostic"]["results"], adapt / "diagnostic")],
        output, diagnostic_dir, "diagnostic")
    diagnostic_records = ([r for r in prompt_report["results"] if r["arm"] == "long_multi"] +
                          report["evaluations"]["diagnostic"]["results"])
    diagnostic_arms = ["long_multi", "untrained_detector", "supervised"]
    data = {
        "runtime": report["runtime"], "gate": gate_report["gate"], "verified_records": verified,
        "selected_prompt": selection["selected_prompt"], "manifest_sha256": report["dataset_manifest_sha256"],
        "prompt_summary": prompt_report["summary"],
        "datasets": {
            "holdout": {
                "name": "新分组评估 · 100张", "arms": ["original_prompt", "untrained_detector", "supervised"],
                "scope": "100张新评估图、10个文件名粗编号。没有参与训练、提示词选择或checkpoint选择；64个参考绝缘子。",
                "images": held_images, "summary": report["evaluations"]["holdout"]["summary"]},
            "diagnostic": {
                "name": "原诊断集 · 适配前后", "arms": diagnostic_arms,
                "scope": "原100张已观察过的诊断图。只能作为开发对照，不能再次声称未见测试。",
                "images": diagnostic_images, "summary": summarize(diagnostic_records, diagnostic_arms, protocol)},
            "prompt_diagnostic": {
                "name": "原诊断集 · 五种提示词", "arms": list(protocol["prompt_arms"]),
                "scope": "五组提示词的完整输出；基线100条复用，400条新推理。只在这组诊断数据上作固定选择。",
                "images": diagnostic_images, "summary": prompt_report["summary"]}
        }}
    data["holdout_error_audit"] = error_audit(data["datasets"]["holdout"])
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = (ROOT / "templates/insplad_adaptation_report.html").read_text()
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", payload))
    flat, per_image = [], []
    for name, dataset in data["datasets"].items():
        for arm in dataset["arms"]:
            summary = dataset["summary"][arm]
            for key, metrics in summary["metrics"].items():
                flat.append({"dataset": name, "arm": arm, "operating_point": key, **metrics,
                             "mean_seconds_per_image": summary["mean_seconds_per_image"]})
            for row in dataset["images"]:
                per_image.append({"dataset": name, "arm": arm, "image_id": row["image_id"],
                                  "file_name": row["file_name"], **row["outputs"][arm]["metrics"][KEY]})
    for filename, rows in (("comparison.csv", flat), ("per_image.csv", per_image)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    plots(data, output, overfit, adapt)
    write_json(output / "error_audit.json", data["holdout_error_audit"])
    lines = ["# 提示词与监督适配：实际结果", "",
             f"适配作业 {report['runtime']['slurm_job_id']}；{report['runtime']['gpu']}；固定20 epochs。",
             "320张训练 / 80张开发 / 100张新评估，粗编号47 / 8 / 10组互不重叠。", "",
             "## 新分组评估：分数≥0.05，IoU≥0.50", "",
             "| 配置 | TP | FP | FN | Precision | Recall | F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in data["datasets"]["holdout"]["arms"]:
        m = data["datasets"]["holdout"]["summary"][arm]["metrics"][KEY]
        lines.append(f"| {LABELS[arm]} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} |")
    lines += ["", "## 改善并不均匀", "",
              "以下按发布方GT材质分组统计定位召回，不是材质分类成绩；仍使用分数≥0.05、IoU≥0.50。", "",
              "| GT材质 | 原始长提示词 | 监督适配 |", "|---|---:|---:|"]
    for material in data["holdout_error_audit"]["original_prompt"]["recall_by_gt_material_not_material_classification"]:
        values = [data["holdout_error_audit"][arm]["recall_by_gt_material_not_material_classification"][material]
                  for arm in ("original_prompt", "supervised")]
        lines.append(f"| {material} | {values[0]['tp']}/{values[0]['total']} | {values[1]['tp']}/{values[1]['total']} |")
    lines += ["", "总分改善不能掩盖某些分组退步；本轮结束后不再用这100张图调参。", "",
              "## 已验证与限制", "",
              f"- 全部 {verified} 条记录从原始预测重新执行NMS、类别合并与一对一IoU匹配；所有指标一致。",
              "- 两图拟合为2 TP / 0 FP / 0 FN，仅证明训练链路可工作；不能称为泛化准确率。",
              "- 适配从原始预训练权重重新开始；另存同一初始化检测模型的零步对照，避免把格式转换误认为训练收益。",
              "- 零步模型是监督训练接口初始化的单类检测器，并不等同于原始开放词汇模型；实际收益应与原始长提示词基线比较。",
              "- checkpoint只按80张开发图的库内fitness选择；开发集只有4个玻璃、28个聚合物绝缘子。",
              "- 新评估图不参与选择。它们来自巴西InsPLAD官方train的重新分组，不是官方测试集，也不是英国评估。",
              "- 文件名粗编号只是保守代理，不能保证实物资产、地点或运营商独立；基础模型预训练重叠未知。",
              "- 只评估绝缘子定位；不是材质分类、缺陷、分割或可投入运维的安全验证。",
              "- 阈值预先规定，表中不是COCO mAP；训练图中库内AP只属于开发集。",
              "- 所有示例保留实际误检/漏检；完整100张均可查看。没有更新GitHub Pages。", "",
              "[交互报告](index.html) · [汇总CSV](comparison.csv) · [逐图CSV](per_image.csv)", "",
              "数据来源：[InsPLAD官方发布](https://data.mendeley.com/datasets/5n3fjgvfyz/1)，CC BY-NC 3.0，本地非商业研究评估。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    verification = {"status": "ALL_ADAPTATION_RESULTS_RECOMPUTED", "records": verified,
                    "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                    "adapt_results_sha256": digest(adapt / "results.json"),
                    "prompt_results_sha256": digest(prompts / "results.json"),
                    "overfit_results_sha256": digest(overfit / "results.json"),
                    "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
                    "control_checkpoint_sha256": report["untrained_control"]["sha256"],
                    "generator_sha256": digest(__file__),
                    "template_sha256": digest(ROOT / "templates/insplad_adaptation_report.html"),
                    "artifacts_sha256": {name: digest(output / name) for name in (
                        "index.html", "RESULTS.md", "comparison.csv", "per_image.csv",
                        "comparison.png", "cases.png", "training.png", "error_audit.json", "case_selection.json")},
                    "report": str(output / "index.html")}
    write_json(output / "verification.json", verification)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
