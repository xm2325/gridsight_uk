"""Pure-Python class-aware object detection metrics, independent of Ultralytics.

Scores are model outputs, not calibrated probabilities. AP is 101-point
interpolation over all boxes, with no area ranges, crowd labels or ignores.
"""
from __future__ import annotations
from collections import defaultdict
import math
from roihu_demo_ablation import iou, nms, counts_to_metrics


def validate_box(box):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Invalid box coordinates")
    if not box[0] < box[2] or not box[1] < box[3]:
        raise ValueError("Empty or inverted box")


def validate_predictions(predictions, width, height, class_count):
    for p in predictions:
        validate_box(p["box"])
        if type(p["class_id"]) is not int or not 0 <= p["class_id"] < class_count:
            raise ValueError("Invalid class ID")
        if not math.isfinite(p["score"]) or not 0 <= p["score"] <= 1:
            raise ValueError("Invalid model score")
        x1, y1, x2, y2 = p["box"]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError("Prediction outside image")


def match_image(predictions, references, threshold=.5, confidence=.25, class_id=None):
    selected = [(i, p) for i, p in enumerate(predictions) if p["score"] >= confidence
                and (class_id is None or p["class_id"] == class_id)]
    selected.sort(key=lambda x: (-x[1]["score"], x[0]))
    unused = {i for i, r in enumerate(references) if class_id is None or r["class_id"] == class_id}
    matches, false_positives = [], []
    for i, p in selected:
        candidates = [j for j in sorted(unused) if references[j]["class_id"] == p["class_id"]]
        best = max(candidates, key=lambda j: iou(p["box"], references[j]["box"]), default=None)
        overlap = iou(p["box"], references[best]["box"]) if best is not None else 0
        if best is not None and overlap >= threshold:
            unused.remove(best)
            matches.append({"prediction_index": i, "reference_index": best, "iou": overlap})
        else:
            false_positives.append(i)
    return {**counts_to_metrics(len(matches), len(false_positives), len(unused)),
            "matches": matches, "false_positive_indices": false_positives, "missed_reference_indices": sorted(unused)}


def class_ap(records, class_id, threshold=.5):
    refs = {r["image_id"]: [a for a in r["references"] if a["class_id"] == class_id] for r in records}
    support = sum(map(len, refs.values()))
    if support == 0:
        return None
    detections = [(p["score"], r["image_id"], index, p) for r in records
                  for index, p in enumerate(r["predictions"]) if p["class_id"] == class_id]
    detections.sort(key=lambda x: (-x[0], str(x[1]), x[2]))
    unused = {key: set(range(len(value))) for key, value in refs.items()}
    tp, fp, curve = 0, 0, []
    for _, key, _, p in detections:
        best = max(sorted(unused[key]), key=lambda j: iou(p["box"], refs[key][j]["box"]), default=None)
        if best is not None and iou(p["box"], refs[key][best]["box"]) >= threshold:
            tp += 1
            unused[key].remove(best)
        else:
            fp += 1
        curve.append((tp / support, tp / (tp + fp)))
    # Interpolated precision envelope, including recall=0 and recall=1.
    return sum(max((p for r, p in curve if r >= level/100), default=0.) for level in range(101))/101


def summarize(records, names, thresholds=(.05, .25, .5), ap_ious=(.5,.55,.6,.65,.7,.75,.8,.85,.9,.95)):
    ids = [r["image_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image records would inflate metrics")
    ap = {}
    for c, name in enumerate(names):
        values = [class_ap(records, c, t) for t in ap_ious]
        present = [v for v in values if v is not None]
        ap[name] = {"support": sum(a["class_id"] == c for r in records for a in r["references"]),
                    "ap50": values[0], "ap50_95": sum(present)/len(present) if present else None,
                    "by_iou": {f"{t:.2f}": v for t, v in zip(ap_ious, values)}}
    aps50 = [v["ap50"] for v in ap.values() if v["ap50"] is not None]
    aps = [v["ap50_95"] for v in ap.values() if v["ap50_95"] is not None]
    operating = {}
    for threshold in thresholds:
        classes = {}
        for c, name in enumerate(names):
            metrics = [match_image(r["predictions"], r["references"], .5, threshold, c) for r in records]
            classes[name] = counts_to_metrics(*(sum(m[key] for m in metrics) for key in ("tp", "fp", "fn")))
        micro = counts_to_metrics(*(sum(m[key] for m in classes.values()) for key in ("tp", "fp", "fn")))
        operating[f"{threshold:.2f}"] = {"per_class": classes, "micro": micro,
                                          "macro_f1": sum(m["f1"] for m in classes.values()) / len(classes)}
    return {"images": len(records), "ap": ap, "map50": sum(aps50)/len(aps50) if aps50 else None,
            "map50_95": sum(aps)/len(aps) if aps else None, "operating_points": operating}


def geometric_confusion(records, names, confidence=.25, threshold=.5):
    """Additional error audit only: geometry matching allows wrong-class matches."""
    k = len(names)
    matrix = [[0]*(k+1) for _ in range(k+1)]
    for r in records:
        unused = set(range(len(r["references"])))
        predictions = sorted((p for p in r["predictions"] if p["score"] >= confidence), key=lambda p: -p["score"])
        for p in predictions:
            best = max(sorted(unused), key=lambda j: iou(p["box"], r["references"][j]["box"]), default=None)
            if best is not None and iou(p["box"], r["references"][best]["box"]) >= threshold:
                matrix[r["references"][best]["class_id"]][p["class_id"]] += 1
                unused.remove(best)
            else:
                matrix[k][p["class_id"]] += 1
        for j in unused:
            matrix[r["references"][j]["class_id"]][k] += 1
    return {"labels": [*names, "background"], "rows": "ground truth", "columns": "prediction",
            "matrix": matrix, "not_the_matching_used_for_AP": True}
