"""Prospective UK material transfer and component-localisation diagnostic on Roihu."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from material_head_v2_common import decide_v2, diagnostic_counts, margin

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def extent(box, width, height, padding):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    return [max(0, int(x0 - dx)), max(0, int(y0 - dy)),
            min(width, int(x1 + dx + .999999)), min(height, int(y1 + dy + .999999))]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x1 - x0) * max(0., y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.


def greedy_matches(references, predictions, threshold):
    candidates = sorted(((iou(r["xyxy"], p["box"]), ri, pi)
                         for ri, r in enumerate(references) for pi, p in enumerate(predictions)), reverse=True)
    used_r, used_p, matches = set(), set(), []
    for overlap, ri, pi in candidates:
        if overlap < threshold:
            break
        if ri not in used_r and pi not in used_p:
            used_r.add(ri); used_p.add(pi)
            matches.append({"reference_index": ri, "prediction_index": pi, "iou": overlap})
    return matches


def head_from(saved, torch):
    head = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.GELU(), torch.nn.Linear(256, 4)).to("cuda")
    head.load_state_dict({name: torch.tensor(saved[name], device="cuda")
                          for name in ("0.weight", "0.bias", "2.weight", "2.bias")})
    return head


def asset_counts(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["record_id"]].append(record)
    decisions = []
    for record_id, rows in grouped.items():
        expected = {r["expected_material"] for r in rows}
        if len(expected) != 1:
            raise ValueError(f"Mixed test material for {record_id}")
        accepted = [r["decision"]["material"] for r in rows if r["decision"]["material"] != "unknown"]
        material = accepted[0] if accepted and len(set(accepted)) == 1 else "unknown"
        decisions.append({"record_id": record_id, "expected_material": expected.pop(), "material": material,
                          "accepted_region_decisions": len(accepted),
                          "reason": "consistent accepted regions" if material != "unknown" else "no accepted region or region disagreement"})
    accepted = [r for r in decisions if r["material"] != "unknown"]
    return {"targets": len(decisions), "accepted": len(accepted),
            "correct_accepted": sum(r["material"] == r["expected_material"] for r in accepted),
            "coverage": len(accepted) / len(decisions) if decisions else None,
            "accepted_accuracy": (sum(r["material"] == r["expected_material"] for r in accepted) / len(accepted)
                                  if accepted else None), "decisions": decisions}


def main(config_name):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gputest":
        raise RuntimeError("Requires Roihu gputest; no local model fallback")
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from ultralytics import YOLOE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocol = ROOT / config_name
    cfg = load(protocol)
    source = ROOT / cfg["source_dataset"]
    pins = [(source / "manifest.json", cfg["source_manifest_sha256"]),
            (ROOT / cfg["encoder"] / "model.safetensors", cfg["encoder_sha256"])]
    for key in ("v2_config", "v2_results", "v2_head", "mpid_features", "mpid_crop_manifest",
                "substation_features", "substation_index", "component_detector"):
        pins.append((ROOT / cfg[key], cfg[key + "_sha256"]))
    for path, expected in pins:
        if sha(path) != expected:
            raise ValueError(f"Hash mismatch: {path}")
    manifest = load(source / "manifest.json")
    if not manifest["selection_frozen_before_model_inference"] or manifest["asset_group_overlap"]:
        raise ValueError("Prospective split boundary is not frozen")
    adaptation = [r for r in manifest["records"] if r["role"] == "adaptation"]
    test = [r for r in manifest["records"] if r["role"] == "prospective_test"]
    if {r["asset_group"] for r in adaptation} & {r["asset_group"] for r in test}:
        raise ValueError("Adaptation/test asset leakage")
    for record in adaptation + test:
        if record["model_inference_performed_before_freeze"]:
            raise ValueError("Test source was not prospective")
        if sha(ROOT / record["image_file"]) != record["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {record['record_id']}")

    out = ROOT / cfg["run"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "code").mkdir()
    for path in (protocol, Path(__file__), ROOT / "scripts/material_head_v2_common.py",
                 ROOT / "scripts/uk_material_prospective_v1.sbatch",
                 ROOT / "scripts/acquire_uk_material_prospective_v1.py"):
        shutil.copy2(path, out / "code" / path.name)
    result = {"status": "PREPARING", "started_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(protocol), "source_manifest_sha256": sha(source / "manifest.json"),
              "runtime": {"job_id": os.environ["SLURM_JOB_ID"], "device": torch.cuda.get_device_name(),
                          "torch": torch.__version__, "transformers": transformers.__version__},
              "encoder_gradient_steps": 0, "head_gradient_steps": 0,
              "test_used_for_training_or_selection": False, "claim_boundary": cfg["claim_boundary"]}
    write(out / "results.json", result)
    started = time.perf_counter()
    try:
        v2cfg, v2result = load(ROOT / cfg["v2_config"]), load(ROOT / cfg["v2_results"])
        names = cfg["classes"]
        if names != v2cfg["classes"]:
            raise ValueError("Class mapping changed")
        processor = AutoProcessor.from_pretrained(ROOT / cfg["encoder"], local_files_only=True, trust_remote_code=False)
        encoder = AutoModel.from_pretrained(ROOT / cfg["encoder"], local_files_only=True, trust_remote_code=False,
                                            use_safetensors=True).to("cuda").eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

        def encode(images):
            chunks = []
            for begin in range(0, len(images), cfg["batch_size"]):
                batch = processor(images=images[begin:begin + cfg["batch_size"]], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    value = encoder.get_image_features(**batch)
                    if not isinstance(value, torch.Tensor):
                        value = value.pooler_output
                    chunks.append(F.normalize(value.float(), dim=-1).cpu().numpy())
            return np.concatenate(chunks) if chunks else np.zeros((0, 768), np.float32)

        region_images, region_rows = [], []
        for split, records in (("adaptation", adaptation), ("prospective_test", test)):
            for record in records:
                image = Image.open(ROOT / record["image_file"]).convert("RGB")
                for index, box in enumerate(record["regions"]):
                    row = {"id": f"{record['record_id']}_r{index + 1}", "record_id": record["record_id"],
                           "asset_group": record["asset_group"], "split": split,
                           "expected_material": record["material"], "expected_class": names.index(record["material"]),
                           "xyxy": box, "views": {}}
                    for view, padding in (("tight", 0), ("context", cfg["context_padding"])):
                        crop = extent(box, image.width, image.height, padding)
                        row["views"][view] = {"feature_index": len(region_images), "crop_xyxy": crop}
                        region_images.append(image.crop(crop))
                    region_rows.append(row)
        result["status"] = "ENCODING_FROZEN_REGIONS"
        write(out / "results.json", result)
        region_features = encode(region_images)
        np.savez_compressed(out / "region_features.npz", embeddings=region_features)

        mpid_features = np.load(ROOT / cfg["mpid_features"])["embeddings"]
        mpid_manifest = load(ROOT / cfg["mpid_crop_manifest"])
        old_features = np.load(ROOT / cfg["substation_features"])["embeddings"]
        old_index = load(ROOT / cfg["substation_index"])
        base_x, base_y, dev_pairs = [], [], []
        for record in mpid_manifest:
            tight = mpid_features[record["views"]["tight"]["feature_index"]]
            context = mpid_features[record["views"]["context"]["feature_index"]]
            if record["split"] == "train":
                base_x.extend((tight, context)); base_y.extend((record["class_id"], record["class_id"]))
            else:
                dev_pairs.append({"expected_class": record["class_id"], "tight": tight, "context": context})
        old_map = {0: 0, 1: 1, 2: 3}
        old_by = defaultdict(dict)
        for index, row in enumerate(old_index):
            old_by[(row["crop_id"], row["split"], old_map[row["class_id"]])][row["view"]] = old_features[index]
        for (_, split, class_id), views in old_by.items():
            if split == "train":
                base_x.extend((views["tight"], views["context"])); base_y.extend((class_id, class_id))
            else:
                dev_pairs.append({"expected_class": class_id, "tight": views["tight"], "context": views["context"]})
        base_x_np = np.stack(base_x)
        base_y_np = np.asarray(base_y)
        base_x_gpu = torch.tensor(base_x_np, device="cuda")
        base_y_gpu = torch.tensor(base_y_np, device="cuda")
        counts = torch.bincount(base_y_gpu, minlength=len(names))
        class_weights = len(base_y_gpu) / (len(names) * counts.float())
        dev_features = np.stack([v for pair in dev_pairs for v in (pair["tight"], pair["context"])])
        saved = np.load(ROOT / cfg["v2_head"])
        centroids = saved["centroids"]
        dev_similarity = dev_features @ centroids.T

        adaptation_rows = [r for r in region_rows if r["split"] == "adaptation"]
        adaptation_indices = [r["views"][v]["feature_index"] for r in adaptation_rows for v in ("tight", "context")]
        adaptation_x = torch.tensor(region_features[adaptation_indices], device="cuda")
        adaptation_y = torch.tensor([r["expected_class"] for r in adaptation_rows for _ in (0, 1)], device="cuda")
        head = head_from(saved, torch)
        for parameter in head[0].parameters():
            parameter.requires_grad_(False)
        optimiser = torch.optim.AdamW(head[2].parameters(), lr=cfg["adaptation"]["learning_rate"],
                                      weight_decay=cfg["adaptation"]["weight_decay"])
        torch.manual_seed(cfg["adaptation"]["seed"])
        losses = []
        result["status"] = "ADAPTING_FINAL_LAYER"
        write(out / "results.json", result)
        for step in range(cfg["adaptation"]["steps"]):
            optimiser.zero_grad()
            base_loss = F.cross_entropy(head(base_x_gpu), base_y_gpu, weight=class_weights)
            uk_loss = F.cross_entropy(head(adaptation_x), adaptation_y)
            loss = cfg["adaptation"]["base_loss_weight"] * base_loss + cfg["adaptation"]["uk_loss_weight"] * uk_loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite adaptation loss")
            loss.backward(); optimiser.step()
            if step in (0, cfg["adaptation"]["steps"] - 1):
                losses.append({"step": step + 1, "total": float(loss.detach()),
                               "base": float(base_loss.detach()), "uk": float(uk_loss.detach())})
        result["head_gradient_steps"] = cfg["adaptation"]["steps"]
        np.savez_compressed(out / "adapted_head.npz", centroids=centroids,
                            **{name: value.detach().cpu().numpy() for name, value in head.state_dict().items()})

        def derive_thresholds(active):
            with torch.inference_mode():
                logits = active(torch.tensor(dev_features, device="cuda")).cpu().numpy()
            support = defaultdict(lambda: {"margin": [], "similarity": []})
            for index, pair in enumerate(dev_pairs):
                tl, cl = logits[2 * index], logits[2 * index + 1]
                expected = pair["expected_class"]
                if int(tl.argmax()) == expected and int(cl.argmax()) == expected:
                    support[expected]["margin"].append(min(margin(tl), margin(cl)))
                    support[expected]["similarity"].append(min(dev_similarity[2 * index][expected],
                                                                  dev_similarity[2 * index + 1][expected]))
            thresholds = {"margin": [], "similarity": []}
            for class_id in range(len(names)):
                if len(support[class_id]["margin"]) < 10:
                    raise RuntimeError(f"Insufficient academic development support for {names[class_id]}")
                thresholds["margin"].append(max(v2cfg["rejection"]["minimum_logit_margin_floor"],
                    float(np.quantile(support[class_id]["margin"], v2cfg["rejection"]["correct_development_margin_quantile"]))))
                thresholds["similarity"].append(float(np.quantile(support[class_id]["similarity"],
                    v2cfg["rejection"]["correct_development_similarity_quantile"])))
            return thresholds

        baseline = head_from(saved, torch).eval()
        head.eval()
        thresholds = {"baseline": v2result["thresholds"], "adapted": derive_thresholds(head)}
        test_rows = [r for r in region_rows if r["split"] == "prospective_test"]
        test_indices = [r["views"][v]["feature_index"] for r in test_rows for v in ("tight", "context")]
        test_features = region_features[test_indices]
        test_similarity = test_features @ centroids.T
        oracle = {}
        for arm, active in (("baseline", baseline), ("adapted", head)):
            with torch.inference_mode():
                logits = active(torch.tensor(test_features, device="cuda")).cpu().numpy()
            rows = []
            for index, source_row in enumerate(test_rows):
                decision = decide_v2(logits[2 * index].tolist(), logits[2 * index + 1].tolist(),
                                     test_similarity[2 * index].tolist(), test_similarity[2 * index + 1].tolist(),
                                     source_row["xyxy"], v2cfg, thresholds[arm])
                rows.append({**source_row, "decision": decision,
                             "raw": {"tight_logits": logits[2 * index].tolist(),
                                     "context_logits": logits[2 * index + 1].tolist(),
                                     "tight_similarity": test_similarity[2 * index].tolist(),
                                     "context_similarity": test_similarity[2 * index + 1].tolist()}})
            oracle[arm] = {"regions": rows, "region_diagnostics": diagnostic_counts(rows),
                           "asset_diagnostics": asset_counts(rows)}

        detector = YOLOE(str(ROOT / cfg["component_detector"])).to("cuda")
        if detector.names != dict(enumerate(cfg["component_classes"])):
            raise ValueError("Component detector class mapping changed")
        component_records, detected_crops, detected_meta = [], [], []
        for source_record in test:
            image = Image.open(ROOT / source_record["image_file"]).convert("RGB")
            pred = detector.predict(image, imgsz=cfg["detector"]["imgsz"], conf=cfg["detector"]["raw_score_floor"],
                                    iou=cfg["detector"]["nms_iou"], max_det=cfg["detector"]["max_det"],
                                    device=0, half=False, verbose=False)[0]
            raw = [{"prediction_index": i, "class_id": int(cls), "score": float(score), "box": list(map(float, box))}
                   for i, (box, score, cls) in enumerate(zip(pred.boxes.xyxy.cpu().tolist(), pred.boxes.conf.cpu().tolist(),
                                                             pred.boxes.cls.cpu().tolist()))]
            insulators = [p for p in raw if p["class_id"] == 2 and p["score"] >= cfg["detector"]["operating_score"]]
            references = [{"id": f"{source_record['record_id']}_r{i + 1}", "xyxy": box,
                           "expected_material": source_record["material"]} for i, box in enumerate(source_record["regions"])]
            matches = greedy_matches(references, insulators, cfg["detector"]["reference_match_iou"])
            match_by_prediction = {m["prediction_index"]: m for m in matches}
            for prediction_index, prediction in enumerate(insulators):
                box = prediction["box"]
                for view, padding in (("tight", 0), ("context", cfg["context_padding"])):
                    crop = extent(box, image.width, image.height, padding)
                    detected_meta.append({"record_id": source_record["record_id"], "prediction_index": prediction_index,
                                          "view": view, "crop_xyxy": crop})
                    detected_crops.append(image.crop(crop))
            component_records.append({"record_id": source_record["record_id"], "image_sha256": source_record["image_sha256"],
                                      "references": references, "raw_predictions": raw,
                                      "accepted_insulator_predictions": insulators, "matches": matches,
                                      "unmatched_reference_indices": sorted(set(range(len(references))) - {m["reference_index"] for m in matches}),
                                      "unmatched_prediction_indices": sorted(set(range(len(insulators))) - {m["prediction_index"] for m in matches})})
        detected_features = encode(detected_crops)
        detected_similarity = detected_features @ centroids.T if len(detected_features) else np.zeros((0, len(names)))
        np.savez_compressed(out / "detected_crop_features.npz", embeddings=detected_features,
                            similarities=detected_similarity)
        for arm, active in (("baseline", baseline), ("adapted", head)):
            with torch.inference_mode():
                detected_logits = (active(torch.tensor(detected_features, device="cuda")).cpu().numpy()
                                   if len(detected_features) else np.zeros((0, len(names))))
            cursor = 0
            for record in component_records:
                material_decisions = []
                match_by_prediction = {m["prediction_index"]: m for m in record["matches"]}
                for prediction_index, prediction in enumerate(record["accepted_insulator_predictions"]):
                    decision = decide_v2(detected_logits[cursor].tolist(), detected_logits[cursor + 1].tolist(),
                                         detected_similarity[cursor].tolist(), detected_similarity[cursor + 1].tolist(),
                                         prediction["box"], v2cfg, thresholds[arm])
                    match = match_by_prediction.get(prediction_index)
                    material_decisions.append({"prediction_index": prediction_index, "decision": decision,
                        "matched_reference_index": match["reference_index"] if match else None,
                        "expected_material": (record["references"][match["reference_index"]]["expected_material"] if match else None),
                        "raw": {"tight_logits": detected_logits[cursor].tolist(), "context_logits": detected_logits[cursor + 1].tolist(),
                                "tight_similarity": detected_similarity[cursor].tolist(),
                                "context_similarity": detected_similarity[cursor + 1].tolist()}})
                    cursor += 2
                record.setdefault("material", {})[arm] = material_decisions

        total_refs = sum(len(r["references"]) for r in component_records)
        total_predictions = sum(len(r["accepted_insulator_predictions"]) for r in component_records)
        total_matches = sum(len(r["matches"]) for r in component_records)
        localisation = {"reference_regions": total_refs, "accepted_predictions": total_predictions,
                        "matched_regions": total_matches, "region_coverage": total_matches / total_refs,
                        "unmatched_predictions": total_predictions - total_matches,
                        "operating_score": cfg["detector"]["operating_score"],
                        "match_iou": cfg["detector"]["reference_match_iou"],
                        "reference_regions_are_expert_ground_truth": False}
        end_to_end = {}
        for arm in ("baseline", "adapted"):
            matched = [m for r in component_records for m in r["material"][arm] if m["matched_reference_index"] is not None]
            accepted = [m for m in matched if m["decision"]["material"] != "unknown"]
            end_to_end[arm] = {"reference_regions": total_refs, "localised": len(matched),
                               "material_accepted": len(accepted),
                               "material_correct": sum(m["decision"]["material"] == m["expected_material"] for m in accepted),
                               "end_to_end_coverage": len(accepted) / total_refs,
                               "accepted_material_accuracy": (sum(m["decision"]["material"] == m["expected_material"] for m in accepted) / len(accepted)
                                                              if accepted else None)}
        write(out / "oracle_decisions.json", oracle)
        write(out / "component_predictions.json", component_records)
        result.update(status="COMPLETE", adaptation_losses=losses, thresholds=thresholds,
                      adapted_head_sha256=sha(out / "adapted_head.npz"),
                      region_features_sha256=sha(out / "region_features.npz"),
                      detected_crop_features_sha256=sha(out / "detected_crop_features.npz"),
                      oracle_decisions_sha256=sha(out / "oracle_decisions.json"),
                      component_predictions_sha256=sha(out / "component_predictions.json"),
                      oracle_diagnostics={arm: {"regions": oracle[arm]["region_diagnostics"],
                                                "assets": {k: v for k, v in oracle[arm]["asset_diagnostics"].items() if k != "decisions"}}
                                          for arm in oracle},
                      localisation_diagnostics=localisation, end_to_end_diagnostics=end_to_end,
                      outputs_are_probabilities=False, deployment_claim=False,
                      elapsed_seconds=time.perf_counter() - started)
        write(out / "results.json", result)
        print(json.dumps({"status": result["status"], "oracle": result["oracle_diagnostics"],
                          "localisation": localisation, "end_to_end": end_to_end}, indent=2), flush=True)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      elapsed_seconds=time.perf_counter() - started)
        write(out / "results.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/uk_material_prospective_v1.json")
    main(parser.parse_args().config)
