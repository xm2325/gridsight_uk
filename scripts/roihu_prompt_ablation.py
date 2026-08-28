#!/usr/bin/env python3
"""Run the fixed prompt factorial on the already-consumed diagnostic images."""
import gc
import json

from prepare_insplad100 import verify_dataset
from insplad_adapt_common import (ROOT, PROTOCOL, load_protocol, start_runtime, create_run,
                                  reuse_diagnostic_baseline, run_predictions, select_prompt, digest, write_json)


def main():
    protocol = load_protocol()
    dataset = ROOT / "data/external/insplad100"
    manifest = verify_dataset(dataset)
    runtime = start_runtime()
    from ultralytics import YOLOE
    import torch
    output, report = create_run("prompts", protocol, runtime)
    report.update(dataset_manifest_sha256=digest(dataset / "manifest.json"), script_sha256=digest(__file__),
                  claim_scope="Adaptive diagnostic data, not untouched test performance")
    write_json(output / "dataset_manifest.json", manifest)
    try:
        report["results"] = reuse_diagnostic_baseline(manifest, dataset, output, protocol)
        write_json(output / "results.json", report)
        for arm, specification in protocol["prompt_arms"].items():
            if arm == "long_multi":
                continue
            model = YOLOE(str(ROOT / protocol["model_checkpoint"])).to("cuda:0")
            model.set_classes(specification["prompts"])
            def progress(record):
                report["results"].append(record)
                write_json(output / "results.json", report)
            run_predictions(model, manifest["images"], dataset, output, arm, specification["target_ids"], protocol,
                            on_progress=progress)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        winner, report["summary"] = select_prompt(report["results"], protocol)
        report["selected_prompt"] = winner
        report["status"] = "COMPLETED_PROMPT_DIAGNOSTIC"
        report["new_gpu_image_inferences"] = sum(not row["reused"] for row in report["results"])
        selection = {"selected_prompt": winner, "specification": protocol["prompt_arms"][winner],
                     "selection_data": "original 100 diagnostic images, not heldout", "source_run": str(output),
                     "protocol_sha256": digest(PROTOCOL), "criterion": protocol["prompt_selection"]}
        write_json(output / "selection.json", selection)
    except Exception as error:
        report["status"] = "FAILED_PARTIAL_PROMPT_RESULTS"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "results.json", report)
    print(json.dumps({"event": "RUN_COMPLETE", "output": str(output), "selected_prompt": winner,
                      "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
