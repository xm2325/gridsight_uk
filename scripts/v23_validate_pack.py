from pathlib import Path
import json
from v23_common import ROOT, dataset_manifest, write_json

expected={"train":{"images":3,"boxes":33},"val":{"images":1,"boxes":9},"test":{"images":1,"boxes":13}}
actual={}
for split in expected:
    imgs=list((ROOT/f"data/images/{split}").glob("*.jpg"))
    labels=list((ROOT/f"data/labels/{split}").glob("*.txt"))
    nboxes=sum(len([x for x in p.read_text().splitlines() if x.strip()]) for p in labels)
    actual[split]={"images":len(imgs),"labels":len(labels),"boxes":nboxes}
    assert actual[split]["images"]==expected[split]["images"], actual
    assert actual[split]["labels"]==expected[split]["images"], actual
    assert actual[split]["boxes"]==expected[split]["boxes"], actual
assert sum(v["boxes"] for v in actual.values())==55
assert (ROOT/"data/images/test/POS_2326530.jpg").exists()
assert (ROOT/"data/images/val/POS_5442616.jpg").exists()
status={"valid":True,"expected":expected,"actual":actual,"dataset_manifest":dataset_manifest()}
write_json(ROOT/"reports/v2_3_pack_validation.json",status)
print(json.dumps(status,indent=2))
