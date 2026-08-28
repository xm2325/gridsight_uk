"""Submit the bounded segmentation run once, recording intent before Slurm."""
import json
import subprocess
from datetime import datetime,timezone
from paper_material_demo import ROOT,load,sha,write
from prepare_component_masks import CONFIG


def main():
    cfg=load(CONFIG);manifest=ROOT/cfg['dataset']/'manifest.json';record=ROOT/'runtime/component_masks_submission.json'
    assert load(manifest)['protocol_sha256']==sha(CONFIG)
    if record.exists() or (ROOT/cfg['run']).exists():raise FileExistsError('Inspect existing receipt/output; no resubmission')
    queue=subprocess.check_output(['squeue','--noheader','--name=gridsight-component-masks-v1','--format=%i'],text=True).strip()
    if queue:raise RuntimeError('A matching job already exists')
    receipt={'status':'SUBMITTING','created_at':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(manifest),
             'protocol_sha256':sha(CONFIG),'runner_sha256':sha(ROOT/'scripts/roihu_component_masks.py'),
             'sbatch_sha256':sha(ROOT/'scripts/component_masks.sbatch')}
    record.parent.mkdir(parents=True,exist_ok=True)
    with record.open('x') as f:json.dump(receipt,f,indent=2)
    result=subprocess.run(['sbatch','--parsable','scripts/component_masks.sbatch',receipt['manifest_sha256']],cwd=ROOT,capture_output=True,text=True)
    receipt.update(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip(),
                   status='SUBMITTED' if result.returncode==0 else 'FAILED_INSPECT_BEFORE_RETRY')
    if result.returncode==0:receipt['job_id']=result.stdout.strip().split(';')[0];assert receipt['job_id'].isdigit()
    write(record,receipt);print(json.dumps(receipt,indent=2));result.check_returncode()


if __name__=='__main__':main()
