"""Submit this one recorded experiment once; uncertain submissions require inspection."""
import json
import subprocess
from datetime import datetime, timezone
from paper_material_demo import ROOT, load, sha, write


def main():
    record=ROOT/'runtime/substation_material_submission.json'
    cfg_path=ROOT/'configs/substation_material_v1.json';cfg=load(cfg_path)
    manifest=ROOT/cfg['dataset']/'manifest.json'
    assert load(manifest)['protocol_sha256']==sha(cfg_path)
    assert sha(ROOT/cfg['pretrained'])==cfg['pretrained_sha256']
    if record.exists() or (ROOT/cfg['run']).exists():
        raise FileExistsError('Existing submission or output: inspect receipt and Slurm; do not resubmit')
    queue=subprocess.check_output(['squeue','--noheader','--name=gridsight-substation-material-v1','--format=%i'],text=True).strip()
    if queue:
        raise RuntimeError('A matching Slurm job already exists')
    receipt={'status':'SUBMITTING','created_at':datetime.now(timezone.utc).isoformat(),
             'manifest_sha256':sha(manifest),'protocol_sha256':sha(cfg_path),
             'runner_sha256':sha(ROOT/'scripts/roihu_substation_material.py'),
             'sbatch_sha256':sha(ROOT/'scripts/substation_material.sbatch')}
    record.parent.mkdir(parents=True,exist_ok=True)
    with record.open('x') as f:
        json.dump(receipt,f,indent=2)
    result=subprocess.run(['sbatch','--parsable','scripts/substation_material.sbatch',receipt['manifest_sha256']],cwd=ROOT,capture_output=True,text=True)
    receipt.update(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
    receipt['status']='SUBMITTED' if result.returncode==0 else 'SUBMISSION_FAILED_INSPECT_BEFORE_RETRY'
    if result.returncode==0:
        receipt['job_id']=result.stdout.strip().split(';')[0]
        assert receipt['job_id'].isdigit()
    write(record,receipt);print(json.dumps(receipt,indent=2))
    result.check_returncode()


if __name__=='__main__':main()
