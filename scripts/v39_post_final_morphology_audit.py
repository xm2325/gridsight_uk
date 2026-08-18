from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ["POS_1283842", "POS_190181", "POS_291727", "POS_3778704", "POS_7060068"]
FINAL = ["POS_3437435", "POS_7561805"]


def read_rows(path: Path):
    rows=[]
    for line in path.read_text().splitlines():
        if line.strip():
            c,xc,yc,w,h=map(float,line.split()); rows.append((int(c),xc,yc,w,h))
    return rows


def features(rows):
    _,tcx,tcy,tw,th=next(r for r in rows if r[0]==0); ty1=tcy-th/2
    out=[]
    for c,cx,cy,w,h in rows:
        if c!=2: continue
        wr=w/tw; hr=h/th
        out.append({
            "abs_x_offset_over_tower_width":abs(cx-tcx)/tw,
            "y_from_tower_top_over_tower_height":(cy-ty1)/th,
            "width_over_tower_width":wr,
            "height_over_tower_height":hr,
            "relative_shape_ratio":hr/wr,
            "vector":[abs(cx-tcx)/tw,(cy-ty1)/th,math.log(wr),math.log(hr)]
        })
    return out


def q(x,p): return float(np.quantile(np.asarray(x,dtype=float),p))


def main():
    train_feats=[]; train_counts={}
    for source in TRAIN:
        rows=read_rows(ROOT/f"data/labels/train/{source}.txt"); f=features(rows); train_feats += f; train_counts[source]=len(f)
    X=np.asarray([x["vector"] for x in train_feats],dtype=float); mean=X.mean(0); cov=np.cov(X,rowvar=False); diag=np.diag(np.diag(cov)); covreg=.75*cov+.25*diag+np.eye(4)*1e-5; inv=np.linalg.inv(covreg)
    td=X-mean; train_d2=np.einsum('ni,ij,nj->n',td,inv,td); q95=float(np.quantile(train_d2,.95)); mx=float(train_d2.max())

    final={}
    for source in FINAL:
        rows=read_rows(ROOT/f"data/final_holdout/labels/{source}.txt"); f=features(rows); Y=np.asarray([x["vector"] for x in f],dtype=float); D=Y-mean; d2=np.einsum('ni,ij,nj->n',D,inv,D)
        final[source]={
            "n_insulators":len(f),
            "mahalanobis_d2":[float(x) for x in d2],
            "median_d2":float(np.median(d2)),
            "n_above_train_q95":int((d2>q95).sum()),
            "n_above_train_max":int((d2>mx).sum()),
            "width_over_tower_width_median":float(np.median([x['width_over_tower_width'] for x in f])),
            "height_over_tower_height_median":float(np.median([x['height_over_tower_height'] for x in f])),
            "relative_shape_ratio_median":float(np.median([x['relative_shape_ratio'] for x in f])),
        }

    report={
      "evidence_type":"post-final-holdout-morphology-shift-audit",
      "analysis_only":True,
      "must_not_be_used_to_retune_v3_8_final_evidence":True,
      "training":{
        "n_sources":len(TRAIN),"n_insulators":len(train_feats),"per_source":train_counts,
        "mahalanobis_d2_q95":q95,"mahalanobis_d2_max":mx,
        "width_over_tower_width_median":float(np.median([x['width_over_tower_width'] for x in train_feats])),
        "height_over_tower_height_median":float(np.median([x['height_over_tower_height'] for x in train_feats])),
        "relative_shape_ratio_median":float(np.median([x['relative_shape_ratio'] for x in train_feats])),
      },
      "final_holdout":final,
      "interpretation":{
        "POS_3437435":"Strong morphology shift: all six reference insulators lie beyond the maximum training geometry distance; wide horizontal/strain-like strings violate the narrow training prior.",
        "POS_7561805":"Geometry is much closer to the training distribution; only a small tail exceeds the training q95, so remaining errors are not explained by morphology shift alone.",
        "system_lesson":"The text-prompt foundation model retained broad recall, while the hand-engineered training-distribution prior overfit tower/component morphology. Future model development requires morphology-diverse labels and a new preregistered holdout, not tuning on these final images."
      }
    }
    out=ROOT/'reports/v3_9_morphology_shift.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    md=ROOT/'reports/v3_9_morphology_shift.md'
    e=final['POS_3437435'];s=final['POS_7561805'];tr=report['training']
    md.write_text(f"""# v3.9 Post-final morphology-shift audit\n\n**Analysis only. These findings must not be used to retune the v3.8 final-holdout result.**\n\n## Training geometry\n\n- 5 training towers / {tr['n_insulators']} insulator references.\n- Training Mahalanobis d² q95: **{tr['mahalanobis_d2_q95']:.3f}**; maximum: **{tr['mahalanobis_d2_max']:.3f}**.\n- Median width/tower-width: **{tr['width_over_tower_width_median']:.3f}**.\n- Median height/tower-height: **{tr['height_over_tower_height_median']:.3f}**.\n- Median relative shape ratio `(h/tower_h)/(w/tower_w)`: **{tr['relative_shape_ratio_median']:.3f}**.\n\n## Frozen final holdout\n\n### POS_3437435 — England\n\n- 6/6 insulators exceed training q95; **{e['n_above_train_max']}/6 exceed even the maximum training d²**.\n- Median d²: **{e['median_d2']:.3f}**.\n- Median width/tower-width: **{e['width_over_tower_width_median']:.3f}**.\n- Median height/tower-height: **{e['height_over_tower_height_median']:.3f}**.\n- Relative shape ratio: **{e['relative_shape_ratio_median']:.3f}**.\n\nThis is a substantial morphology shift toward wide, horizontally oriented/strain-like insulators. It explains why the frozen geometry prior removed correct YOLOE candidates.\n\n### POS_7561805 — Scotland\n\n- {s['n_above_train_q95']}/6 exceed training q95; {s['n_above_train_max']}/6 exceed training maximum.\n- Median d²: **{s['median_d2']:.3f}**.\n- Median width/tower-width: **{s['width_over_tower_width_median']:.3f}**.\n- Median height/tower-height: **{s['height_over_tower_height_median']:.3f}**.\n- Relative shape ratio: **{s['relative_shape_ratio_median']:.3f}**.\n\nThis tower is much closer to the training geometry distribution.\n\n## Engineering conclusion\n\nThe independent holdout falsified the assumption that a narrow geometry prior learned from five similar towers would generalise across UK tower designs. The open-vocabulary YOLOE text model preserved recall much better; the post-processing prior created the larger generalisation failure. The correct next phase is a new morphology-diverse acquisition/training cycle with a **new preregistered holdout**, not tuning against these two final images.\n""")
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
