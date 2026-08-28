"""Classifier decisions shared by the GPU runner and independent verifier."""
import math

def decide(tight, context, box, cfg):
    names=cfg['classes']
    if len(tight)!=len(names) or len(context)!=len(names) or not all(math.isfinite(x) for x in [*tight,*context,*box]):
        raise ValueError('Invalid or non-finite material decision input')
    a=max(range(len(tight)),key=tight.__getitem__)
    b=max(range(len(context)),key=context.__getitem__)
    margins=[sorted(v,reverse=True)[0]-sorted(v,reverse=True)[1] for v in [tight,context]]
    w,h=box[2]-box[0],box[3]-box[1]
    if min(w,h)<cfg['minimum_native_side'] or w*h<cfg['minimum_native_area']:
        label,reason='unknown','insufficient native pixels'
    elif a!=b:
        label,reason='unknown','tight/context disagreement'
    elif a==2:
        label,reason='unknown','other/background classification'
    elif min(margins)<cfg['rejection']['minimum_logit_margin']:
        label,reason='unknown','small uncalibrated logit margin'
    else:
        label,reason=names[a],'provisional supervised material; not calibrated or expert verified'
    return {'material':label,'reason':reason,'tight_argmax':names[a],'context_argmax':names[b],
            'logit_margins':margins,'material_verified':False,'scores_are_probabilities':False}
