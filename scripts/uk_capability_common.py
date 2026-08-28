"""Geometry and explicit abstention for unvalidated material diagnostics."""
import math


def crop_extent(box,width,height,padding=0.):
    x1,y1,x2,y2=box;px=(x2-x1)*padding;py=(y2-y1)*padding
    return [max(0,math.floor(x1-px)),max(0,math.floor(y1-py)),
            min(width,math.ceil(x2+px)),min(height,math.ceil(y2+py))]


def material_quality(box,cfg):
    w=box[2]-box[0];h=box[3]-box[1]
    return min(w,h)>=cfg['minimum_native_short_side'] and w*h>=cfg['minimum_native_area']


def diagnostic_decision(tight,context,labels):
    a=max(range(len(labels)),key=lambda i:tight[i]);b=max(range(len(labels)),key=lambda i:context[i])
    if a!=b:reason='crop_context_disagreement'
    elif labels[a] not in ['glass','porcelain','polymer']:reason='non_insulator_hypothesis'
    else:reason='uncalibrated_no_target_validation'
    return {'material':'unknown','accepted':False,'reason':reason,
            'tight_hypothesis':labels[a],'context_hypothesis':labels[b],
            'scores_are_probabilities':False}
