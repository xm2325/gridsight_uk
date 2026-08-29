"""Decision and metric helpers for the frozen four-class material head."""
import math


def margin(values):
    ordered=sorted(values,reverse=True)
    return ordered[0]-ordered[1]


def decide_v2(tight_logits,context_logits,tight_similarity,context_similarity,box,cfg,thresholds):
    names=cfg['classes'];values=[*tight_logits,*context_logits,*tight_similarity,*context_similarity,*box]
    if len(tight_logits)!=len(names) or len(context_logits)!=len(names) or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Invalid or non-finite v2 material decision input')
    tight=max(range(len(names)),key=tight_logits.__getitem__);context=max(range(len(names)),key=context_logits.__getitem__)
    margins=[margin(tight_logits),margin(context_logits)];w,h=box[2]-box[0],box[3]-box[1]
    reasons=[]
    if min(w,h)<cfg['minimum_native_side'] or w*h<cfg['minimum_native_area']:reasons.append('insufficient native pixels')
    if tight!=context:reasons.append('tight/context disagreement')
    predicted=tight
    if tight==context and names[predicted]=='other':reasons.append('other/background classification')
    if tight==context:
        if min(margins)<thresholds['margin'][predicted]:reasons.append('below development-derived logit-margin threshold')
        if min(tight_similarity[predicted],context_similarity[predicted])<thresholds['similarity'][predicted]:
            reasons.append('outside development embedding support')
    material='unknown' if reasons else names[predicted]
    return {'material':material,'reasons':reasons or ['passes fixed v2 diagnostic gate'],
      'tight_argmax':names[tight],'context_argmax':names[context],'logit_margins':margins,
      'predicted_class_similarity':[tight_similarity[predicted],context_similarity[predicted]],
      'thresholds':{'margin':thresholds['margin'][predicted],'similarity':thresholds['similarity'][predicted]},
      'material_verified':False,'scores_are_probabilities':False}


def diagnostic_counts(records):
    material=[r for r in records if r['expected_material']!='other']
    accepted=[r for r in material if r['decision']['material']!='unknown']
    correct=[r for r in accepted if r['decision']['material']==r['expected_material']]
    other=[r for r in records if r['expected_material']=='other']
    return {'material_targets':len(material),'accepted_material_targets':len(accepted),'correct_accepted_material_targets':len(correct),
      'coverage':len(accepted)/len(material) if material else None,
      'accepted_accuracy':len(correct)/len(accepted) if accepted else None,
      'other_targets':len(other),'other_false_accepts':sum(r['decision']['material']!='unknown' for r in other)}
