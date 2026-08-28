"""Raster agreement and unscored pole-end geometry, with no model execution."""
import math


def decode_masks(raw):
    import numpy as np
    n,h,w=map(int,raw['mask_shape'])
    if min(n,h,w)<0 or h*w>1280*1280 or n>100:raise ValueError('Unexpected mask dimensions')
    bits=raw['mask_bits']
    if bits.shape!=(n,math.ceil(h*w/8)):raise ValueError('Packed mask size mismatch')
    return np.unpackbits(bits,axis=1,count=h*w).reshape((n,h,w)).astype(bool)


def raster_polygon(points,source_size,working_size):
    import numpy as np
    from PIL import Image,ImageDraw
    w,h=working_size;sw,sh=source_size
    canvas=Image.new('1',(w,h))
    ImageDraw.Draw(canvas).polygon([(p['x']*w/sw,p['y']*h/sh) for p in points],fill=1)
    return np.asarray(canvas,dtype=bool)


def rectangle_mask(box,size):
    import numpy as np
    w,h=size;m=np.zeros((h,w),dtype=bool)
    x1,y1=max(0,math.floor(box[0])),max(0,math.floor(box[1]))
    x2,y2=min(w,math.ceil(box[2])),min(h,math.ceil(box[3]))
    m[y1:y2,x1:x2]=True
    return m


def mask_iou(a,b):
    import numpy as np
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else 0.


def mask_matches(predictions,masks,references,reference_masks,threshold=.25,iou_threshold=.5):
    used=set();matches=[];false=[]
    for p in sorted((p for p in predictions if p['score']>=threshold),key=lambda p:-p['score']):
        index=p['prediction_index'];candidates=[(mask_iou(masks[index],m),j) for j,(r,m) in enumerate(zip(references,reference_masks))
                                                  if r['class_id']==p['class_id'] and j not in used]
        score,j=max(candidates,default=(0.,-1))
        if j>=0 and score>=iou_threshold:
            used.add(j);matches.append({'prediction_index':index,'reference_index':j,'class_id':p['class_id'],'iou':score})
        else:false.append(index)
    return {'matches':matches,'false_predictions':false,'missed_references':[j for j in range(len(references)) if j not in used],
            'tp':len(matches),'fp':len(false),'fn':len(references)-len(used)}


def pole_end_candidate(mask,component_boxes):
    """Choose an axis end near components, or abstain; never a supervised tip label."""
    import numpy as np
    result={'status':'unknown','reason':'insufficient mask support','point':None,
            'derived':True,'score':None,'supervised_pole_top':False}
    y,x=np.nonzero(mask)
    if len(x)<32:return result
    points=np.column_stack([x,y]).astype(float);centre=points.mean(0)
    covariance=np.cov(points-centre,rowvar=False);values,vectors=np.linalg.eigh(covariance)
    if values[-1]<4*max(values[0],1e-8):
        return dict(result,reason='mask axis is not sufficiently elongated')
    axis=vectors[:,-1];projected=(points-centre)@axis
    low,high=np.quantile(projected,[.01,.99]);length=high-low
    ends=[np.median(points[projected<=low],axis=0),np.median(points[projected>=high],axis=0)]
    result['axis_ends']=[e.tolist() for e in ends]
    if length<32 or not component_boxes:return dict(result,reason='no reliable attached-component context')
    centres=np.array([[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in component_boxes])
    distance=[float(np.linalg.norm(centres-e,axis=1).min()) for e in ends]
    chosen=int(np.argmin(distance));near=distance[chosen];far=distance[1-chosen]
    result['component_distances']=distance
    if near>.35*length or far<max(2*near,near+.1*length):
        return dict(result,reason='ambiguous end-to-component association')
    point=ends[chosen];h,w=mask.shape
    if min(point[0],point[1],w-1-point[0],h-1-point[1])<=2:
        return dict(result,reason='candidate end is truncated at the working image edge')
    return dict(result,status='geometry_candidate',reason='axis endpoint nearest predicted components; physical tip unverified',point=point.tolist())
