"""Independently reconstruct every saved working raster from cached original bytes."""
import hashlib
from paper_material_demo import ROOT,load,sha,write
from prepare_component_masks import CONFIG


def main():
    from PIL import Image
    cfg=load(CONFIG);out=ROOT/cfg['run']/'inference';result=load(out/'results.json')
    assert result['status']=='COMPLETE' and result['protocol_sha256']==sha(CONFIG)
    data=ROOT/cfg['dataset'];uk=ROOT/cfg['uk_dataset']
    records={r['image_id']:(r,data) for r in load(data/'manifest.json')['images'] if r['split']=='dev'}
    records.update({r['image_id']:(r,uk) for r in load(uk/'manifest.json')['images']})
    checked=[]
    for entry in result['predictions']:
        p=out/entry['file'];assert sha(p)==entry['sha256'];d=load(p);r,source=records[d['image_id']]
        original=source/r['image_file'];assert sha(original)==r['sha256']==d['source_image_sha256']
        with Image.open(original) as image:
            rgb=image.convert('RGB');assert list(rgb.size)==d['source_size']
            scale=min(1.,cfg['inference']['long_side']/max(rgb.size))
            size=(round(rgb.width*scale),round(rgb.height*scale));assert list(size)==d['working_size']
            expected=rgb.resize(size,Image.Resampling.LANCZOS)
        input_file=p.parent/'input.png';assert sha(input_file)==d['input_sha256']
        with Image.open(input_file) as actual:
            assert actual.size==expected.size and actual.convert('RGB').tobytes()==expected.tobytes()
        checked.append({'image_id':r['image_id'],'working_pixel_sha256':hashlib.sha256(expected.tobytes()).hexdigest()})
    assert set(records)=={c['image_id'] for c in checked} and len(checked)==107
    write(out/'source_pixel_verification.json',{'status':'VERIFIED','results_sha256':sha(out/'results.json'),
                                               'protocol_sha256':sha(CONFIG),'checked':checked})
    print({'verified_original_to_working_images':len(checked)})


if __name__=='__main__':main()
