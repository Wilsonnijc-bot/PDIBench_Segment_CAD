"""Precision grounding: choose a few numbered image locations, not free coordinates."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from persistent_masking.v1_mask import spread
from persistent_masking.palm_recovery import PALM_FOCUS,json_object

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results_palm_recovery/anatomical_edges'


def prepare(args):
    masks=np.load(ROOT/'reg2inv_results_v1/LVP_0049/guided_masks.npz')['masks']
    wrists=np.load(ROOT/'reg2inv_results/LVP_0049/masks/links.npz')['object_masks'][:,4]
    for f in [13,20,28,36]:
        out=BASE/f'frame_{f:05d}'/args.variant;out.mkdir(exist_ok=True)
        parent=out.parent;m=json.loads((parent/'manifest.json').read_text());box=m['calls'][0]['box_xyxy']
        rgb=np.asarray(Image.open(parent/f'frame_{f:05d}.png').convert('RGB'))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        y,x=np.where(masks[f]);width=x.max()-x.min()+1;height=y.max()-y.min()+1
        yy,xx=np.indices(gray.shape);sx=xx+box[0];sy=yy+box[1]
        roi=(sx>=x.min()-width*.8)&(sx<=x.max()+width*.5)&(sy>=y.min()-height*1.2)&(sy<=y.max()+height*.3)
        # Image evidence only creates candidate locations, never the final SAM mask.
        dark=roi&(gray<75)
        if args.exclude_wrist:
            wrist=np.asarray(Image.fromarray(wrists[f].astype(np.uint8)).crop(box)).astype(bool)
            wrist=cv2.dilate(wrist.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
            dark &= ~wrist & (sy<y.min()+height*.25)
        dark=cv2.erode(dark.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        white=roi&(gray>125)&(sx>=x.min())&(sx<=x.max())&(sy>=y.min())&(sy<=y.max())
        white=cv2.erode(white.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        candidates=[]
        for category,mask,n in [('dark',dark,14),('white',white,4)]:
            if mask.any():
                for xy in spread(mask,n):candidates.append(dict(id=len(candidates)+1,kind=category,crop_xy=xy.tolist(),source_xy=(xy+box[:2]).tolist()))
        image=Image.fromarray(rgb).resize((rgb.shape[1]*4,rgb.shape[0]*4))
        draw=ImageDraw.Draw(image);font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
        for c in candidates:
            px,py=np.asarray(c['crop_xy'])*4
            draw.ellipse((px-3,py-3,px+3,py+3),fill='red')
            text=str(c['id']);bb=draw.textbbox((px+4,py-9),text,font=font);draw.rectangle(bb,fill='white');draw.text((px+4,py-9),text,fill='black',font=font)
        image.save(out/'numbered.png')
        (out/'candidates.json').write_text(json.dumps(candidates,indent=2))
        m['candidate_policy']=dict(exclude_wrist=args.exclude_wrist,black_upper_region_only=args.exclude_wrist,variant=args.variant)
        (out/'manifest.json').write_text(json.dumps(m,indent=2))


def review(args):
    from generation.link_crop_wrapper.run_segment_vlm import load_model,generate
    processor,model,architecture=load_model(args.model)
    for f in [13,20,28,36]:
        parent=BASE/f'frame_{f:05d}';out=parent/args.variant
        candidates=json.loads((out/'candidates.json').read_text());byid={c['id']:c for c in candidates}
        dark=[c['id'] for c in candidates if c['kind']=='dark'];white=[c['id'] for c in candidates if c['kind']=='white']
        prompt=f'''Image 1 is the original unmarked candidate frame. Image 2 is the SAME image with numbered red-dot locations.
Identify the broad BLACK DEFORMED PALM housing directly below the shiny metallic wrist collar, including its sides, and the WHITE palm remnant attached below/right.
Choose exactly TWO numbered dots in BLACK PALM material from {dark}, and ONE in WHITE PALM material from {white}.
Exclude the metallic collar, wrist, background surfaces and their shadows, held objects, and thin finger tips.
The two black points must be on different areas of the broad black housing, not both at the junction with white material.
Return only {{"black_ids":[id1,id2],"white_id":id3}}. If no valid distinct points exist, return {{"black_ids":[],"white_id":null}}.
Use printed IDs only. Do not invent coordinates. Each label belongs to the nearby red dot.'''
        with Image.open(parent/f'frame_{f:05d}.png') as raw,Image.open(out/'numbered.png') as marked:
            messages=[dict(role='system',content=[dict(type='text',text='You select precise image locations for segmenting the gripper palm. Select only locations whose anatomical identity is clear. Output ONLY a JSON object with black_ids and white_id. No analysis, numbered explanation, or markdown. Start the response with { and end with }.')]),
                dict(role='user',content=[dict(type='image',image=raw.convert('RGB')),dict(type='image',image=marked.convert('RGB')),dict(type='text',text=prompt)])]
            response=generate(model,processor,messages,350)
        result=dict(frame=f,architecture=architecture,status='localization_failed',raw=response,prompt=prompt)
        try:
            d=json_object(response);ids=d['black_ids']+[d['white_id']]
            if set(d)!={'black_ids','white_id'} or len(d['black_ids'])!=2 or len(set(ids))!=3:raise ValueError('Need three distinct IDs')
            if any(type(i)!=int or i not in byid for i in ids):raise ValueError('Invalid IDs')
            if any(i not in dark for i in ids[:2]) or ids[2] not in white:raise ValueError('Wrong color class')
            points=[byid[i]['source_xy'] for i in ids]
            result.update(status='ready_for_sam',chosen_ids=ids,proposal=dict(status='ok',points_xy=points,labels=[1,1,1],
                evidence='Qwen selected numbered image candidates; class and geometry checked, semantic accuracy requires review'))
        except (ValueError,TypeError,KeyError) as e:result['error']=str(e)
        (out/'correction.json').write_text(json.dumps(result,indent=2))
        print(f,result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','review']);p.add_argument('--model',type=Path)
    p.add_argument('--variant',default='candidates');p.add_argument('--exclude-wrist',action='store_true')
    args=p.parse_args();prepare(args) if args.stage=='prepare' else review(args)
