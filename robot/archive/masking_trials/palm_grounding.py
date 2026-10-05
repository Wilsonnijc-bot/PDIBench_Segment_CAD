"""Short single-point Qwen grounding calls, retaining each raw answer."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from persistent_masking.palm_recovery import json_object, PALM_FOCUS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--black-edges',action='store_true',help='Two distinct black palm edge interiors and one white palm positive')
    parser.add_argument('--three-positive-only',action='store_true',help='Only run the retained three positive queries; omit old negative queries')
    parser.add_argument('--frame',type=int,help='Explicit additional frame for localization experiments; not a new diagnosis')
    parser.add_argument('--anatomical-edges',action='store_true',help='Target the black housing below the collar; avoid wrist negatives')
    parser.add_argument('--literal',action='store_true',help='Simple single-image requests, three positives only')
    args=parser.parse_args();out=args.output
    from generation.link_crop_wrapper.run_segment_vlm import load_model, generate
    manifest=json.loads((out/'manifest.json').read_text())
    diagnoses=json.loads((out/'diagnoses.json').read_text())
    good=[d for d in diagnoses if d['state']=='deformed' and not d['parse_error']]
    if not good:raise ValueError('No confirmed positive for grounding')
    first=min(good,key=lambda d:d['frame']);box=first['box_xyxy']
    if args.frame is not None:
        first=next(c for c in manifest['calls'] if c['frame']==args.frame);box=first['box_xyxy']
    processor,model,architecture=load_model(args.model)
    tasks=[('palm_dark',1,'the interior of deformed DARK PALM HOUSING material, not the wrist, fingers, shadow or held object'),
           ('palm_white',1,'the interior of surviving WHITE PALM HOUSING material, not an articulated finger'),
           ('held_object',0,'the interior of the held object, not any robot material'),
           ('wrist',0,'the interior of the wrist joint housing ABOVE the palm, not the palm')]
    if args.black_edges:
        tasks=[('black_edge_left',1,'the BLACK PALM material just INSIDE its left visible boundary, below the metallic wrist sleeve. Stay a few image pixels INSIDE the black material, never on the outline, background, white patch, wrist or articulated finger'),
               ('black_edge_right',1,'the BLACK PALM material just INSIDE a DIFFERENT visible boundary on its right side near the white palm remnant. Stay a few pixels INSIDE black palm material, not white material, wrist, finger or background'),
               ('palm_white',1,'the interior of surviving WHITE PALM HOUSING material, not an articulated finger'),
               ('held_object',0,'the interior of the held object, not any robot material'),
               ('wrist',0,'the interior of the wrist joint housing ABOVE the palm, not the palm')]
    if args.anatomical_edges:
        tasks=[('black_edge_left',1,'the broad BLACK PALM HOUSING immediately BELOW the shiny metallic wrist collar, near its left outer side. Choose black surface ABOVE the fingers and ABOVE the white hanging remnant, not the junction on the white remnant. Place the point a few pixels inward from the black outline'),
               ('black_edge_right',1,'the broad BLACK PALM HOUSING immediately BELOW the shiny metallic wrist collar, near its right outer side. Choose black surface ABOVE the fingers and WHITE hanging remnant. The target is the opposite side of the BLACK housing, not the white panel or metallic collar'),
               ('palm_white',1,'the center of the surviving WHITE PALM remnant connected to the black housing. This is the white body being tracked from Image 1; do not select the metallic wrist'),
               ('finger_tip',0,'a confidently identifiable NARROW articulated finger TIP below the black palm housing and beside the held object. Exclude the broad white palm remnant from this finger-tip choice. If finger/palm identity is ambiguous, return visible false'),
               ('held_object',0,'the center of the held object, well away from robot material')]
    if args.literal:
        tasks=[('black_left',1,'In Image 2, point just inside the left edge of the black palm.'),
               ('black_right',1,'In Image 2, point just inside the right edge of the black palm.'),
               ('white_palm',1,'In Image 2, point inside the white palm.')]
    if args.three_positive_only:
        tasks=[task for task in tasks if task[1]==1]
    records=[];points=[];labels=[]
    masks=np.load(manifest['masks'])['masks'];h,w=masks.shape[1:]
    for name,label,description in tasks:
        request=f'''Image 1 is the initial reference. Image 2 is the candidate.
In IMAGE 2, locate ONE point strictly inside {description}.
Coordinates use integers from 0 to 1000: x runs left-to-right and y top-to-bottom
over IMAGE 2. Return only {{"visible":true,"point":[x,y]}}.
If you cannot confidently identify that material, return {{"visible":false,"point":null}}.
Return ONE point only. No list of points, explanation, or additional fields.'''
        if (args.black_edges or args.anatomical_edges) and name=='black_edge_right' and records[0].get('status')=='ok':
            prior=(np.array(records[0]['source_xy'])-box[:2])/[box[2]-box[0],box[3]-box[1]]*1000
            request+=f'\nThe other black-palm point is already at {np.round(prior).astype(int).tolist()}. Select a spatially distinct location, not the same point.'
        with Image.open(out/'reference.png') as ref,Image.open(out/first['crop']) as candidate:
            messages=[dict(role='system',content=[dict(type='text',text=PALM_FOCUS)]),dict(role='user',content=[
                dict(type='text',text='Image 1: initial palm reference.'),dict(type='image',image=ref.convert('RGB')),
                dict(type='text',text='Image 2: candidate for point localization.'),dict(type='image',image=candidate.convert('RGB')),
                dict(type='text',text=request)])]
            if args.literal:
                request=description+' Return one point_2d in JSON. Coordinates: 0 to 1000.'
                messages=[dict(role='system',content=[dict(type='text',text='Image 1 shows the original palm. Locate the same part in Image 2. Return JSON only.')]),
                    dict(role='user',content=[dict(type='image',image=ref.convert('RGB')),dict(type='image',image=candidate.convert('RGB')),dict(type='text',text=request)])]
            raw=generate(model,processor,messages,100)
        record=dict(target=name,label=label,prompt=request,raw=raw)
        try:
            d=json_object(raw)
            if 'point_2d' in d and set(d)<= {'point_2d','label'}:
                d=dict(visible=d['point_2d'] is not None,point=d['point_2d'])
            if set(d)!={'visible','point'} or not isinstance(d['visible'],bool):raise ValueError('Invalid fields')
            if not d['visible']:
                if d['point'] is not None:raise ValueError('Invisible point must be null')
                record['status']='not_localized'
            else:
                xy=np.asarray(d['point'],dtype=float)
                if xy.shape!=(2,) or not np.isfinite(xy).all() or (xy<0).any() or (xy>1000).any():raise ValueError('Invalid point')
                xy=xy/1000*np.array([box[2]-box[0],box[3]-box[1]])+box[:2]
                if xy[0]<0 or xy[1]<0 or xy[0]>=w or xy[1]>=h:raise ValueError('Point in padding')
                points.append(xy.tolist());labels.append(label);record.update(status='ok',source_xy=xy.tolist())
        except (ValueError,TypeError,KeyError) as e:record.update(status='invalid',error=str(e))
        records.append(record);print(record,flush=True)
        (out/'grounding_calls.json').write_text(json.dumps(records,indent=2))
    required_positives=3 if args.black_edges or args.anatomical_edges or args.literal else 2
    ready=sum(labels)==required_positives and len({tuple(p) for p in points})==len(points)
    positives=np.asarray([xy for xy,label in zip(points,labels) if label])
    if ready and (args.black_edges or args.anatomical_edges):
        ready=bool(np.linalg.norm(positives[0]-positives[1])>=5)
    result=dict(architecture=architecture,frame=first['frame'],grounding_mode='literal' if args.literal else ('anatomical_edges' if args.anatomical_edges else ('black_edges' if args.black_edges else 'interior')),status='ready_for_sam' if ready else 'localization_failed',
        binary_list=[dict(frame=d['frame'],deformed=None if d['state']=='unclear' else d['state']=='deformed') for d in diagnoses],
        proposal=dict(status='ok' if ready else 'unclear',points_xy=points,labels=labels,
                      evidence='Separate Qwen point queries; geometry validation only, not semantic ground truth'))
    (out/'correction.json').write_text(json.dumps(result,indent=2))
    im=Image.open(out/first['crop']).convert('RGB');draw=ImageDraw.Draw(im)
    for i,(xy,label) in enumerate(zip(points,labels)):
        x,y=np.array(xy)-box[:2];color='lime' if label else 'red'
        draw.ellipse((x-3,y-3,x+3,y+3),fill=color);draw.text((x+4,y),('P' if label else 'N')+str(i),fill=color)
    im.resize((im.width*3,im.height*3)).save(out/'grounding_points.png')
    print(result['status'],flush=True)


if __name__=='__main__':main()
