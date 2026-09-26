"""Summarize real Qwen outputs and render the returned localization for review."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_palm_recovery/LVP_0049'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    diagnoses=json.loads((OUT/'diagnoses.json').read_text())
    correction=json.loads((OUT/'correction.json').read_text())
    counts={state:sum(d['state']==state for d in diagnoses) for state in ['normal','deformed','unclear']}
    lines=['# Qwen3.5 palm review','',
        'Official Qwen3.5-9B; palm-only adaptation of loose-v2; actual generated responses.',
        '',f'Area events: {manifest["event_frames"]}. Reviewed {len(diagnoses)} candidate frames.',
        f'Results: {counts}. These are VLM judgments, not independently verified labels.',
        '', '| Frame | Time (s) | State | Probability | Evidence |', '|---|---:|---|---:|---|']
    for d in diagnoses:
        text=str(d.get('evidence',d.get('parse_error',''))).replace('|','/').replace('\n',' ')
        lines.append(f'| {d["frame"]} | {d["time"]:.3f} | {d["state"]} | {d.get("probability", "unavailable")} | {text} |')
    lines+=['',f'Localization status: **{correction["status"]}**.',
            'Earliest positive means earliest among the reviewed frames; earlier unreviewed frames may already be deformed.',
            'A positive result does not validate coordinate accuracy or prove that SAM recovery will work.']
    if correction.get('proposal',{}).get('status')=='ok':
        f=correction['frame'];call=next(c for c in manifest['calls'] if c['frame']==f)
        im=Image.open(OUT/call['crop']).convert('RGB');draw=ImageDraw.Draw(im)
        x0,y0=call['box_xyxy'][:2]
        for i,((x,y),label) in enumerate(zip(correction['proposal']['points_xy'],correction['proposal']['labels'])):
            x-=x0;y-=y0;color='lime' if label else 'red'
            draw.ellipse((x-3,y-3,x+3,y+3),fill=color);draw.text((x+4,y),str(i),fill=color)
        im.resize((im.width*3,im.height*3)).save(OUT/'localization_review.png')
        lines+=['',f'Proposed correction frame: {f}.', '[Returned point locations](localization_review.png) (green positive, red negative).']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(counts,correction['status'])


if __name__=='__main__':main()
