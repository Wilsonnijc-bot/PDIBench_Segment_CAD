"""Regenerate the requested Cosmos3_0048 replay from its recorded prompts."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

from pathlib import Path
import cv2
import numpy as np
from robot.preprocessing.link7_persistent.v1_mask import predictor

ROOT=(_workspace_root())
VIDEO=Path('/root/autodl-tmp/hierarchical-deformation-gdrive/15sRTmHwkBQO0FmDRnp0BePAArCad2_Gj/0048.mp4')


def main():
    cap=cv2.VideoCapture(str(VIDEO));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(bgr)
    cap.release();h,w=frames[0].shape[:2]
    points=np.array([[694.4,259.616],[701.6,263.048],[689.6,233.876],[875,70]])
    p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=str(VIDEO),offload_video_to_cpu=True))['session_id']
    masks=np.zeros((len(frames),h,w),bool)
    try:
        d=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=10,obj_id=0,points=(points/[w,h]).tolist(),point_labels=[1,1,1,0]))['outputs']
        ids=list(d['out_obj_ids']);masks[10]=d['out_binary_masks'][ids.index(0)]
        for direction in ['forward','backward']:
            for r in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction=direction,start_frame_index=10)):
                ids=list(r['outputs']['out_obj_ids'])
                if 0 in ids:masks[int(r['frame_index'])]=r['outputs']['out_binary_masks'][ids.index(0)]
    finally:p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()
    out=(_workspace_root() / 'qwen_results/Cosmos3_0048');out.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out/'masking_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h))
    for image,mask in zip(frames,masks):
        overlay=image.copy();overlay[mask]=(.55*image[mask]+.45*np.array([255,190,0])).astype(np.uint8)
        writer.write(np.concatenate([image,overlay],axis=1))
    writer.release();print('Replay complete',len(frames),flush=True)


if __name__=='__main__':main()
