#!/usr/bin/env python3
"""Build a browser replay of a case's base and persistent link7 masks."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path


RESULT_ROOT = ((_workspace_root() / 'results/selected-45-v1'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npy_header(stream) -> dict:
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("invalid NumPy array")
    major, minor = stream.read(2)
    if major == 1:
        header_size = int.from_bytes(stream.read(2), "little")
    elif major in (2, 3):
        header_size = int.from_bytes(stream.read(4), "little")
    else:
        raise ValueError(f"unsupported NumPy array version {major}.{minor}")
    return ast.literal_eval(stream.read(header_size).decode("latin1"))


def object_names(archive: zipfile.ZipFile) -> list[str]:
    with archive.open("object_names.npy") as stream:
        header = npy_header(stream)
        dtype = header["descr"]
        if not dtype.startswith("<U") or header["fortran_order"]:
            raise ValueError("unexpected object-name encoding")
        width = int(dtype[2:])
        count, = header["shape"]
        data = stream.read(count * width * 4)
        return [data[i * width * 4:(i + 1) * width * 4]
                .decode("utf-32-le").rstrip("\0") for i in range(count)]


def horizontal_runs(mask: bytes, height: int, width: int) -> list[list[int]]:
    runs = []
    for y in range(height):
        row = mask[y * width:(y + 1) * width]
        x = row.find(b"\x01")
        while x != -1:
            end = row.find(b"\x00", x)
            if end == -1:
                end = width
            runs.append([y, x, end])
            x = row.find(b"\x01", end)
    return runs


def read_masks(base: Path, refined: Path) -> tuple[tuple[int, int, int], list, list, int]:
    with zipfile.ZipFile(base) as old_zip, zipfile.ZipFile(refined) as new_zip:
        names = object_names(old_zip)
        if names != object_names(new_zip) or names != [f"link{i}" for i in range(2, 8)]:
            raise ValueError(f"unexpected six-link order: {names}")
        with old_zip.open("object_masks.npy") as old, new_zip.open("object_masks.npy") as new:
            old_header, new_header = npy_header(old), npy_header(new)
            if old_header != new_header or old_header["descr"] not in ("|b1", "?"):
                raise ValueError("base and refined mask arrays differ in shape or type")
            if old_header["fortran_order"]:
                raise ValueError("Fortran-order masks are unsupported")
            frames, objects, height, width = old_header["shape"]
            if objects != 6:
                raise ValueError("expected six link masks")
            plane_size = height * width
            base_runs, refined_runs = [], []
            changed_frames = 0
            for frame in range(frames):
                for index in range(objects):
                    old_mask = old.read(plane_size)
                    new_mask = new.read(plane_size)
                    if len(old_mask) != plane_size or len(new_mask) != plane_size:
                        raise ValueError(f"truncated mask at frame {frame}")
                    if index < 5:
                        if old_mask != new_mask:
                            raise ValueError(f"link{index + 2} changed at frame {frame}")
                    else:
                        changed_frames += old_mask != new_mask
                        base_runs.append(horizontal_runs(old_mask, height, width))
                        refined_runs.append(horizontal_runs(new_mask, height, width))
            if old.read(1) or new.read(1):
                raise ValueError("mask array contains trailing data")
    return (frames, height, width), base_runs, refined_runs, changed_frames


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>COSMOS3_0015 · link7 mask replay</title>
<style>
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#101719;color:#e7f0eb}
*{box-sizing:border-box}body{margin:0}main{max-width:1420px;margin:auto;padding:28px 28px 52px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:18px;flex-wrap:wrap}
h1{font-size:clamp(26px,3vw,42px);line-height:1.12;margin:6px 0 15px;letter-spacing:-.035em}
.eyebrow{font-size:12px;letter-spacing:.16em;color:#8bb6a4;text-transform:uppercase}
.notice{border-left:4px solid #54d4a4;background:#1b302a;padding:15px 18px;margin:12px 0 22px}
.notice strong{color:#a7f2cc}.notice p{margin:4px 0 0;color:#d6e8dd}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:18px}
.stage,.side{background:#182225;border:1px solid #344548;border-radius:8px;overflow:hidden}
.screen{position:relative;width:100%;aspect-ratio:__WIDTH__/__HEIGHT__;background:#050909}
video,canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
canvas{pointer-events:none}.divider{position:absolute;top:0;bottom:0;width:2px;background:#fff9;pointer-events:none}
.divider:after{content:'';position:absolute;top:12px;left:-30px;width:60px;text-align:center;background:#102326d9;border:1px solid #fff9;border-radius:3px;color:white;font-size:11px;padding:3px;content:'DRAG SPLIT'}
.transport{padding:14px 18px 16px;border-top:1px solid #344548;display:flex;align-items:center;gap:13px;flex-wrap:wrap}
button,select{background:#223c38;border:1px solid #5c8277;border-radius:4px;color:#edfff6;padding:8px 12px;font:inherit;cursor:pointer}
button:hover,button[aria-pressed=true]{background:#35685a}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid #f8cf76;outline-offset:2px}
input[type=range]{accent-color:#69dbab;cursor:pointer}.frame-slider{flex:1;min-width:180px}
.readout{font-variant-numeric:tabular-nums;color:#b8ccc3;min-width:130px;text-align:right}
.side{padding:18px}.side h2{margin:0 0 12px;font-size:17px}.mode{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:20px}
.mode button{font-size:13px}.control{display:block;margin:16px 0}.control span{display:flex;justify-content:space-between;font-size:13px;color:#b7cac1}.control input{width:100%;margin-top:9px}
.legend{display:grid;gap:9px;font-size:13px;color:#c7d8cf;margin:18px 0}.swatch{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:8px;vertical-align:-2px}.base{background:#ff7767}.refined{background:#45e6a3}
.side p{font-size:13px;color:#a9bcb4}.links{display:grid;gap:9px;margin-top:20px}a{color:#8edfc0}.foot{color:#99ada5;font-size:13px;margin:17px 2px 0}
@media(max-width:970px){.layout{grid-template-columns:1fr}.side{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 20px}.side h2,.side .links{grid-column:1/-1}}
@media(max-width:600px){main{padding:18px 12px 32px}.side{display:block}.transport{padding:12px}.readout{min-width:0}}
</style><script src="masks.js"></script></head><body><main>
<header><div><div class="eyebrow">Selected 45 · V1 · COSMOS3_0015</div><h1>Link7 mask replay</h1></div><div class="eyebrow">189 source frames · 24 fps</div></header>
<div class="notice" role="status"><strong>Masking fix applied</strong><p>The validated persistent link7 mask replaced the fresh SAM3 base mask in all __CHANGED_FRAMES__ of __FRAMES__ frames; __CHANGED_PIXELS__ pixels differ. Links 2–6 were preserved. The green mask is the one used for V1 scoring.</p></div>
<div class="layout"><section class="stage" aria-label="Interactive mask replay">
<div class="screen" id="screen"><video id="video" src="../v1/replay/interactive_exact-group/source.mp4" preload="metadata" playsinline></video><canvas id="mask" width="__WIDTH__" height="__HEIGHT__"></canvas><div class="divider" id="divider"></div></div>
<div class="transport"><button id="play" type="button">Play</button><input class="frame-slider" id="frame" type="range" min="0" max="__LAST_FRAME__" value="0" aria-label="Video frame"><span class="readout" id="readout">Frame 1 / __FRAMES__</span><label>Speed <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label></div>
</section><aside class="side"><h2>Inspect the correction</h2><div class="mode" role="group" aria-label="Mask view"><button type="button" data-mode="split" aria-pressed="true">Split comparison</button><button type="button" data-mode="base" aria-pressed="false">Base SAM3</button><button type="button" data-mode="refined" aria-pressed="false">Persistent V1</button></div>
<label class="control"><span><span>Split position</span><output id="split-value">50%</output></span><input id="split" type="range" min="0" max="100" value="50"></label>
<label class="control"><span><span>Mask opacity</span><output id="opacity-value">58%</output></span><input id="opacity" type="range" min="0" max="100" value="58"></label>
<div class="legend"><div><span class="swatch base"></span>Left: original six-link SAM3 link7</div><div><span class="swatch refined"></span>Right: persistent replacement used in V1</div></div>
<p>Play, scrub, or pause at any frame. Drag the split to compare the two masks on the same source frame. Validation checked full-video coverage and provenance; use this view to judge the mask visually.</p>
<div class="links"><a href="../v1/replay/interactive_exact-group/link7_exact-group.html">Open link7 rigidity and point-pair replay</a><a href="../refined_segmentation.json">Mask merge provenance</a></div></aside></div>
<p class="foot">Masks are the exact link7 planes in <code>base_segmentation.npz</code> and <code>refined_segmentation.npz</code>. Frame 0 corresponds to the source video’s first frame.</p>
</main><script>
const data=window.MASK_REPLAY,video=document.getElementById('video'),canvas=document.getElementById('mask'),ctx=canvas.getContext('2d');
const frameSlider=document.getElementById('frame'),splitSlider=document.getElementById('split'),opacitySlider=document.getElementById('opacity');
const play=document.getElementById('play'),readout=document.getElementById('readout'),divider=document.getElementById('divider');
let mode='split',lastFrame=-1,raf=null;
function currentFrame(){return Math.max(0,Math.min(data.frames-1,Math.floor(video.currentTime*data.fps+0.01)));}
function drawRuns(runs,color){ctx.fillStyle=color;for(const [y,x1,x2] of runs)ctx.fillRect(x1,y,x2-x1,1);}
function render(force=false){
 const frame=currentFrame();if(!force&&frame===lastFrame)return;lastFrame=frame;frameSlider.value=frame;
 readout.textContent=`Frame ${frame+1} / ${data.frames} · ${(frame/data.fps).toFixed(2)} s`;
 ctx.clearRect(0,0,data.width,data.height);ctx.globalAlpha=Number(opacitySlider.value)/100;
 const boundary=Number(splitSlider.value)/100*data.width;
 if(mode==='base'){drawRuns(data.base[frame],'#ff7767');}
 else if(mode==='refined'){drawRuns(data.refined[frame],'#45e6a3');}
 else{ctx.save();ctx.beginPath();ctx.rect(0,0,boundary,data.height);ctx.clip();drawRuns(data.base[frame],'#ff7767');ctx.restore();
      ctx.save();ctx.beginPath();ctx.rect(boundary,0,data.width-boundary,data.height);ctx.clip();drawRuns(data.refined[frame],'#45e6a3');ctx.restore();}
 ctx.globalAlpha=1;divider.style.display=mode==='split'?'block':'none';divider.style.left=`${splitSlider.value}%`;
}
function tick(){render();if(!video.paused&&!video.ended)raf=requestAnimationFrame(tick);}
function updatePlay(){play.textContent=video.paused?'Play':'Pause';if(!video.paused){cancelAnimationFrame(raf);tick();}}
play.addEventListener('click',()=>{if(video.paused)video.play();else video.pause();});
video.addEventListener('play',updatePlay);video.addEventListener('pause',updatePlay);video.addEventListener('ended',updatePlay);
video.addEventListener('loadedmetadata',()=>render(true));video.addEventListener('seeked',()=>render(true));video.addEventListener('timeupdate',()=>render());
frameSlider.addEventListener('input',()=>{video.currentTime=Number(frameSlider.value)/data.fps;render(true);});
document.getElementById('speed').addEventListener('change',e=>{video.playbackRate=Number(e.target.value);});
document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{mode=button.dataset.mode;document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));render(true);}));
splitSlider.addEventListener('input',()=>{document.getElementById('split-value').value=splitSlider.value+'%';render(true);});
opacitySlider.addEventListener('input',()=>{document.getElementById('opacity-value').value=opacitySlider.value+'%';render(true);});
document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target.tagName!=='BUTTON'&&e.target.tagName!=='INPUT'){e.preventDefault();play.click();}});
render(true);
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="COSMOS3_0015")
    args = parser.parse_args()
    case = (_workspace_root() / 'results/selected-45-v1/cases') / args.case
    status = json.loads((case / "status.json").read_text())
    manifest = json.loads((case / "refined_segmentation.json").read_text())
    base_source = json.loads((case / "base_segmentation_source.json").read_text())
    base, refined = case / "base_segmentation.npz", case / "refined_segmentation.npz"
    source = case / "v1/replay/interactive_exact-group/source.mp4"
    if status.get("state") != "complete" or not source.is_file():
        raise ValueError("case must be complete with a local source video")
    if status.get("mask_policy") != "validated persistent link7 replacement":
        raise ValueError("case did not use the persistent link7 replacement")
    if sha256(base) != base_source["base_segmentation_sha256"]:
        raise ValueError("base mask hash differs from provenance")
    if sha256(refined) != manifest["output_segmentation_sha256"]:
        raise ValueError("refined mask hash differs from provenance")
    shape, base_runs, refined_runs, changed_frames = read_masks(base, refined)
    frames, height, width = shape
    metadata = json.loads((case / "base_generation/segmentation.json").read_text())
    fps = metadata["video"]["fps"]
    if (frames, height, width) != (metadata["video"]["frames"],
                                   metadata["video"]["height"], metadata["video"]["width"]):
        raise ValueError("mask dimensions differ from source-video metadata")
    if changed_frames != manifest["changed_target_frames"] or frames != manifest["frame_count"]:
        raise ValueError("mask frame changes differ from merge manifest")
    output = case / "mask-replay"
    output.mkdir(exist_ok=True)
    payload = {"case": args.case, "frames": frames, "height": height,
               "width": width, "fps": fps, "base": base_runs, "refined": refined_runs}
    (output / "masks.js").write_text(
        "window.MASK_REPLAY=" + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    page = HTML
    for key, value in {
        "__WIDTH__": width, "__HEIGHT__": height, "__FRAMES__": frames,
        "__LAST_FRAME__": frames - 1,
        "__CHANGED_FRAMES__": changed_frames,
        "__CHANGED_PIXELS__": f"{manifest['changed_target_pixels']:,}",
    }.items():
        page = page.replace(key, str(value))
    page = page.replace("COSMOS3_0015", args.case)
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")
    print(f"frames={frames} changed={changed_frames} base_runs={sum(map(len, base_runs))} "
          f"refined_runs={sum(map(len, refined_runs))}")


if __name__ == "__main__":
    main()
