import json
from pathlib import Path

import cv2
import numpy as np

from sam3.model_builder import build_sam3_video_predictor


root = Path("/root/autodl-tmp/pdi")
output_dir = root / "runs/cosmos-2.5/0000/prompt-diagnostic"
video = root / "runs/cosmos-2.5/videos/0000.mp4"
prompts = [
    "robot base",
    "robot shoulder",
    "robot upper arm",
    "robot elbow",
    "robot forearm",
    "robot wrist",
    "robot hand",
    "white robot arm segment",
    "black robot joint",
]
colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 0),
]

output_dir.mkdir(parents=True, exist_ok=True)
predictor = build_sam3_video_predictor(
    checkpoint_path=str(root / "models/sam3/sam3.pt"),
    bpe_path=str(root / "models/sam3/bpe_simple_vocab_16e6.txt.gz"),
)
session_id = predictor.handle_request(
    {
        "type": "start_session",
        "resource_path": str(video),
        "offload_video_to_cpu": True,
    }
)["session_id"]
capture = cv2.VideoCapture(str(video))
ok, frame = capture.read()
capture.release()
if not ok:
    raise RuntimeError(f"Cannot read {video}")

summary = {}
try:
    for index, prompt in enumerate(prompts):
        outputs = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
                "output_prob_thresh": 0.10,
            }
        )["outputs"]
        masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
        scores = np.asarray(outputs["out_probs"], dtype=float)
        summary[prompt] = {
            "count": len(masks),
            "scores": scores.tolist(),
            "areas": [int(mask.sum()) for mask in masks],
        }
        preview = frame.copy()
        for mask in masks:
            tint = np.zeros_like(frame)
            tint[:] = colors[index]
            preview[mask] = cv2.addWeighted(
                frame[mask], 0.35, tint[mask], 0.65, 0.0
            )
        filename = f"{index:02d}-{prompt.replace(' ', '_')}.jpg"
        cv2.imwrite(str(output_dir / filename), preview)
finally:
    predictor.handle_request({"type": "close_session", "session_id": session_id})
    predictor.shutdown()

(output_dir / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
