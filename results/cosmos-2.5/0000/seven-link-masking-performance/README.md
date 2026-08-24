# Seven-Link SAM3 + CAD Mask Diagnostic

Status: **diagnostic only; not benchmark-verified**

Input: frame 0 of `0000.mp4` (1280x704)

Method:

1. Load SAM3 once.
2. Query each link independently with the shared phrase `white robotic arm link`
   and one positive, link-localized box.
3. Rank returned candidates against the 24 rendered CAD silhouettes for that
   specific Franka link.
4. Measure mask containment and pairwise overlap.

CAD is not directly ingestible by SAM3. The `.dae` meshes are rendered into
silhouettes and used as shape evidence after SAM3; the link boxes are the
geometric prompts accepted by SAM3.

## Result

| Link | SAM3 score | CAD similarity | Pixels | Inside prompt box |
| --- | ---: | ---: | ---: | ---: |
| link1 | 0.875 | 0.380 | 11,969 | 95.6% |
| link2 | 0.914 | 0.518 | 19,620 | 100.0% |
| link3 | 0.887 | 0.694 | 34,212 | 99.0% |
| link4 | 0.914 | 0.710 | 30,179 | 98.3% |
| link5 | 0.863 | 0.091 | 23,181 | 99.7% |
| link6 | 0.855 | 0.503 | 27,860 | 95.6% |
| link7 | 0.902 | 0.667 | 11,022 | 100.0% |

- Seven of seven masks are non-empty.
- Total query time after model/video setup: 6.27 seconds.
- Maximum pairwise mask IoU: 0.0503.
- Extra area caused by overlapping masks: 3.79% of their union.
- `link5` fails the CAD shape check and needs a better localization/pose prior.

Artifacts:

- `0000/seven-mask-overlay.jpg`: all seven masks and input boxes.
- `0000/seven-link-cad-comparison.jpg`: each SAM3 mask beside the closest CAD
  silhouette for its assigned link.
- `0000/link1.jpg` through `0000/link7.jpg`: individual comparisons.
- `0000/summary.json`: complete measurements and pairwise IoU matrix.

## Combined-Prompt Constraint

The installed SAM3 video API supports one text phrase plus box evidence, but its
initial visual-prompt path accepts only one box. A new semantic prompt resets the
semantic state. It therefore cannot represent seven independent text phrases or
seven initial link boxes in one `add_prompt` call.

The practical combined strategy is to keep one loaded SAM3 model and run seven
independent text+box queries, then resolve overlap and propagate the accepted
seven masks together. For stronger localization, generate each box from a
projected articulated CAD model, robot keypoints, or a whole-arm skeleton rather
than storing pose-specific boxes.
