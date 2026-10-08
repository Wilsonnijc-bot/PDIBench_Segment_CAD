# Current Link5 positive-point guard

The accepted initial reference is fixed. Do not replace or regenerate it during retries or reruns.

P1 has no default coordinate. The VLM places P1 and reviews P2/P3, returning all three coordinates. Gemini 3.8 Flash high is primary; GPT-6 Luna xhigh is fallback. Both use a 65,536-token output budget.

## Exact system prompt

```text
You review SAM3 point prompts for the robot's Link 5 forearm. Image 1 is an annotated reference example of ideal P1, P2, and P3 placement; Image 2 is the current full frame with the proposed Link 5 box and proposed P2 and P3 only. P1 has no default placement. Reason carefully from the images, but return only the requested answer. Place P1 and review P2 and P3, returning all three green positive points explicitly.
```

## Placement criteria

- P1: just inside the left interior edge of the rounded pale left lobe enclosed by the black border. Its center must lie on pale material with a clear margin from the black rim, away from the external wrist/collar.
- P2: farther right on the pale inset enclosed by the black border.
- P3: on the right white forearm panel, beyond the black inset and before the circular joint.

The current input image shows P2/P3 proposals only. The reference shows all three points. Valid P2/P3 proposals can be kept. P1 always requires an explicit VLM coordinate; PASS is not accepted for the positive call.

## Required output

```json
{"decision":"PLACE","positive_points_xy":[[p1x,p1y],[p2x,p2y],[p3x,p3y]]}
```

Coordinates use original full-frame integer pixels before display enlargement. Missing/invalid placement prevents SAM submission and uses the configured retry/fallback route.

[Prepared reference](../../../documentation/data/references/link5_guard/positive_three_points_reference.png) · [Reference provenance](../../../documentation/data/references/link5_guard/positive_three_points_reference.json)

## Exact user prompt used for LVP035

The current P2/P3 coordinates and frame dimensions vary by case; the placement rules stay the same.

```text
Ideal Link 5 prompt: three green positive points lie INSIDE the white forearm link. P1 lies just inside the LEFT interior edge of the rounded/circular pale left lobe enclosed by the black border, with clear pale material between its center and the black rim. This is the rounded LEFT end of the inset, not the adjacent wrist housing, metal collar, or right circular joint. P2 lies farther right on the pale inset enclosed by the black border; P3 lies on the right-hand white forearm panel, to the right of the black inset and left of the circular joint.
Use Image 1 to place P1 in Image 2. P1 has NO proposed or default coordinate; determine it from the visible robot anatomy. Then check P2 and P3 in Image 2 against Image 1: (1) is P2's center on the visible pale inset rather than its black rim, wrist, or background; (2) is P3's center on the visible right-hand white panel rather than the black inset, dark joint rim, pale circular joint disk, or background? Keep a valid P2 or P3 unchanged; move any invalid point clearly inside its intended white region, not a tiny edge nudge. Always reply with exactly {"decision":"PLACE","positive_points_xy":[[p1x,p1y],[p2x,p2y],[p3x,p3y]]}. Return all three coordinates, even when P2 and P3 are valid. Do not reply PASS. Current P2=[566, 65], P3=[740, 42]. Coordinates are integer pixels of the FULL Image 2 frame before display enlargement, width=832, height=480; top-left is (0,0). If the anatomy is unclear, do not invent points; reply with exactly REJECT.
```
