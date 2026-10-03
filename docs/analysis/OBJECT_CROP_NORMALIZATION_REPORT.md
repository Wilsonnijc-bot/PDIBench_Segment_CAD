# AnomalyDINO with matched crop canvases

Binary AUROC: **0.779720 → 0.779720**.

Controlled comparison: the same 380 selected pairs in 38 videos, the same model/checkpoint/settings, and the same revised labels and exclusions. Binary AUROC uses 37 videos (11 deformed, 26 non-deformed); one moderate label is omitted. No thresholds or parameters were fitted against these labels.

The new inputs trim transparent margins, then use identical centered canvas sizes per pair. Every visible RGB/alpha pixel, silhouette and native pixel area is preserved exactly. There is no interpolation, warping, equal-area matching or physical-scale estimation. Existing frame-0 correspondence estimates are unchanged. Both images now receive the same DINO shorter-edge resize factor. Black alpha compositing, eight reference rotations and no PCA masking remain unchanged.

Mean pair-score change: +0.002016; largest absolute pair-score change: 0.112511. A lower score is not itself evidence of improved anomaly detection. This is an evaluation on the existing selected dataset, not held-out validation.

| Case | Label | Previous sum | Normalized sum | Change |
|---|---:|---:|---:|---:|
| COSMOS2.5_0001 | 0 | — | — | excluded_by_user |
| COSMOS2.5_0005 | 1 | — | — | excluded_by_user |
| COSMOS2.5_0010 | 0 | — | — | excluded_by_user |
| COSMOS2.5_0015 | 0 | 5.273111 | 5.305735 | +0.032625 |
| COSMOS2.5_0021 | 0 | — | — | unavailable_reference |
| COSMOS2.5_0025 | 0 | 4.848323 | 5.005012 | +0.156689 |
| COSMOS2.5_0030 | 0 | 5.876441 | 5.916475 | +0.040034 |
| COSMOS2.5_0035 | 0 | 4.286703 | 4.365135 | +0.078432 |
| COSMOS2.5_0040 | 1 | 4.999637 | 4.954908 | -0.044729 |
| COSMOS2.5_0044 | 0 | 3.959054 | 3.940612 | -0.018442 |
| COSMOS2.5_0046 | 0 | 4.374643 | 4.271148 | -0.103495 |
| COSMOS2.5_0054 | 1 | 5.435957 | 5.586092 | +0.150135 |
| COSMOS2.5_0056 | 1 | 5.378403 | 5.245839 | -0.132565 |
| COSMOS2.5_0060 | 0 | 3.321624 | 3.391860 | +0.070235 |
| COSMOS2.5_0065 | 0 | — | — | excluded_by_user |
| COSMOS3_0001 | 0 | — | — | excluded_by_user |
| COSMOS3_0005 | 0 | 4.907949 | 4.996481 | +0.088532 |
| COSMOS3_0010 | 1 | 6.787095 | 6.465555 | -0.321540 |
| COSMOS3_0015 | 1 | 4.310871 | 4.483101 | +0.172230 |
| COSMOS3_0021 | 0 | 4.756233 | 4.651127 | -0.105106 |
| COSMOS3_0025 | 0 | 4.208287 | 4.080461 | -0.127826 |
| COSMOS3_0030 | 0.5 | 4.097289 | 4.131941 | +0.034652 |
| COSMOS3_0035 | 0 | 4.487278 | 4.357846 | -0.129432 |
| COSMOS3_0040 | 0 | 4.268938 | 4.474064 | +0.205126 |
| COSMOS3_0044 | 0 | 3.977582 | 4.153672 | +0.176090 |
| COSMOS3_0046 | 1 | 4.507823 | 4.510485 | +0.002662 |
| COSMOS3_0054 | 0 | 2.615270 | 2.595174 | -0.020097 |
| COSMOS3_0056 | 0 | 3.916096 | 3.826905 | -0.089191 |
| COSMOS3_0060 | 0 | 4.208132 | 4.352711 | +0.144579 |
| COSMOS3_0065 | 1 | 5.845632 | 5.936476 | +0.090845 |
| LVP_ROBOWM_0001 | 0 | — | — | excluded_by_user |
| LVP_ROBOWM_0005 | 1 | 4.581621 | 4.889268 | +0.307647 |
| LVP_ROBOWM_0010 | 0 | 5.343803 | 5.513300 | +0.169497 |
| LVP_ROBOWM_0015 | 0 | 3.940145 | 4.075227 | +0.135082 |
| LVP_ROBOWM_0021 | 0 | 3.769806 | 3.804051 | +0.034244 |
| LVP_ROBOWM_0025 | 0 | 3.843700 | 4.034555 | +0.190855 |
| LVP_ROBOWM_0030 | 1 | 4.517562 | 4.644158 | +0.126596 |
| LVP_ROBOWM_0035 | 0 | 4.084314 | 3.940649 | -0.143665 |
| LVP_ROBOWM_0040 | 1 | 3.574843 | 3.558105 | -0.016738 |
| LVP_ROBOWM_0044 | 0 | 3.301526 | 3.445367 | +0.143841 |
| LVP_ROBOWM_0046 | 0 | 3.465355 | 3.448616 | -0.016739 |
| LVP_ROBOWM_0054 | 0 | 4.894037 | 4.731437 | -0.162600 |
| LVP_ROBOWM_0056 | 1 | 5.663350 | 5.417121 | -0.246229 |
| LVP_ROBOWM_0060 | 0 | 5.185664 | 5.159581 | -0.026083 |
| LVP_ROBOWM_0065 | 0 | 3.514031 | 3.434017 | -0.080014 |
