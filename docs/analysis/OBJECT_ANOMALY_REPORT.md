# Object anomaly scores: all cases except the three 0001 videos

Current binary AUROC: **0.795977**, using 12 deformed and 29 non-deformed videos.

All 42 requested videos are scored (420 pairs); one moderate label is omitted only from binary AUROC. The only exclusions are COSMOS2.5_0001, COSMOS3_0001 and LVP_ROBOWM_0001. No score is imputed or set to zero.

| Stage | Scored videos | Binary AUROC |
|---|---:|---:|
| Original ten-frame scoring | 35 | 0.790514 |
| Refresh COSMOS2.5_0056 on the original scored cohort | 35 | 0.802372 |
| Add repaired LVP_ROBOWM_0010,0015,0060 | 38 | 0.779720 |
| Matched-canvas normalization, same 38 videos | 38 | 0.779720 |
| Include every case except the three 0001 videos | 42 | 0.795977 |

The original 0.790514 and later 0.779720 describe different scored cohorts. Refreshing COSMOS2.5_0056 alone gives 0.802372 on the old 35-video cohort; adding the three repaired LVP videos gives 0.779720 on 38 videos. Matched-canvas normalization leaves that 38-video AUROC unchanged. The new current value adds four videos, so it is a cohort expansion, not a controlled normalization-improvement claim.

Newly included: COSMOS2.5_0005, COSMOS2.5_0010, COSMOS2.5_0065 and COSMOS2.5_0021. The first three already had valid normalized crop pairs. COSMOS2.5_0021 uses its hash-verified accepted VLM3 frame0 repair from the earlier GPU experiment; the default synchronized handoff refreshed occlusion detection, replay and V5 ten-frame crops. Its frame-0 reference now has zero link7 overlap. No new VLM points or mask inference were fabricated.

Model/checkpoint/inference settings match the preceding normalized run. The 38 existing videos retain their exact crop files and scores. The 40 newly eligible pairs are scored using the same model. Visible pixels and contours are preserved through integer translation and shared-canvas padding; existing correspondence estimates remain estimated. No threshold or parameter was fitted against these labels.

| Case | Label | Status | Pair count | Anomaly sum |
|---|---:|---|---:|---:|
| COSMOS2.5_0001 | 0 | excluded_by_user | 0 | — |
| COSMOS2.5_0005 | 1 | newly included | 10 | 6.157103 |
| COSMOS2.5_0010 | 0 | newly included | 10 | 4.419661 |
| COSMOS2.5_0015 | 0 | scored | 10 | 5.305735 |
| COSMOS2.5_0021 | 0 | newly included | 10 | 3.999616 |
| COSMOS2.5_0025 | 0 | scored | 10 | 5.005012 |
| COSMOS2.5_0030 | 0 | scored | 10 | 5.916475 |
| COSMOS2.5_0035 | 0 | scored | 10 | 4.365135 |
| COSMOS2.5_0040 | 1 | scored | 10 | 4.954908 |
| COSMOS2.5_0044 | 0 | scored | 10 | 3.940612 |
| COSMOS2.5_0046 | 0 | scored | 10 | 4.271148 |
| COSMOS2.5_0054 | 1 | scored | 10 | 5.586092 |
| COSMOS2.5_0056 | 1 | scored | 10 | 5.245839 |
| COSMOS2.5_0060 | 0 | scored | 10 | 3.391860 |
| COSMOS2.5_0065 | 0 | newly included | 10 | 5.001851 |
| COSMOS3_0001 | 0 | excluded_by_user | 0 | — |
| COSMOS3_0005 | 0 | scored | 10 | 4.996481 |
| COSMOS3_0010 | 1 | scored | 10 | 6.465555 |
| COSMOS3_0015 | 1 | scored | 10 | 4.483101 |
| COSMOS3_0021 | 0 | scored | 10 | 4.651127 |
| COSMOS3_0025 | 0 | scored | 10 | 4.080461 |
| COSMOS3_0030 | 0.5 | scored | 10 | 4.131941 |
| COSMOS3_0035 | 0 | scored | 10 | 4.357846 |
| COSMOS3_0040 | 0 | scored | 10 | 4.474064 |
| COSMOS3_0044 | 0 | scored | 10 | 4.153672 |
| COSMOS3_0046 | 1 | scored | 10 | 4.510485 |
| COSMOS3_0054 | 0 | scored | 10 | 2.595174 |
| COSMOS3_0056 | 0 | scored | 10 | 3.826905 |
| COSMOS3_0060 | 0 | scored | 10 | 4.352711 |
| COSMOS3_0065 | 1 | scored | 10 | 5.936476 |
| LVP_ROBOWM_0001 | 0 | excluded_by_user | 0 | — |
| LVP_ROBOWM_0005 | 1 | scored | 10 | 4.889268 |
| LVP_ROBOWM_0010 | 0 | scored | 10 | 5.513300 |
| LVP_ROBOWM_0015 | 0 | scored | 10 | 4.075227 |
| LVP_ROBOWM_0021 | 0 | scored | 10 | 3.804051 |
| LVP_ROBOWM_0025 | 0 | scored | 10 | 4.034555 |
| LVP_ROBOWM_0030 | 1 | scored | 10 | 4.644158 |
| LVP_ROBOWM_0035 | 0 | scored | 10 | 3.940649 |
| LVP_ROBOWM_0040 | 1 | scored | 10 | 3.558105 |
| LVP_ROBOWM_0044 | 0 | scored | 10 | 3.445367 |
| LVP_ROBOWM_0046 | 0 | scored | 10 | 3.448616 |
| LVP_ROBOWM_0054 | 0 | scored | 10 | 4.731437 |
| LVP_ROBOWM_0056 | 1 | scored | 10 | 5.417121 |
| LVP_ROBOWM_0060 | 0 | scored | 10 | 5.159581 |
| LVP_ROBOWM_0065 | 0 | scored | 10 | 3.434017 |
