# Temporal Localization of V2 Rigidity

Generated: September 7, 2026

## Bottom line

The v2 per-frame rigidity history contains weak local timing information, but the strongest rigidity increase is not a reliable deformation-onset estimator. Results that look favorable under a uniform circular-time null mostly disappear under the stricter dataset-and-component timestamp shuffle, which preserves the strong concentration of human annotations near seconds 0 and 1.

The primary mapping evaluates 82 quality-available component events from 46 videos. Its strongest increase has a median absolute timing error of 1.47 s. The absolute rigidity peak has a median error of 0.94 s.

## Overall alignment

Primary mapping: upper arm = link2, forearm = mean(link4, link5), gripper = link7.

| Detector and window | Observed | Circular chance | Circular p | Stratified chance | Stratified p |
|---|---:|---:|---:|---:|---:|
| Strongest increase, +/-0.25 s | 10/82 (12.2%) | 8.0% | 0.1152 | 12.9% | 0.6603 |
| Strongest increase, +/-0.5 s | 15/82 (18.3%) | 15.9% | 0.3134 | 19.9% | 0.7315 |
| Strongest increase, +/-1 s | 34/82 (41.5%) | 31.2% | 0.0236 | 38.9% | 0.2723 |
| Either of two strongest increases, +/-0.25 s | 20/82 (24.4%) | 15.5% | 0.0253 | 23.9% | 0.5113 |
| Either of two strongest increases, +/-0.5 s | 29/82 (35.4%) | 29.5% | 0.1464 | 36.1% | 0.6302 |
| Either of two strongest increases, +/-1 s | 51/82 (62.2%) | 51.7% | 0.0266 | 59.9% | 0.2763 |
| Absolute peak, +/-0.25 s | 8/82 (9.8%) | 8.0% | 0.3305 | 12.4% | 0.8903 |
| Absolute peak, +/-0.5 s | 20/82 (24.4%) | 15.9% | 0.0295 | 24.0% | 0.5176 |
| Absolute peak, +/-1 s | 46/82 (56.1%) | 31.2% | <0.0001 | 46.4% | 0.0043 |

The circular null asks whether detections beat a uniformly shifted time. The stratified shuffle is the more conservative result: it shuffles detected times among events from the same dataset and component, retaining startup effects and the annotation-time distribution.

Timestamp and boundary concentration are substantial: 81.7% of usable human events are at or before 1 second. 36.6% of strongest increases and 48.8% of absolute peaks occur within 0.5 seconds of the start or end of a clip.

## Mapping comparison

| Mapping | Events | Surge within 0.5 s | Stratified p | Peak within 0.5 s | Stratified p |
|---|---:|---:|---:|---:|---:|
| recommended | 82 | 15/82 (18.3%) | 0.7315 | 20/82 (24.4%) | 0.5176 |
| combined | 82 | 15/82 (18.3%) | 0.7315 | 20/82 (24.4%) | 0.5176 |
| single | 82 | 14/82 (17.1%) | 0.7933 | 20/82 (24.4%) | 0.5403 |

## Component results

| Component | Events | Median surge error | Surge within 0.5 s | Stratified p | Peak within 0.5 s | Stratified p |
|---|---:|---:|---:|---:|---:|---:|
| upper arm | 7 | 4.33 s | 0/7 (0.0%) | 1.0000 | 1/7 (14.3%) | 1.0000 |
| forearm | 31 | 1.56 s | 7/31 (22.6%) | 0.5725 | 7/31 (22.6%) | 0.6431 |
| gripper | 44 | 1.23 s | 8/44 (18.2%) | 0.8040 | 12/44 (27.3%) | 0.3765 |

## Dataset results

| Dataset | Events | Median surge error | Surge within 0.5 s | Stratified p | Peak within 0.5 s | Stratified p |
|---|---:|---:|---:|---:|---:|---:|
| LVP_ROBOWM | 17 | 0.88 s | 3/17 (17.6%) | 0.7917 | 5/17 (29.4%) | 0.2948 |
| COSMOS2.5 | 31 | 1.19 s | 7/31 (22.6%) | 0.7822 | 9/31 (29.0%) | 0.7765 |
| COSMOS3 | 34 | 3.90 s | 5/34 (14.7%) | 0.4579 | 6/34 (17.6%) | 0.6565 |

## Interpretation

The strongest-increase result from the previous 55-video analysis does not reproduce with the v2 histories, the additional timestamped COSMOS2.5 cases, and the stricter shuffle control. Allowing either of the two strongest increases raises the raw hit rate but also raises the chance rate by a similar amount.

The absolute peak is closer to many annotations than the strongest increase, especially with a +/-1 s tolerance. This should not be interpreted as clean onset localization: many annotations and many rigidity peaks occur early, and the score can contain a large startup transient. The strict shuffle result indicates whether the remaining association is video-specific.

The practical output of this analysis is a candidate timestamp, not a validated detector. The per-frame signal should be combined with visibility/coverage changes and tested against independently re-annotated onset intervals before it is used as a benchmark metric.

## Method

1. Smooth each component's interpolated rigidity-deviation history with a centered 0.25-second moving average.
2. Define the surge as the largest increase over the preceding 0.25 seconds.
3. Also evaluate the two strongest separated increases and the maximum smoothed rigidity level.
4. Convert frames with source rates of 16 FPS for LVP_ROBOWM and COSMOS2.5, and 24 FPS for COSMOS3.
5. Exclude a component event only when none of its mapped links has a complete per-frame history.
6. Use video-cluster bootstrap intervals, circular-time permutations, and dataset/component-stratified timestamp shuffles.

## Timestamp curation

- COSMOS2.5_0002 says 'minor deform at sec1' without assigning that time to a component; the structured component and prose anatomy also conflict, so it is excluded.
- LVP_ROBOWM_0031 has an upper-arm structured label but says forearm deformation at sec1; it is excluded from component-matched localization.
- COSMOS3_0014 contains an ambiguous second value written as 05; only the unambiguous gripper event at 1 s is used.
- COSMOS2.5_0047 says deformation at sec0 and sec1 without ordering the forearm and gripper events; both are represented by the interval midpoint 0.5 s.
- Second-half-of-sec0 annotations are represented by 0.75 s. Start-of-secN is represented by N seconds, and end-of-sec1 is represented by 2 seconds.
- Human timestamps are approximate and were not independently re-annotated from the source videos.
