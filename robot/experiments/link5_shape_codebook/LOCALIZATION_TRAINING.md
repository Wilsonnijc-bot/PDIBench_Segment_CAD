# Link5: teach where the deformation is

The normal codebook still comes only from frame0 of `COSMOS3_0046`,
`COSMOS2.5_0044`, `LVP_ROBOWM_0044`, and `COSMOS3_0056`. Each is an individual
training sample. The34,280-point concatenated cloud remains a visualization of
those observations. Heldout clouds and later frames never update this codebook.

For each normal cloud `S`, generate a corresponding deformed cloud `A`. The
restoration target is `S - A`. The synthetic generator also supplies the region
label `m`. Train on **both** shapes:

| Input/region | Offset target | Anomaly-probability target | Raw-score objective |
|---|---|---|---|
| Clean `S`, all points | zero | zero | low |
| Deformed `A`, labeled region | `S - A` | one | higher than the same clean points |
| Deformed `A`, background | generator's restoration offset | zero | low |

The upstream Gaussian generator has small tails outside its labeled region;
that background is not necessarily exactly unchanged. We keep its returned
coordinates, offsets and labels intact, and verify correspondence numerically.

The scalar score stays `sum(abs(predicted_offset)) * sigmoid(mask_logit)`.
There is no per-cloud rescaling in comparisons.

## Why another default-generator run needs changes

The earlier `paper_original` checkpoint already trained with all six repository
generators. Three modes—`holes`, `angle_displacement`, and `plane_missing`—return
unchanged coordinates with nonzero restoration targets and positive labels.
Training on identical shapes with contradictory labels undermines the desired
clean/deformed distinction. The new trial uses the unmodified, actual-displacement
generators `sink`, `concavity`, and `bulges`, with the default severities
`0.001`, `0.01`, `0.1`. It records an audit of all six modes.

## Exact experimental loss

Let `R(v)` be the average of the foreground mean and background mean, giving
the two regions equal weight. If only one region exists, use that region's mean.
Let `d_i = sum(abs(S_i - A_i))` and `q = max(max(d_i), 0.001)`.

```
L_offset = [R(sum(abs(o_A - (S-A)))) + mean(sum(abs(o_S)))] / (2*q)
L_mask   = [R(BCE(logit_A, m)) + mean(BCE(logit_S, 0))] / 2
L_rank   = mean_foreground(relu(0.5*d - (score_A-score_S))) / q
L        = L_offset + 0.5*L_mask + 0.5*L_rank
```

Clean offset supervision prevents the probability gate from merely concealing
a large predicted correction. Region balancing prevents a small defect from
being overwhelmed by the clean majority. The margin connects training directly
to the raw score being inspected in the replay. This replaces the native loss
for this experiment; it is not presented as a paper-exact reproduction.

## Controlled comparison

- `paired_native`: upstream network with this paired objective.
- `paired_point_residual`: identical objective, data and synthetic draws, adding
  the frozen original point features to the attention output. This gives the
  prediction head point-dependent information even when every patch retrieves
  the same template. It adds no new parameters.

Both use Adam, learning rate0.001, batch size1, seed0,1500 epochs/6000 updates
and continuous normal-only codebook refresh. Normals are cached on the fixed
seed0 ordering of the observed points. No duplication, individual scaling or
new masks/depth clouds are introduced. Two processes run concurrently on one
H200; launcher snapshots and logs are allocation-local and immutable.

## What counts as evidence

At epochs300 and1500, evaluate each of41 heldout frame0 clouds clean and with
nine fresh synthetic variants. Record raw score maps, point AUROC/AP, scores
inside versus outside the label, and inside versus corresponding clean points.
Set a threshold from the four clean training clouds'99.5th score percentile and
report heldout-clean false positives and defect recall at that fixed threshold.
Do not infer localization AUROC for a deformation covering the whole cloud.

The complete matrices live in the current round's
`localization/<variant>/evaluation_epoch_1500.json`. Real model-input/target/output
archives for LVP0001 and checkpoint/loss/provenance files sit beside them.
These are research diagnostics. The old checkpoints and original full-video
sanity gate remain in place; completing training does not establish detection
success or justify full-video inference.
