# Link5 guard and cached-mask review

Open [the selected review](results/20261008/index.html). It contains the ten-case
negative VLM review and earlier included/excluded mask replays. Source inputs,
cached positive coordinates, exact prompts/responses and manual selection
provenance are retained. These guard proposals were not rerun through SAM.

`run_guard_box_retry.py` runs explicit frozen-input negative-only VLM tests;
`retest_negative_case.py` uses the remaining three-call budget for one saved case.
`build_excluded_mask_replays.py` and `add_excluded_guard_prompts.py` export cached
mask evidence and original prompts without inference. They use this review's
cohort manifest, not an ablation package. Paid requests require an explicit run.

Production places P1/P2/P3 through the VLM. Only N1 must lie inside the green
box or five-pixel margin; N2 is unrestricted by the box. Parsing failures and
failed N1 checks retry identical inputs, at most three calls. Exhaustion is
terminal in the production coordinator. The accepted initial reference is fixed.
