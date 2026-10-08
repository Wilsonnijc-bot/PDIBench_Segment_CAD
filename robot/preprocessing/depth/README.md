# Link5 observed-depth support

`link5_depth_filter.py` is the reusable robot preprocessing implementation used by
Link5 shape-codebook preparation, guide preparation and full-video evaluation.
It retains the source mask, erodes support in original-image pixels, aligns that
support to the native depth grid, and rejects invalid or sparsely supported depth
pixels. Retained XYZ and depth values are never smoothed or moved. Apply the same
configuration to reference and test frames. Moving this implementation did not
change its numerical code or enable it in unrelated robot/object pipelines.
