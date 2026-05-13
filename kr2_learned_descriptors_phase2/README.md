# KR2 Phase 2: Learned Descriptor Matching

This directory is the phase-2 workspace for KR2:

> Use learned local descriptors before COLMAP, then compare the result against the fixed COLMAP/SIFT baseline.

## Scope

The phase-1 baseline is already fixed under:

```text
baselines/scene_09_20260317_firstloop_stride4_236
```

This phase focuses on the learned-descriptor frontend:

1. Use the same 236 keyframes as the SIFT baseline.
2. Build a deterministic image-pair schedule.
3. Extract SuperPoint keypoints and descriptors.
4. Match selected image pairs with LightGlue.
5. Save pair metrics and match visualizations.
6. Use these outputs later as the input for COLMAP database injection.

## Fixed Input

```text
data/scene_08_20260317_first_loop_then_stride4/first_loop_segment/keyframes_stride4_approx250/rgb
```

The current dataset contains 236 RGB keyframes.

## Pair Strategy

The first pass uses local temporal pairs:

- `stride=1`: adjacent-frame matching, e.g. `i -> i+1`
- `stride=4`: short-range matching, e.g. `i -> i+4`
- `stride=8`: wider local matching, e.g. `i -> i+8`

This produces enough matching coverage for a first COLMAP learned-feature frontend while keeping runtime bounded.

## Outputs

```text
outputs/scene_09_superpoint_lightglue/
  features/                 cached SuperPoint features
  matches/                  per-pair match arrays
  match_visualizations/     selected match review images
  reports/
    pair_schedule.csv       planned image pairs
    pair_metrics.csv        match counts and geometry checks
    dependency_check.txt    local environment check
  logs/
```

## Run

First build the image-pair schedule:

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

python3 kr2_learned_descriptors_phase2/scripts/build_pair_schedule.py \
  --config kr2_learned_descriptors_phase2/configs/phase2.env
```

Then run SuperPoint + LightGlue matching after the `lightglue` package is installed:

```bash
python3 kr2_learned_descriptors_phase2/scripts/run_superpoint_lightglue.py \
  --config kr2_learned_descriptors_phase2/configs/phase2.env
```

## Current Dependency Status

The local environment currently has:

- `torch`: available
- `opencv-python`: available as `cv2`
- `lightglue`: not installed yet

So the schedule and workspace can be prepared now. The actual learned matching step needs `lightglue` installed before it can run.

## Leadership Summary

This phase verifies whether a learned local-feature frontend can replace the default SIFT frontend before COLMAP. The expected comparison is:

```text
COLMAP + SIFT baseline
vs
SuperPoint + LightGlue matching + COLMAP mapper
```

The phase-2 output is not the final trajectory yet. It prepares the learned feature and matching evidence needed for the next step: writing learned matches into `database.db` and running COLMAP mapper.
