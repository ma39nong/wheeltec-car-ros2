# V2 Visual-Change Region Demo

This version finds room-level candidate boundaries from adjacent-frame visual changes.

Strategy:

1. Read ordered RGB keyframes.
2. Extract HSV color histograms and compact edge-texture features.
3. Score visual change between adjacent frames.
4. Select the strongest boundary candidates with a minimum segment length constraint.
5. Generate `index.html`, `segments.csv`, and `segments.json`.
6. Generate boundary before/after review images and a visual-change score curve.

## Run

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

python3 prototypes/room_region_v2_visual_change/build_visual_change_demo.py \
  --image-dir data/scene_08_20260317_first_loop_then_stride4/first_loop_segment/keyframes_stride4_approx250/rgb \
  --output prototypes/room_region_v2_visual_change/outputs/scene_09_visual_change \
  --num-regions 5 \
  --min-segment-len 24 \
  --samples-per-region 6 \
  --labels '会议室内侧走廊前半段|会议室大屏幕处走廊|会议室外侧走廊|会议室杂物区|会议室内侧走廊后半段' \
  --clean
```

## Outputs

- `outputs/scene_09_visual_change/index.html`
- `outputs/scene_09_visual_change/segments.csv`
- `outputs/scene_09_visual_change/segments.json`
- `outputs/scene_09_visual_change/visual_change_curve.svg`
- `outputs/scene_09_visual_change/boundary_pairs/`
- `outputs/scene_09_visual_change/thumbnails/`

The HTML page includes:

- selected boundary before/after comparisons, such as `frame 70 -> frame 71`
- visual change score curve with selected boundaries marked in red
- representative frames for every generated region
