# Room Classifier Prototype

This prototype demonstrates a simple image-strategy approach for room-level region partitioning.

The first MVP uses a fixed-frame strategy:

1. Read an ordered RGB image sequence.
2. Split the sequence into a fixed number of regions.
3. Pick representative frames from each region.
4. Generate `segments.csv`, `segments.json`, and an HTML review page.

## Run

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

python3 prototypes/room_classifier/build_room_demo.py \
  --image-dir data/scene_08_20260317_first_loop_then_stride4/first_loop_segment/keyframes_stride4_approx250/rgb \
  --output prototypes/room_classifier/outputs/scene_09_room_demo \
  --num-regions 5 \
  --samples-per-region 6 \
  --clean
```

## Outputs

- `outputs/scene_09_room_demo/index.html`: visual demo page
- `outputs/scene_09_room_demo/segments.csv`: tabular region partition result
- `outputs/scene_09_room_demo/segments.json`: machine-readable region partition result
- `outputs/scene_09_room_demo/thumbnails`: representative frames

## Next Improvements

- Replace fixed-frame splitting with visual-change boundary detection.
- Add manually editable room labels.
- Fuse COLMAP trajectory positions to split regions by motion and revisits.
