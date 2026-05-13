# V3 Trajectory-Assisted Region Demo

This version uses COLMAP camera trajectory positions to assist room-level region partitioning.

Strategy:

1. Read ordered RGB keyframes.
2. Read COLMAP sparse `images.txt` poses and match them back to image names.
3. Use the COLMAP-derived `trajectory.tum` as trajectory provenance/checking input.
4. Compute camera-center positions and cumulative path length.
5. Split the trajectory into balanced path-length regions.
6. Generate `index.html`, `segments.csv`, `segments.json`, `trajectory_positions.csv`, and `trajectory_overview.svg`.

## Run

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

python3 prototypes/room_region_v3_trajectory_assisted/build_trajectory_demo.py \
  --image-dir data/scene_08_20260317_first_loop_then_stride4/first_loop_segment/keyframes_stride4_approx250/rgb \
  --images-txt baselines/scene_09_20260317_firstloop_stride4_236/runs/scene_09_docker_cpu_optimized_20260430_164041/workspace/colmap/sparse_txt/images.txt \
  --trajectory-tum baselines/scene_09_20260317_firstloop_stride4_236/runs/scene_09_docker_cpu_optimized_20260430_164041/results/trajectory_plots/trajectory.tum \
  --output prototypes/room_region_v3_trajectory_assisted/outputs/scene_09_trajectory_assisted \
  --num-regions 5 \
  --samples-per-region 6 \
  --clean
```

## Outputs

- `outputs/scene_09_trajectory_assisted/index.html`
- `outputs/scene_09_trajectory_assisted/segments.csv`
- `outputs/scene_09_trajectory_assisted/segments.json`
- `outputs/scene_09_trajectory_assisted/trajectory_positions.csv`
- `outputs/scene_09_trajectory_assisted/trajectory_overview.svg`
- `outputs/scene_09_trajectory_assisted/thumbnails/`
