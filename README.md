# wheeltec_colmap

This workspace is a dedicated COLMAP experiment area for Wheeltec room-scale data.
It is organized around four immediate goals:

1. Build a stable reference trajectory on one fixed scene.
2. Keep a reproducible sparse-reconstruction baseline.
3. Prepare a minimal room-labeling prototype workspace.
4. Compare classical and learned local-feature pipelines against the same reference.

## Recommended workflow

1. Put one image sequence into `data/scene_01/images`.
2. Adjust `configs/experiment.env` if needed.
3. Run sparse reconstruction:

```bash
bash scripts/run_colmap_stable.sh
```

4. Export TUM trajectory:

```bash
python3 scripts/colmap_images_to_tum.py \
  --input workspace/colmap/sparse_txt/images.txt \
  --output results/trajectory_plots/trajectory.tum
```

5. Export a delivery package:

```bash
bash scripts/export_artifacts.sh
```

6. Append unified metrics for the current run:

```bash
python3 scripts/summarize_sparse_model.py \
  --images-txt workspace/colmap/sparse_txt/images.txt \
  --points3d-txt workspace/colmap/sparse_txt/points3D.txt \
  --total-images "$(find data/scene_01/images -maxdepth 1 -type f | wc -l)" \
  --duration-sec 0 \
  --output results/reports/experiment_metrics.csv
```

## Layout

- `configs`: fixed experiment settings
- `data/scene_01/images`: input image sequence for the highest-priority reference run
- `workspace/colmap`: local COLMAP database and sparse outputs
- `results/trajectory_plots`: TUM trajectory and visualizations
- `results/screenshots`: screenshots for delivery/reporting
- `results/reports`: reproducible metric logs and notes
- `results/exports`: packaged artifacts
- `prototypes/room_classifier`: minimal room-label prototype area
- `comparisons`: notes and results for feature-descriptor comparison

## First milestone checklist

- Sparse model exists at `workspace/colmap/sparse/0`
- TUM trajectory exists at `results/trajectory_plots/trajectory.tum`
- At least one screenshot exists in `results/screenshots`
- One metrics row is appended to `results/reports/experiment_metrics.csv`
