# KR2 2026-05-09 LiDAR-COLMAP Evaluation

本目录用于保存 2026-05-09 新采集数据包的实验流程与结果。结构仿照上一版 `kr2_20260508_lidar_colmap_eval`，但当前目录保持干净，先只配置提取和后续实验所需脚本。

## 输入数据

```text
ROS2 bag:
/home/kiyi/COLMAP/wheeltec_colmap/bag/rosbag2_2026_05_09-11_53_56

2D LiDAR map:
/home/kiyi/COLMAP/wheeltec_colmap/map/WHEELTEC.yaml
/home/kiyi/COLMAP/wheeltec_colmap/map/WHEELTEC.pgm
```

## 已确认话题

```text
RGB:        /camera1_SD0140820L0057/rgb_raw
Depth:      /camera1_SD0140820L0057/depth_raw
LaserScan:  /scan
Odom:       /odom
Odom fused: /odom_combined
Map:        /map
TF:         /tf, /tf_static
```

## 目录

```text
configs/       场景配置
scripts/       本工作区脚本
data/extracted 从 bag 提取出的连续 RGB-D 数据
data/visual_loop 后续人工确认第一圈后保存切段
data/keyframes 后续关键帧
outputs/       odom 候选、LightGlue 输出、轨迹对齐输出
runs/          COLMAP 运行结果
reports/       汇报材料
figures/       可视化图片
logs/          日志
```

## 第一步：提取 RGB-D

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

CONFIG_FILE=kr2_20260509_lidar_colmap_eval/configs/scene_20260509.env \
bash kr2_20260509_lidar_colmap_eval/scripts/run_extract_full_rgbd.sh
```

输出：

```text
kr2_20260509_lidar_colmap_eval/data/extracted/full_rgbd/rgb
kr2_20260509_lidar_colmap_eval/data/extracted/full_rgbd/depth
kr2_20260509_lidar_colmap_eval/data/extracted/full_rgbd/manifest.json
```

## 第二步：导出里程计候选

RGB-D 提取完成后再运行：

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

CONFIG_FILE=kr2_20260509_lidar_colmap_eval/configs/scene_20260509.env \
bash kr2_20260509_lidar_colmap_eval/scripts/run_export_odom.sh
```

输出：

```text
kr2_20260509_lidar_colmap_eval/outputs/odom/
```

## 后续流程

1. 人工查看 `data/extracted/full_rgbd/rgb`，确认起点到第一次明显回环完成的图像范围。
2. 用 `scripts/copy_rgbd_segment_keyframes.py` 或 `scripts/copy_rgbd_segment_keyframes_skip_ranges.py` 生成关键帧。
3. 跑 SIFT + COLMAP baseline。
4. 跑 SuperPoint + LightGlue + COLMAP。
5. 与 LiDAR 地图或 LiDAR/SLAM 轨迹做对齐评估。
