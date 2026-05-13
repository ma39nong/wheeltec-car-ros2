# 2026-05-09 COLMAP 首圈实验阶段总结

## 目标

基于 2026-05-09 新采集的 ROS2 bag，先完成传统 SIFT + COLMAP 基线建图，观察第一圈回环效果，再决定是否进入 SuperPoint + LightGlue 或 LiDAR 对齐评估。

## 数据来源

```text
ROS2 bag:
/home/kiyi/COLMAP/wheeltec_colmap/bag/rosbag2_2026_05_09-11_53_56

工作目录:
/home/kiyi/COLMAP/wheeltec_colmap/kr2_20260509_lidar_colmap_eval
```

已从 bag 中提取完整 RGB-D 数据：

```text
kr2_20260509_lidar_colmap_eval/data/extracted/full_rgbd/rgb
kr2_20260509_lidar_colmap_eval/data/extracted/full_rgbd/depth
```

## 当前首圈选择策略

人工检查 RGB 图像后，第一圈回环大约在 1600 帧附近形成。直接使用 `0..1600` 会失败，主要原因是存在低运动/近静止段，干扰 COLMAP 初始化与建图稳定性。

当前有效预处理策略：

```text
起始帧: 30
低运动段剔除: 934..985
终止帧已测试: 1600、1602、1606
```

含义：

```text
1. 去掉开头基本未运动的帧。
2. 去掉中间一段变化很小的重复视角。
3. 尝试让终点更接近起点，以减小首尾缺口。
```

## 已完成实验记录

| 方案 | 图像范围 | stride | 关键帧 | 注册结果 | 稀疏点 | 耗时 | 观察结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| SIFT A | 30..1600, skip 934..985 | 4 | 380 | 380/380 | 77709 | 896s | 成功建图，但首尾有明显小缺口 |
| SIFT B | 30..1600, skip 934..985 | 3 | 507 | 507/507 | 102654 | 1659s | 点云更密，但首尾仍未闭合 |
| SIFT C | 30..1602, skip 934..985 | 4 | 381 | 381/381 | 77908 | 906s | 缺口略小，但仍不算闭环 |
| SIFT D | 30..1606, skip 934..985 | 4 | 382 | 382/382 | 78304 | 883s | 当前较好的 stride4 结果，但仍有缺口 |
| SIFT E | 30..1606, skip 934..985 | 3 | 509 | 509/509 | 102839 | 1594s | 当前较好的 stride3 结果，点更多，但仍不算完全闭环 |
| SIFT F | 30..1660, skip 934..985 | 4 | 395 | 2/395 | 112 | 544s | 建图失败，仅注册 2 张，相关 run 已剔除 |
| SIFT G | 30..1620, skip 934..985 | 4 | 385 | 385/385 | 79115 | 954s | 有效建图，缺口有变化但仍未形成完整圈 |
| SIFT H | 30..1700, skip 934..985 | 4 | 405 | 405/405 | 82713 | 1037s | 当前采用的 SIFT 基线；轨迹可见完整一圈，首尾允许重合 |
| SuperPoint + LightGlue | 30..1700, skip 934..985 | 4 | 405 | 405/405 | 28370 | 263s | 深度学习特征建图成功；点数少于 SIFT，建图阶段更快 |
| SuperPoint + LightGlue | 30..1600, skip 934..985 | 4 | 380 | 380/380 | 26704 | 228s | 流程跑通，但视觉效果较差，相关目录已剔除 |

## 当前保留的有效结果

当前目录中只保留最终基线相关材料：

```text
1. SIFT baseline: 30..1700 + skip 934..985 + stride4
   runs/sift_colmap_manual_loop_000030_001700_stride4_skip934-985_overlap30
```

其他中间实验结果只保留在上表记录中，相关 run、keyframes、visual_loop、configs 已从目录中剔除，以保持工作区干净。

当前保留路径：

```text
configs/sift_colmap_manual_loop_000030_001700_stride4_skip934-985_overlap30.env
data/visual_loop/rgbd_000030_001700_manual_skip934-985
data/keyframes/manual_loop_000030_001700_stride4_skip934-985
runs/sift_colmap_manual_loop_000030_001700_stride4_skip934-985_overlap30
```

## 当前判断

当前采用领导确认后的评价口径：

```text
不要求严格几何闭环；
允许首尾轨迹重合一段；
但必须能看到完整一圈。
```

按这个口径，`30..1700 + skip 934..985 + stride4` 可作为当前 SIFT + COLMAP 基线：

```text
registered_images: 405/405
sparse_points: 82713
trajectory: runs/sift_colmap_manual_loop_000030_001700_stride4_skip934-985_overlap30/results/trajectory_plots/trajectory.tum
```

## 下一步建议

当前建议不再继续扩大终止帧，避免引入第二圈干扰。后续建议基于 `30..1700 + stride4` 做：

```text
1. 保存 COLMAP 截图作为可视化证据。
2. 使用已导出的 trajectory.tum 与 LiDAR/SLAM 轨迹或地图对齐。
3. 若后续需要深度学习描述子对比，应先复用同一批 405 张关键帧。
```

## 深度学习特征实验准备

为保证与 SIFT 基线公平对比，SuperPoint + LightGlue 使用同一批 405 张关键帧：

```text
kr2_20260509_lidar_colmap_eval/data/keyframes/manual_loop_000030_001700_stride4_skip934-985/rgb
```

已生成配置：

```text
SuperPoint + LightGlue 匹配配置:
kr2_20260509_lidar_colmap_eval/configs/lightglue_manual_loop_000030_001700_stride4_skip934-985.env

LightGlue matches 导入 COLMAP 并建图配置:
kr2_20260509_lidar_colmap_eval/configs/colmap_lightglue_manual_loop_000030_001700_stride4_skip934-985.env
```

已生成图像对计划：

```text
图像数: 405
匹配间隔: 1,2,4,6,8,12
图像对数: 2397
pair_schedule:
kr2_20260509_lidar_colmap_eval/outputs/superpoint_lightglue_manual_loop_000030_001700_stride4_skip934-985/reports/pair_schedule.csv
```

已完成运行结果：

```text
registered_images: 405/405
sparse_points: 28370
COLMAP mapping duration_sec: 263
trajectory:
kr2_20260509_lidar_colmap_eval/runs/superpoint_lightglue_colmap_manual_loop_000030_001700_stride4_skip934-985/results/trajectory_plots/trajectory.tum
```

匹配阶段耗时：

```text
device: cpu
images: 405
pairs: 2397
max_keypoints: 4096
feature_extract_sec: 392.792
lightglue_match_sec: 844.470
total_sec: 1266.094
total_min: 21.102
```

运行顺序：

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

python3 kr2_learned_descriptors_phase2/scripts/run_superpoint_lightglue.py \
  --config kr2_20260509_lidar_colmap_eval/configs/lightglue_manual_loop_000030_001700_stride4_skip934-985.env

bash kr2_learned_descriptors_phase2/scripts/run_colmap_mapper_lightglue.sh \
  kr2_20260509_lidar_colmap_eval/configs/colmap_lightglue_manual_loop_000030_001700_stride4_skip934-985.env
```

## LiDAR 辅助验证

LiDAR 地图文件：

```text
map/WHEELTEC.pgm
map/WHEELTEC.yaml
```

注意：不能直接把 `/odom_combined` 当作 `map` 坐标系叠加到 `WHEELTEC.pgm` 上。正确做法是从 TF 中合成：

```text
map -> odom_combined -> base_footprint
```

当前更精细的 map-frame 参考轨迹：

```text
kr2_20260509_lidar_colmap_eval/outputs/odom/map_odom_combined_projected.csv
```

正确对齐结果目录：

```text
kr2_20260509_lidar_colmap_eval/results/lidar_alignment_map_projected_dt05
```

报告可用可视化图：

```text
全局视图:
kr2_20260509_lidar_colmap_eval/results/lidar_alignment_map_projected_dt05/colmap_vs_lidar_map_report_full.png

裁剪放大视图:
kr2_20260509_lidar_colmap_eval/results/lidar_alignment_map_projected_dt05/colmap_vs_lidar_map_report_crop.png
```

图例说明：

```text
灰色背景: 2D LiDAR occupancy grid map
绿色: LiDAR/TF map-frame reference trajectory
蓝色: SIFT + COLMAP aligned trajectory
橙色: SuperPoint + LightGlue + COLMAP aligned trajectory
圆点: 起点
叉号: 终点
箭头: 运动方向
```

对齐方式：

```text
使用 /odom_combined 高频轨迹；
通过 TF 中的 map -> odom_combined 将其投影到 map 坐标系；
对参考轨迹进行时间插值；
视觉-参考时间窗口: 0.5 s；
参与对齐的视觉位姿数: 317/405。
```

对齐指标：

```text
SIFT + COLMAP:
rmse_m: 0.345436
mean_error_m: 0.302750
max_error_m: 0.764041
loop_closure_delta_m: 1.053462

SuperPoint + LightGlue:
rmse_m: 0.407321
mean_error_m: 0.351521
max_error_m: 0.924916
loop_closure_delta_m: 0.924251
```

当前结论：

```text
在 LiDAR/TF map-frame 轨迹辅助验证下，SIFT + COLMAP 比 SuperPoint + LightGlue 更贴近参考轨迹。
```

说明：

```text
曾尝试通过大幅放宽 map -> odom_combined TF 时间容差获得 400/405 个关联点，
但该版本会导致里程计参考轨迹局部变形，因此已剔除。
当前采用 0.5 s 视觉-参考时间窗口的平衡版本，317/405 个视觉位姿参与评估；
该版本下，SIFT 的 RMSE 和平均误差均低于 SuperPoint + LightGlue。
```

## 汇报用一句话

```text
目前 2026-05-09 数据包中，传统 SIFT + COLMAP 基线采用 30..1700、skip 934..985、stride4，共 405 张关键帧；
结果实现 405/405 全注册、82713 个稀疏点，并能看到完整一圈轨迹。
该结果不追求严格几何闭环，但满足当前“完整一圈、允许局部重合”的阶段目标，可进入 LiDAR 对齐评估。
```
