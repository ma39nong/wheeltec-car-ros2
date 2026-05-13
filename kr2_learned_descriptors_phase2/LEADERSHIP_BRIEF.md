# KR2 第二阶段汇报说明

## 当前目标

本目录用于推进 KR2 的第二阶段：

> 在 COLMAP 建图之前，先使用深度学习局部特征与匹配方法处理同一批图像，生成可评估的图像匹配结果，为后续写入 COLMAP database 并重新建图做准备。

第一阶段传统 COLMAP/SIFT 基线已经完成，因此本阶段不再继续大规模调 COLMAP 参数，而是固定数据和基线，验证深度学习描述子前端。

## 固定输入

```text
scene_09 的 236 张关键帧图像
data/scene_08_20260317_first_loop_then_stride4/first_loop_segment/keyframes_stride4_approx250/rgb
```

## 本阶段已完成

1. 已建立独立实验目录：

```text
kr2_learned_descriptors_phase2
```

2. 已固定第二阶段配置：

```text
kr2_learned_descriptors_phase2/configs/phase2.env
```

3. 已生成图像对匹配计划：

```text
kr2_learned_descriptors_phase2/outputs/scene_09_superpoint_lightglue/reports/pair_schedule.csv
```

当前计划覆盖：

```text
图像数量：236
匹配间隔：1, 4, 8
图像对数量：695
```

4. 已完成 SuperPoint + LightGlue 匹配：

```text
kr2_learned_descriptors_phase2/scripts/run_superpoint_lightglue.py
```

实际输出包括：

```text
features/                 SuperPoint 特征缓存
matches/                  每对图像的匹配结果
match_visualizations/     匹配可视化图
reports/pair_metrics.csv  匹配数量、内点数量、内点率等指标
```

5. 已将 learned matches 写入 COLMAP 数据库并完成 mapper：

```text
database.db:
kr2_learned_descriptors_phase2/runs/scene_09_superpoint_lightglue_colmap/workspace/colmap/database.db

sparse model:
kr2_learned_descriptors_phase2/runs/scene_09_superpoint_lightglue_colmap/workspace/colmap/sparse/0

metrics:
kr2_learned_descriptors_phase2/runs/scene_09_superpoint_lightglue_colmap/results/reports/experiment_metrics.csv
```

## 当前结果

```text
SuperPoint + LightGlue 匹配图像对：695
COLMAP mapper 注册图像：236 / 236
稀疏点数量：16523
本次端到端用时：92s
```

需要注意：

```text
之前出现过只注册 3 张图的错误结果。
经检查，该问题不是 LightGlue 匹配本身导致的，而是 COLMAP 数据库注入流程的问题：
1. 数据库里的相机内参仍是初始值，没有使用第一阶段基线优化后的内参。
2. two_view_geometries 不应手工写入，改为用 colmap matches_importer 做几何验证。
```

已修复后的流程采用：

```text
SuperPoint keypoints
+ LightGlue raw matches
+ 基线 refined camera intrinsics
+ COLMAP matches_importer 几何验证
+ COLMAP mapper
```

## 后续执行路线

复现实验执行：

```bash
cd /home/kiyi/COLMAP/wheeltec_colmap

python3 kr2_learned_descriptors_phase2/scripts/run_superpoint_lightglue.py \
  --config kr2_learned_descriptors_phase2/configs/phase2.env

bash kr2_learned_descriptors_phase2/scripts/run_colmap_mapper_lightglue.sh
```

运行完成后，重点检查：

```text
1. registered_images 是否保持 236 / 236
2. sparse_points 与 SIFT baseline 的差距
3. trajectory 可视化是否连贯
4. 弱纹理、转角、走廊区域是否存在明显误注册
```

## 需要确认的问题

请确认 KR2 的第二阶段是否按以下方向继续：

```text
使用 SuperPoint + LightGlue 替代 COLMAP 默认 SIFT 前端，
先生成深度学习特征与匹配结果，
再写入 COLMAP database.db，
最后运行 COLMAP mapper 并与当前 SIFT baseline 对比。
```

当前建议下一步：

```text
把 SIFT baseline 与 SuperPoint + LightGlue + COLMAP 的结果放到同一张对比表：
1. 注册图像数
2. 稀疏点数量
3. 运行时间
4. 轨迹形状
5. 弱纹理/转角区域表现
```
