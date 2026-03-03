# YOLO Pose 训练指南

## 1. 数据集概览

| 项目 | 值 |
|------|-----|
| 来源 | Roboflow (xingxings-workspace) |
| 格式 | YOLOv8 Pose |
| 类别 | 1 (badminton_court) |
| 关键点 | 22 个地面点 (Z=0) |
| 总计 | 1,194 张 |
| 默认拆分 | train 77% / valid 17% / test 6% |

### 三个集合的作用

| 集合 | 用途 | 说明 |
|------|------|------|
| **train** (训练集) | 模型从中学习参数 | 参与梯度更新 |
| **valid** (验证集) | 训练过程中评估泛化能力 | 每个 epoch 结束后评估，影响早停和超参数调整，但**不参与梯度更新** |
| **test** (测试集) | 训练完成后的最终评估 | 只在最终评估时使用一次，是对模型真实能力的无偏估计 |

---

## 2. 关键点排序映射

数据集采用**行扫描排序**（远端→近端，每行从左到右），与 `plan.md` / `base_definitions.md` 的结构分组排序不同。

### 完整映射表

```
数据集索引 → plan索引 → 名称                    → 世界坐标(X, Y)
─────────────────────────────────────────────────────────────────
第1行：远端底线（5个点）
ds0  → plan0  → 远端左双打角                    → (-3.05, +6.70)
ds1  → plan4  → 远端左单打角                    → (-2.59, +6.70)
ds2  → plan20 → 远端底线中点                    → ( 0.00, +6.70)
ds3  → plan5  → 远端右单打角                    → (+2.59, +6.70)
ds4  → plan1  → 远端右双打角                    → (+3.05, +6.70)

第2行：远端双打后发球线（2个点）
ds5  → plan14 → 远端左后发球角(双打)            → (-3.05, +5.94)
ds6  → plan15 → 远端右后发球角(双打)            → (+3.05, +5.94)

第3行：远端前发球线（3个点）
ds7  → plan10 → 远端左前发球线端                → (-3.05, +1.98)
ds8  → plan18 → 远端中线/前发球线交点           → ( 0.00, +1.98)
ds9  → plan11 → 远端右前发球线端                → (+3.05, +1.98)

第4行：球网地面线（2个点）
ds10 → plan8  → 网左端底部                      → (-3.05,  0.00)
ds11 → plan9  → 网右端底部                      → (+3.05,  0.00)

第5行：近端前发球线（3个点）
ds12 → plan12 → 近端左前发球线端                → (-3.05, -1.98)
ds13 → plan19 → 近端中线/前发球线交点           → ( 0.00, -1.98)
ds14 → plan13 → 近端右前发球线端                → (+3.05, -1.98)

第6行：近端双打后发球线（2个点）
ds15 → plan16 → 近端左后发球角(双打)            → (-3.05, -5.94)
ds16 → plan17 → 近端右后发球角(双打)            → (+3.05, -5.94)

第7行：近端底线（5个点）
ds17 → plan2  → 近端左双打角                    → (-3.05, -6.70)
ds18 → plan6  → 近端左单打角                    → (-2.59, -6.70)
ds19 → plan21 → 近端底线中点                    → ( 0.00, -6.70)
ds20 → plan7  → 近端右单打角                    → (+2.59, -6.70)
ds21 → plan3  → 近端右双打角                    → (+3.05, -6.70)
```

### 推理时的转换

模型输出使用数据集排序。在接入 Module 1 (Homography) 等下游模块前，需通过 `training/keypoint_mapping.py` 中的 `dataset_to_plan()` 函数转换为 plan 排序。

---

## 3. flip_idx 说明

水平翻转时，左右对称的关键点需要交换索引。

**正确的 flip_idx（已在 data.yaml 中修正）**：
```
[4, 3, 2, 1, 0, 6, 5, 9, 8, 7, 11, 10, 14, 13, 12, 16, 15, 21, 20, 19, 18, 17]
```

| 交换对 | 说明 |
|--------|------|
| ds0 ↔ ds4 | 远端左双打角 ↔ 远端右双打角 |
| ds1 ↔ ds3 | 远端左单打角 ↔ 远端右单打角 |
| ds2 → ds2 | 远端底线中点（中线上，不变） |
| ds5 ↔ ds6 | 远端左后发球 ↔ 远端右后发球 |
| ds7 ↔ ds9 | 远端左前发球 ↔ 远端右前发球 |
| ds8 → ds8 | 远端中线交点（中线上，不变） |
| ds10 ↔ ds11 | 网左端 ↔ 网右端 |
| ds12 ↔ ds14 | 近端左前发球 ↔ 近端右前发球 |
| ds13 → ds13 | 近端中线交点（中线上，不变） |
| ds15 ↔ ds16 | 近端左后发球 ↔ 近端右后发球 |
| ds17 ↔ ds21 | 近端左双打角 ↔ 近端右双打角 |
| ds18 ↔ ds20 | 近端左单打角 ↔ 近端右单打角 |
| ds19 → ds19 | 近端底线中点（中线上，不变） |

---

## 4. 数据集拆分

所有原始数据统一存储在 `all/` 目录中。通过 `training/split_dataset.py` 按可配置比例随机拆分，生成文本文件列表供 YOLO 读取，**数据文件本身不做任何移动或复制**。

### 目录结构

```
BadmintonCourtDetection.yolov8/
├── data.yaml          ← train/val/test 指向 .txt 文件
├── train.txt          ← 训练集图片路径列表
├── val.txt            ← 验证集图片路径列表
├── test.txt           ← 测试集图片路径列表
└── all/               ← 唯一的数据存储目录
    ├── images/        ← 所有 1,194 张图片
    └── labels/        ← 所有 1,194 个标注文件
```

### 工作原理

拆分脚本将随机选中的图片路径写入 txt 文件，例如 `train.txt` 内容：

```
./all/images/badminton_-14-_png_jpg.rf.0999caa3eb5f753344209a2556e28ecd.jpg
./all/images/match_02_05770_jpg.rf.4c427aa5d98f276b4d0083e7b0294872.jpg
...
```

`data.yaml` 引用这些文件：

```yaml
train: train.txt
val: val.txt
test: test.txt
```

YOLO 读到 `.txt` 路径时，逐行加载图片，并自动将 `images/` 替换为 `labels/` 找到对应标注。

### 独立使用

```bash
# 默认比例 (train=77%, valid=17%, test=6%)
python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8

# 自定义比例
python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8 \
    --train-ratio 0.8 --valid-ratio 0.15 --test-ratio 0.05

# 使用不同随机种子（产生不同的拆分结果）
python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8 --seed 123
```

### 配合训练使用

在 `train.py` 中通过 `--split` 参数在训练前自动执行拆分：

```bash
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 \
    --split --train-ratio 0.8 --valid-ratio 0.15 --test-ratio 0.05
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-dir` | (必填) | 数据集根目录路径 |
| `--train-ratio` | 0.77 | 训练集比例 |
| `--valid-ratio` | 0.17 | 验证集比例 |
| `--test-ratio` | 0.06 | 测试集比例 |
| `--seed` | 42 | 随机种子（相同种子 = 相同拆分结果） |

### 设计要点

1. **零数据移动** — 拆分仅生成 txt 文件列表，all/ 中的数据文件不做任何移动或复制
2. **幂等性** — 如果 `all/` 已存在，直接从中读取数据重新拆分，不重复合并
3. **可复现** — 固定随机种子保证相同的拆分结果
4. **比例验证** — 三个比例之和必须等于 1.0（允许 ±0.01 误差）

---

## 5. 模型选择

| 大小 | 预训练权重 | 参数量 | 推理速度 | 适用场景 |
|------|-----------|--------|----------|----------|
| nano | yolov8n-pose.pt | ~3M | 最快 | 快速原型验证 |
| **small** | **yolov8s-pose.pt** | **~11M** | **快** | **推荐起点（当前默认）** |
| medium | yolov8m-pose.pt | ~26M | 中等 | 更高精度 |
| large | yolov8l-pose.pt | ~44M | 较慢 | 高精度需求 |
| xlarge | yolov8x-pose.pt | ~69M | 最慢 | 最高精度 |

通过 `--model-size` 参数切换：
```bash
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 --model-size medium
```

---

## 6. 硬件配置

### 自动检测

训练脚本默认自动检测设备（CUDA > MPS > CPU）。

### 手动指定

| 预设 | 设备 | 批次大小 | 说明 |
|------|------|---------|------|
| `cpu` | CPU | 8 | Intel Mac / 无GPU |
| `mps` | MPS | 16 | Apple Silicon (M1/M2/M3/M4) |
| `gpu_low` | CUDA:0 | 16 | NVIDIA GPU (4-6GB 显存) |
| `gpu` | CUDA:0 | 32 | NVIDIA GPU (8GB+) |

```bash
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 --profile cpu
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 --profile mps
```

---

## 7. 训练参数

### 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 200 | 训练总轮数 |
| patience | 50 | 早停轮数（验证指标无改善则停止） |
| imgsz | 640 | 输入图像尺寸 |
| optimizer | AdamW | 优化器 |
| lr0 | 0.001 | 初始学习率 |
| lrf | 0.01 | 最终学习率因子 (final_lr = lr0 × lrf) |
| warmup_epochs | 5.0 | 预热轮数 |
| cos_lr | True | 余弦退火调度 |
| close_mosaic | 10 | 最后 N 轮关闭 Mosaic |

### 损失权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| box | 7.5 | 边界框回归损失 |
| cls | 0.5 | 分类损失（单类可适当降低） |
| pose | 12.0 | 关键点坐标损失（重要，设较高值） |
| kobj | 1.0 | 关键点目标性损失 |

### 参数覆盖

所有参数均可通过命令行覆盖：
```bash
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 \
    --epochs 300 --batch 8 --lr0 0.0005
```

---

## 8. 数据增强策略

| 增强方法 | 参数 | 默认值 | 目的 |
|---------|------|--------|------|
| 水平翻转 | fliplr | 0.5 | 左右视角对称，最重要的增强 |
| 垂直翻转 | flipud | 0.0 | **不启用**（上下翻转后透视关系不合理） |
| Mosaic | mosaic | 1.0 | 拼接4张图，增加上下文多样性 |
| Mixup | mixup | 0.1 | 低概率图像混合，增强泛化能力 |
| 旋转 | degrees | 10.0 | 应对手持拍摄的轻微倾斜 |
| 平移 | translate | 0.1 | 球场不在画面中央 |
| 缩放 | scale | 0.5 | 应对不同拍摄距离 |
| 剪切 | shear | 2.0 | 轻微几何形变 |
| 透视 | perspective | 0.0005 | 极小透视畸变 |
| 色调抖动 | hsv_h | 0.015 | 应对不同灯光色温 |
| 饱和度抖动 | hsv_s | 0.7 | 应对不同场馆灯光强度 |
| 亮度抖动 | hsv_v | 0.4 | 应对明暗变化 |
| 随机擦除 | erasing | 0.3 | 模拟球员/裁判遮挡球场线 |

### 增强策略设计考量

1. **水平翻转** 是最重要的增强 — 球场左右对称，可将训练数据等效翻倍。前提是 `flip_idx` 正确。
2. **不做垂直翻转** — 摄像机总是从一侧拍摄，上下翻转后近大远小的透视关系完全反转，不符合真实场景。
3. **Mosaic 增强** 在训练后期关闭（`close_mosaic=10`），让模型在最后10轮学习完整图像。
4. **随机擦除** 模拟球员、裁判等遮挡球场线的场景，增强模型在遮挡情况下的鲁棒性。
5. **透视变换极小** — 数据集本身已经包含了不同视角的透视效果，过度变换会产生不合理的几何关系。

---

## 9. 使用指南

### 环境准备

```bash
pip install ultralytics>=8.1.0 pyyaml
```

### 快速开始

```bash
# 从项目根目录运行
cd /path/to/airforce

# 1. 冒烟测试（nano 模型 + 5% 数据 + 2 轮，仅验证流程可走通）
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 \
    --smoke-test --model-size nano

# 2. 正式训练（small 模型，自动检测设备）
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8

# 3. 指定配置
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 \
    --model-size medium --profile mps --epochs 300

# 4. 先拆分数据再训练（自定义比例）
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 \
    --split --train-ratio 0.8 --valid-ratio 0.15 --test-ratio 0.05
```

### 数据拆分（独立运行）

```bash
# 合并所有数据到 all/ 并按默认比例拆分
python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8

# 自定义比例和随机种子
python -m training.split_dataset --dataset-dir BadmintonCourtDetection.yolov8 \
    --train-ratio 0.8 --valid-ratio 0.15 --test-ratio 0.05 --seed 123
```

### 数据验证（独立运行）

```bash
# 验证数据集并修正 flip_idx
python -m training.prepare_data --dataset-dir BadmintonCourtDetection.yolov8 --fix-yaml

# 仅验证
python -m training.prepare_data --dataset-dir BadmintonCourtDetection.yolov8 --validate-only
```

### 恢复训练

```bash
python -m training.train --dataset-dir BadmintonCourtDetection.yolov8 --resume
```

### 训练输出

训练结果保存在 `runs/pose/badminton_court/`：

```
runs/pose/badminton_court/
├── weights/
│   ├── best.pt         # 最佳模型（验证指标最优）
│   └── last.pt         # 最后一轮模型
├── results.csv         # 每轮训练指标
├── confusion_matrix.png
├── results.png         # 训练曲线图
├── val_batch*_pred.jpg # 验证集预测可视化
└── args.yaml           # 实际使用的训练参数
```

---

## 10. 调优建议

### 如果模型欠拟合（训练/验证 loss 都高）

- 增大模型：`--model-size medium` 或 `large`
- 增加训练轮数：`--epochs 300`
- 减小学习率：`--lr0 0.0005`

### 如果模型过拟合（训练 loss 低但验证 loss 高）

- 增加数据增强（可在 `config.py` 中调大 `mixup`、`erasing`）
- 减小模型：`--model-size nano`
- 增大 `patience` 让早停更保守

### 如果训练内存不足（OOM）

- 减小批次大小：`--batch 4`
- 减小图像尺寸：`--imgsz 480`
- 使用更小模型：`--model-size nano`

---

## 11. 关键设计决策

1. **保持数据集原始排序** — 不修改标注文件，避免引入重映射错误。在推理阶段通过 `keypoint_mapping.py` 转换到 plan 排序。
2. **AdamW 优化器** — 比 SGD 对学习率更不敏感，初始学习率设为 0.001（比 SGD 的 0.01 小一个数量级）。
3. **余弦退火 + 预热** — 前5轮线性预热，之后余弦衰减到 lr0 × 0.01，平滑收敛。
4. **关键点损失权重=12** — 高于默认值，因为关键点精度是本任务的核心目标。
5. **单类分类损失=0.5** — 只有一个类别（badminton_court），分类相对简单，不需要过高权重。
