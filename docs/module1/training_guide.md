# YOLO Pose 训练指南

## 1. 概览

本项目使用两个独立的 YOLO Pose 模型，各自训练、各自推理：

| 项目 | 球场关键点模型 | 球网关键点模型 |
|------|---------------|---------------|
| 目标 | 检测22个地面关键点 | 检测4个球网角点 |
| 数据集 | yolo/datasets/court | yolo/datasets/net |
| 来源 | Roboflow | Roboflow |
| 总图片数 | 1,194 张 | 495 张 |
| kpt_shape | [22, 3] | [4, 3] |
| 类别 | badminton_court | net |

两个模型共享相同的训练框架（`training/` 模块），通过 `--model-type` 参数切换。

### 三个集合的作用

| 集合 | 用途 | 说明 |
|------|------|------|
| **train** (训练集) | 模型从中学习参数 | 参与梯度更新 |
| **valid** (验证集) | 训练过程中评估泛化能力 | 每个 epoch 结束后评估，影响早停和超参数调整，但**不参与梯度更新** |
| **test** (测试集) | 训练完成后的最终评估 | 只在最终评估时使用一次，是对模型真实能力的无偏估计 |

---

## 2. 关键点排序详解

### 2.1 排序为什么重要

YOLO Pose 模型中，每个关键点通过**索引号**标识语义。排序影响训练和推理的每个环节：

| 环节 | 影响 | 说明 |
|------|------|------|
| **训练** | 模型学习"索引N = 某个语义特征" | 索引0在所有标注中必须一致地指向同一个物理位置 |
| **flip_idx** | 水平翻转时左右关键点互换 | 必须知道哪个索引是左、哪个是右，才能写出正确的 flip_idx |
| **推理** | 模型输出按训练时的索引顺序 | 如果不知道排序，无法正确解读输出中每个关键点的含义 |

**如果排序错误**：flip_idx 写错 → 水平翻转增强时左右关键点不交换或错误交换 → 训练数据被污染 → 模型学到错误的映射 → 推理结果无意义。

### 2.2 Roboflow 标注的排序机制

在 Roboflow 平台标注关键点的流程：

1. **创建骨架模板**：定义 N 个关键点，每个点有名称和固定索引。**添加顺序 = 导出顺序**。
2. **标注图片**：将模板中的点拖拽到图像对应位置。标注动作不改变索引顺序。
3. **导出数据**：标注文件中关键点按模板定义的固定顺序排列。

**YOLO 格式的局限**：导出后的 `.txt` 标注文件只包含数值坐标，不包含点名信息。`data.yaml` 中也只有 `kpt_shape` 和 `flip_idx`，没有每个索引对应什么语义点的说明。

### 2.3 如何确定和验证排序（关键环节）

每次导入新数据集时，**必须**确认关键点的排序。推荐以下验证方法（按可靠性排序）：

**方法 1：查看 Roboflow 项目设置**（最权威）

在 Roboflow UI → 项目设置 → Annotation → Keypoint Skeleton 页面，可以看到骨架模板中每个点的名称和定义顺序。这是排序的权威来源。

**方法 2：可视化验证脚本**（最直观）

编写脚本读取标注文件，在原图上画出每个关键点并标注索引号，通过人眼直观确认每个索引对应哪个物理位置。

**方法 3：坐标统计分析**（辅助验证）

分析多个标注文件中关键点坐标的空间分布规律。例如，如果某个索引在所有样本中 x 值都偏小、y 值都偏小，则它大概率是"左上"的点。

> **建议**：每个新数据集导入时，至少用方法 1 确认，再用方法 2 或 3 双重验证。排序确认后记录在 `keypoint_mapping` 模块中。

### 2.4 排序不需要和文档排序一致

项目中存在两套排序体系：

| 排序体系 | 定义来源 | 用途 |
|---------|---------|------|
| **数据集排序** | Roboflow 骨架模板 | 训练标注、模型输入输出 |
| **文档排序** | base_definitions.md | 全局统一的逻辑编号，下游模块使用 |

两者可以不同。`training/keypoint_mapping.py`（球场）和 `training/keypoint_mapping_net.py`（球网）负责在两套排序间转换。

**核心原则**：保持数据集原始排序不变，永远不修改标注文件，仅在推理后通过映射模块转换。

---

## 3. 球场关键点模型（22点）

### 3.1 数据集概览

| 项目 | 值 |
|------|-----|
| 来源 | Roboflow (xingxings-workspace) |
| 格式 | YOLOv8 Pose |
| 关键点 | 22 个地面点 (Z=0) |
| 总计 | 1,194 张 |
| 默认拆分 | train 77% / valid 17% / test 6% |

### 3.2 关键点排序映射

数据集采用**行扫描排序**（远端→近端，每行从左到右），与 `base_definitions.md` 的结构分组排序不同。

```
数据集索引 → 全局索引 → 名称                    → 世界坐标(X, Y)
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

推理时通过 `training/keypoint_mapping.py` 中的 `dataset_to_plan()` 转换为全局排序。

### 3.3 flip_idx

```
court_flip_idx = [4, 3, 2, 1, 0, 6, 5, 9, 8, 7, 11, 10, 14, 13, 12, 16, 15, 21, 20, 19, 18, 17]
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

## 4. 球网关键点模型（4点）

### 4.1 数据集概览

| 项目 | 值 |
|------|-----|
| 来源 | Roboflow (xingxings-workspace, net-h9lgg) |
| 格式 | YOLOv8 Pose |
| 关键点 | 4 个球网角点 |
| 总计 | 495 张 |
| 当前状态 | 仅有 train 集，需拆分 |

### 4.2 关键点排序映射

通过分析标注文件坐标的空间分布确认排序（需到 Roboflow 项目设置二次确认）：

```
数据集索引 → 全局索引 → 名称          → 世界坐标(X, Y, Z)      → 确认依据
──────────────────────────────────────────────────────────────────────────────
ds0 → plan22 → 网左端顶部    → (-3.05, 0.00, 1.55)  → x小, y小(图像上方)
ds1 → plan24 → 网右端顶部    → (+3.05, 0.00, 1.55)  → x大, y小(图像上方)
ds2 → plan23 → 网左端底部    → (-3.05, 0.00, 0.00)  → x小, y大(图像下方)
ds3 → plan25 → 网右端底部    → (+3.05, 0.00, 0.00)  → x大, y大(图像下方)
```

> **注意**：数据集排序（ds0=左上, ds1=右上, ds2=左下, ds3=右下）与全局编号排序（22=左上, 23=左下, 24=右上, 25=右下）不同。这是正常的，由 `training/keypoint_mapping_net.py` 中的映射处理。

推理时通过 `training/keypoint_mapping_net.py` 中的 `dataset_to_plan()` 转换为全局排序。

### 4.3 flip_idx

```
net_flip_idx = [1, 0, 3, 2]
```

| 交换对 | 说明 |
|--------|------|
| ds0 ↔ ds1 | 网左端顶部 ↔ 网右端顶部 |
| ds2 ↔ ds3 | 网左端底部 ↔ 网右端底部 |

---

## 5. 通用训练流程

以下内容对球场模型和球网模型通用，通过 `--model-type court|net` 切换。

### 5.1 数据集拆分

所有原始数据统一存储在 `all/` 目录中。通过 `training/split_dataset.py` 按可配置比例随机拆分，生成文本文件列表供 YOLO 读取，**数据文件本身不做任何移动或复制**。

#### 目录结构

```
<dataset>/
├── data.yaml          ← train/val/test 指向 .txt 文件
├── train.txt          ← 训练集图片路径列表
├── val.txt            ← 验证集图片路径列表
├── test.txt           ← 测试集图片路径列表
└── all/               ← 唯一的数据存储目录
    ├── images/        ← 所有图片
    └── labels/        ← 所有标注文件
```

#### 独立使用

```bash
# 球场数据集
python -m training.split_dataset --dataset-dir yolo/datasets/court

# 球网数据集
python -m training.split_dataset --dataset-dir yolo/datasets/net

# 自定义比例
python -m training.split_dataset --dataset-dir <dataset> \
    --train-ratio 0.8 --valid-ratio 0.15 --test-ratio 0.05
```

#### 设计要点

1. **零数据移动** — 拆分仅生成 txt 文件列表，all/ 中的数据文件不做任何移动或复制
2. **幂等性** — 如果 `all/` 已存在，直接从中读取数据重新拆分，不重复合并
3. **可复现** — 固定随机种子保证相同的拆分结果
4. **比例验证** — 三个比例之和必须等于 1.0（允许 ±0.01 误差）

### 5.2 模型选择

| 大小 | 预训练权重 | 参数量 | 推理速度 | 适用场景 |
|------|-----------|--------|----------|----------|
| nano | yolov8n-pose.pt | ~3M | 最快 | 快速原型验证 |
| **small** | **yolov8s-pose.pt** | **~11M** | **快** | **推荐起点（当前默认）** |
| medium | yolov8m-pose.pt | ~26M | 中等 | 更高精度 |
| large | yolov8l-pose.pt | ~44M | 较慢 | 高精度需求 |
| xlarge | yolov8x-pose.pt | ~69M | 最慢 | 最高精度 |

> 球网模型只有4个关键点，结构较简单，nano 或 small 通常足够。

### 5.3 硬件配置

训练脚本默认自动检测设备（CUDA > MPS > CPU）。

| 预设 | 设备 | 批次大小 | 说明 |
|------|------|---------|------|
| `cpu` | CPU | 8 | Intel Mac / 无GPU |
| `mps` | MPS | 16 | Apple Silicon (M1/M2/M3/M4) |
| `gpu_low` | CUDA:0 | 16 | NVIDIA GPU (4-6GB 显存) |
| `gpu` | CUDA:0 | 32 | NVIDIA GPU (8GB+) |

### 5.4 训练参数

#### 默认超参数

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

#### 损失权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| box | 7.5 | 边界框回归损失 |
| cls | 0.5 | 分类损失（单类可适当降低） |
| pose | 12.0 | 关键点坐标损失（重要，设较高值） |
| kobj | 1.0 | 关键点目标性损失 |

### 5.5 数据增强策略

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

**增强策略说明**：

1. **水平翻转**是最重要的增强 — 球场和球网都是左右对称的。前提是 `flip_idx` 正确。
2. **不做垂直翻转** — 摄像机总是从一侧拍摄，上下翻转后透视关系完全反转。
3. **Mosaic** 在训练后期关闭（`close_mosaic=10`），让模型在最后10轮学习完整图像。
4. **随机擦除**模拟球员、裁判等遮挡。
5. **透视变换极小** — 数据集本身已经包含了不同视角。

### 5.6 使用指南

#### 环境准备

```bash
pip install ultralytics>=8.1.0 pyyaml
```

#### 快速开始

```bash
cd /path/to/airforce

# ── 球场模型 ──

# 冒烟测试
python -m training.train --dataset-dir yolo/datasets/court \
    --smoke-test --model-size nano

# 正式训练（默认 small 模型，自动检测设备）
python -m training.train --dataset-dir yolo/datasets/court

# 指定配置
python -m training.train --dataset-dir yolo/datasets/court \
    --model-size medium --profile mps --epochs 300

# ── 球网模型 ──

# 冒烟测试
python -m training.train --dataset-dir yolo/datasets/net \
    --model-type net --smoke-test --model-size nano

# 正式训练
python -m training.train --dataset-dir yolo/datasets/net --model-type net

# 先拆分数据再训练
python -m training.train --dataset-dir yolo/datasets/net --model-type net \
    --split --train-ratio 0.8 --valid-ratio 0.15 --test-ratio 0.05
```

#### 数据验证（独立运行）

```bash
# 球场数据集
python -m training.prepare_data --dataset-dir yolo/datasets/court --fix-yaml

# 球网数据集
python -m training.prepare_data --dataset-dir yolo/datasets/net --model-type net --fix-yaml

# 仅验证
python -m training.prepare_data --dataset-dir yolo/datasets/net --model-type net --validate-only
```

#### 恢复训练

```bash
python -m training.train --dataset-dir <dataset> --model-type <court|net> --resume
```

#### 训练输出

训练结果保存在 `yolo/runs/<name>/`：

```
yolo/runs/<court|net>/
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

## 6. 调优建议

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

## 7. 关键设计决策

1. **保持数据集原始排序** — 不修改标注文件，避免引入重映射错误。在推理阶段通过 `keypoint_mapping*.py` 转换到全局排序。
2. **两个模型完全独立** — 球场和球网各自训练、各自推理，通过全局编号体系（base_definitions.md）在下游融合。
3. **AdamW 优化器** — 比 SGD 对学习率更不敏感，初始学习率设为 0.001。
4. **余弦退火 + 预热** — 前5轮线性预热，之后余弦衰减到 lr0 × 0.01。
5. **关键点损失权重=12** — 高于默认值，因为关键点精度是核心目标。
6. **单类分类损失=0.5** — 只有一个类别，分类简单。
7. **排序验证是必须的** — 每次从 Roboflow 导入新数据集，必须确认关键点排序并记录到映射模块。
