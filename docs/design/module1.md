# 模块1实现计划：球场检测与渲染

## Context

YOLO 模型权重已训练完毕（`yolo/weights/court_yolo.pt` 和 `net_yolo.pt`），设计文档齐备（`docs/yolo_pose_detector.md`、`homography.md`、`court_renderer.md`、`base_definitions.md`）。现在需要实现模块1的全部代码：YOLO 推理封装、Homography 计算与验证、球场线条渲染，以及一个独立运行入口脚本。

## 需要创建的文件（10个）

```
config/
├── __init__.py                  # 空
└── court_config.py              # 球场几何常量（单一数据源）

utils/
├── __init__.py                  # 空
└── geometry.py                  # 几何工具函数

module1/
├── __init__.py                  # 空
├── yolo_detector.py             # 1.1 YOLO Pose 推理封装
├── homography.py                # 1.2 Homography 计算与验证
├── court_renderer.py            # 1.3 球场线条渲染
└── net_overlay.py               # 1.3 球网底部投影

scripts/
└── run_module1.py               # 独立运行入口（已存在 scripts/ 目录）
```

---

## Step 1: `config/court_config.py` — 球场几何常量

**目的**：所有模块共享的单一数据源，消除 `draw_court_diagram.py` 等脚本中的重复定义。

**内容**：
- BWF 标准尺寸常量（FULL_LENGTH=13.40, DOUBLES_WIDTH=6.10, ...）
- `COURT_KEYPOINTS_2D`: shape (22, 2) numpy 数组，plan 排序（索引 0-21 对应 base_definitions.md）
- `NET_KEYPOINTS_3D`: shape (4, 3) numpy 数组，plan 排序（全局索引 22-25）
- `COURT_LINE_SEGMENTS`: 13 条线段的 (i, j) 元组列表
- `KEYPOINT_NAMES`: 0-25 的中文名称字典
- `NUM_GROUND_KEYPOINTS = 22`, `NUM_NET_KEYPOINTS = 4`, `NUM_TOTAL_KEYPOINTS = 26`
- 模块级 assert 校验数组 shape

坐标数据来源：`docs/base_definitions.md` §3 和 `scripts/draw_court_diagram.py` 第 23-46 行（两者一致）。

---

## Step 2: `utils/geometry.py` — 几何工具

仅包含 `line_intersection(line1_coeffs, line2_coeffs)` 函数，求两条直线 ax+by+c=0 的交点。模块1暂不需要，但 `docs/implementation_guide.md` 阶段1要求创建此文件，模块2会用到。

---

## Step 3: `module1/yolo_detector.py` — YOLO Pose 推理封装

### 数据结构

```python
@dataclass
class KeypointDetection:
    index: int              # 0-25 全局索引
    pixel_xy: np.ndarray    # shape (2,)
    confidence: float
    visible: bool

@dataclass
class CourtDetectionResult:
    keypoints: list[KeypointDetection]   # 长度 26
    bbox: np.ndarray                      # shape (4,)
    bbox_confidence: float
    num_visible: int
    smoothed: bool = False
```

### CourtDetectionResult 辅助方法

- `get_visible_keypoints(min_conf=0.5)` → `list[KeypointDetection]`
- `to_pixel_array(min_conf=0.5)` → `(indices: ndarray, pixel_coords: ndarray)`
- `get_ground_keypoints(min_conf=0.5)` → `(indices, coords)` 仅索引 0-21
- `get_net_keypoints(min_conf=0.5)` → `(indices, coords)` 仅索引 22-25

### YoloPoseDetector 类

```python
class YoloPoseDetector:
    def __init__(self, court_model_path=None, net_model_path=None,
                 confidence_threshold=0.5, device=None)
    def detect(self, frame: np.ndarray) -> CourtDetectionResult | None
```

**关键实现细节**：
- 从 `training.keypoint_mapping` 导入 `DATASET_TO_PLAN_INDEX`（球场，22点）
- 从 `training.keypoint_mapping_net` 导入 `DATASET_TO_PLAN_INDEX`（球网，4点）
- 球场模型推理 → 取最高置信度检测 → 提取 shape (22, 3) 的 kpts_data → 按映射转换索引
- 球网模型推理 → 同上 → shape (4, 3) → 映射到全局索引 22-25
- **注意映射非顺序**：球网 ds1→24（非23）, ds2→23（非24）
- 合并为 26 关键点结果；任一模型失败时，对应关键点置信度=0
- 默认权重路径：`yolo/weights/court_yolo.pt` 和 `net_yolo.pt`

---

## Step 4: `module1/homography.py` — Homography 计算与验证

### 数据结构

```python
@dataclass
class HomographyResult:
    H: np.ndarray              # (3,3) 球场→像素
    H_inv: np.ndarray          # (3,3) 像素→球场
    inlier_mask: np.ndarray    # (N,) bool
    num_inliers: int
    num_correspondences: int
    reprojection_error: float  # 内点平均误差 (px)
    used_indices: np.ndarray
    used_court_pts: np.ndarray
    used_pixel_pts: np.ndarray
```

### 函数

```python
def compute_homography(detection, court_keypoints_2d, min_confidence=0.5,
                       ransac_threshold=5.0) -> HomographyResult | None

def validate_homography(result: HomographyResult) -> dict
```

**compute_homography**：
1. 调用 `detection.get_ground_keypoints(min_confidence)` 获取高置信度地面关键点
2. 不足4个点 → 返回 None
3. ≥5 个点用 `cv2.findHomography(RANSAC)`；恰好4个点用 method=0（最小二乘）
4. `cv2.perspectiveTransform` 计算重投影误差
5. `np.linalg.inv(H)` 计算逆矩阵

**validate_homography** 5 项检查：

| # | 检查项 | 阈值 | 说明 |
|---|--------|------|------|
| 1 | 重投影误差 | < 5.0 px | |
| 2 | 内点比例 | > 0.7 | |
| 3 | abs(det(H)) | > 1e-10 | 球场Y向上→像素Y向下，det<0 正常 |
| 4 | 条件数 | < 1e6 | |
| 5 | 奇异值比(max/min) | < 1e5 | 真实透视变换 SV 比可达 1e4 量级 |

返回 `dict`，含 `overall_ok` 布尔值。不通过则记录警告日志。

---

## Step 5: `module1/court_renderer.py` — 球场线条渲染

### 核心工具函数

```python
def project_points_batch(H, court_pts) -> np.ndarray
    # 使用 cv2.perspectiveTransform 批量投影，输入 float32

def project_point(H, court_pt) -> np.ndarray
    # 单点投影（内部调用 project_points_batch）
```

### 渲染函数

```python
def draw_court_overlay(frame, H, keypoints, segments, color=(0,255,0),
                       thickness=2, num_samples=21) -> np.ndarray
```

- 遍历 13 条线段，每条线段沿线采样 21 个点
- `np.outer(1-t, pt_a) + np.outer(t, pt_b)` 生成采样点
- 批量投影后用 `cv2.line` 连续绘制，使用 `cv2.LINE_AA` 抗锯齿
- 返回帧副本（不修改原帧）

### 调试辅助

```python
def draw_keypoint_markers(frame, detection, min_conf=0.5, color=(0,0,255),
                          radius=5) -> np.ndarray
```

在帧上标注检测到的关键点位置和编号。

---

## Step 6: `module1/net_overlay.py` — 球网底部投影

```python
def get_net_post_base_pixels(H) -> tuple[np.ndarray, np.ndarray]
```

- 投影球网立柱底部 (-3.05, 0.0) 和 (+3.05, 0.0) 到像素坐标
- 调用 `module1.court_renderer.project_points_batch`
- 返回 `(left_base_px, right_base_px)` 各 shape (2,)
- 供模块2的 `net_top_detector_yolo.py` 使用

---

## Step 7: `scripts/run_module1.py` — 独立运行入口

支持图片和视频输入，命令行参数：

```bash
python -m scripts.run_module1 --input video.mp4 --output output.mp4
python -m scripts.run_module1 --input frame.jpg --output result.jpg [--show-keypoints]
```

处理流程（每帧）：
1. `detector.detect(frame)` → CourtDetectionResult
2. `compute_homography(detection, COURT_KEYPOINTS_2D)` → HomographyResult
3. `validate_homography(h_result)` → metrics
4. `draw_court_overlay(frame, H, ...)` → 叠加帧
5. 可选 `draw_keypoint_markers()` 标注关键点

视频模式：逐帧处理，每100帧打印进度和成功率。

---

## 实现顺序与验证

| 步骤 | 文件 | 验证方式 |
|------|------|----------|
| 1 | `config/__init__.py` + `court_config.py` | `from config.court_config import COURT_KEYPOINTS_2D; assert shape==(22,2)` |
| 2 | `utils/__init__.py` + `geometry.py` | `from utils.geometry import line_intersection` |
| 3 | `module1/__init__.py` + `yolo_detector.py` | 加载两个模型成功，对测试图推理返回 CourtDetectionResult |
| 4 | `module1/homography.py` | 合成测试：已知H投影22点加噪声，恢复H，重投影 < 3px |
| 5 | `module1/court_renderer.py` | `from module1.court_renderer import project_points_batch` |
| 6 | `module1/net_overlay.py` | 导入成功 |
| 7 | `scripts/run_module1.py` | 对数据集中图片运行端到端测试 |

## 文档保存

实现前，先将本设计文档保存到 `docs/design/module1.md`，作为模块1的实现设计参考文档。

---

## 关键设计决策

1. **索引映射复用**：从 `training.keypoint_mapping` 和 `training.keypoint_mapping_net` 导入映射，不重复定义
2. **部分失败容忍**：球网模型失败时仍返回球场关键点（置信度=0 的球网点）
3. **cv2.perspectiveTransform**：替代手写齐次坐标乘法，数值稳定且支持批量
4. **验证与计算分离**：`compute_homography` 和 `validate_homography` 独立调用，单一职责
5. **project_points_batch 定义在 court_renderer.py**：作为项目通用投影工具，后续模块2也导入使用
