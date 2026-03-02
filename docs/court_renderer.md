# 子模块1.3：球场线条渲染 & 球网地面线投影

## 功能概述

利用 Homography 矩阵 H 将标准球场的所有线段投影到视频帧上并绘制，实现球场线条叠加渲染效果。同时提供球网立柱底部像素坐标的投影函数，供模块2的球网端点推断使用。

## 依赖关系

- **上游**：Homography 计算(1.2) → `HomographyResult.H`
- **下游**：球网端点推断(2.4)（使用 `get_net_post_base_pixels`）
- **基础数据**：[base_definitions.md](base_definitions.md) 中的 `COURT_KEYPOINTS_2D` 和 `COURT_LINE_SEGMENTS`

## 对应代码文件

`module1/court_renderer.py`、`module1/net_overlay.py`

---

## 输入/输出数据结构

**球场渲染**：
- 输入：BGR帧、H矩阵、关键点坐标数组、线段定义
- 输出：叠加球场线条的帧图像

**球网底部投影**：
- 输入：H矩阵
- 输出：球网立柱左/右底部的像素坐标 `(ndarray(2,), ndarray(2,))`

## 算法详解

### 批量点投影工具函数

使用 `cv2.perspectiveTransform` 批量投影，避免手写矩阵乘法：

```
function project_points_batch(H, court_pts):
    """批量投影球场坐标到像素坐标。
    court_pts: shape (N, 2)
    返回: shape (N, 2) 像素坐标
    """
    pts = court_pts.reshape(1, -1, 2).astype(np.float32)
    result = cv2.perspectiveTransform(pts, H.astype(np.float32))
    return result.reshape(-1, 2)

function project_point(H, court_pt):
    """单点投影（内部调用 cv2.perspectiveTransform）"""
    return project_points_batch(H, court_pt.reshape(1, 2))[0]
```

> **设计决策**：使用 `cv2.perspectiveTransform` 而非手写齐次坐标乘法。该函数内部处理了齐次除法和数值稳定性，且支持批量操作。

### 球场线条渲染

```
function draw_court_overlay(frame, H, keypoints, segments, color=(0,255,0)):
    overlay = frame.copy()
    for (i, j) in segments:
        pt_a = keypoints[i]
        pt_b = keypoints[j]
        # 沿线段均匀采样21个点（20个子段）
        t_values = np.linspace(0, 1, 21)
        sample_pts = np.array([pt_a * (1-t) + pt_b * t for t in t_values])  # (21, 2)
        # 批量投影
        pixel_pts = project_points_batch(H, sample_pts)  # (21, 2)
        # 绘制连续线段
        for k in range(20):
            p0 = pixel_pts[k].astype(int)
            p1 = pixel_pts[k+1].astype(int)
            cv2.line(overlay, tuple(p0), tuple(p1), color, 2)
    return overlay
```

> **说明**：Homography 保持直线性（射影变换的基本性质），理论上每条线段投影两个端点再连线即可。但采样多个中间点可以处理极端透视下的浮点精度问题，且计算开销极小。

### 球网地面线投影

球网地面线连接关键点8和9，已包含在 `COURT_LINE_SEGMENTS` 中，会被 `draw_court_overlay` 自动绘制。额外提供独立函数供模块2使用：

```
function get_net_post_base_pixels(H):
    # 批量投影两个球网底部点
    net_bases = np.array([[-3.05, 0.0], [+3.05, 0.0]])
    pixels = project_points_batch(H, net_bases)   # 使用 cv2.perspectiveTransform
    return pixels[0], pixels[1]
```

## 设计要点

- `project_points_batch` 是全项目通用的投影工具函数，多个模块复用
- 线段采样21点绘制，兼顾精度和性能
- 球网底部投影是模块1对模块2的关键接口

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 渲染覆盖 | 在真实帧上渲染所有线段并可视化 | 线条与实际球场线对齐 |
| 批量投影精度 | 已知H，投影所有关键点与真值比较 | 误差 < 1px |
| 球网底部投影 | 投影结果与手工标注比较 | 误差 < 5px |
