# 子模块2.4：球网端点推断

## 功能概述

确定球网两侧顶部端点在图像中的像素坐标，从而获得两组非共面的3D-2D对应点（Z=1.55m）。核心思路：通过 Homography 投影球网立柱底部到像素空间，构造垂直线与球网顶线求交点，再用 YOLO 关键点交叉验证。

## 依赖关系

- **上游**：Homography(1.2) → H；双路融合(2.3) → `FusionResult.fused_line`；YOLO检测(1.1) → 关键点22、23（可选交叉验证）；球场渲染(1.3) → `project_points_batch`
- **下游**：相机标定(2.5)
- **共享工具**：`utils/geometry.py` 中的 `line_intersection`

## 对应代码文件

`module2/net_endpoint_inference.py`

---

## 输入/输出数据结构

**输入**：
- `H`：Homography 矩阵
- `fused_net_line`：融合后的球网顶线（`NetTopLineResult`）
- `yolo_detection`：YOLO检测结果（可选，用于交叉验证）

**输出**：
```
NetEndpointResult:
    left_top_pixel: ndarray (2,)      # 左端顶部像素坐标
    right_top_pixel: ndarray (2,)     # 右端顶部像素坐标
    left_base_pixel: ndarray (2,)     # 左端底部像素坐标
    right_base_pixel: ndarray (2,)    # 右端底部像素坐标
    left_top_3d: ndarray (3,)         # (-3.05, 0, 1.55)
    right_top_3d: ndarray (3,)        # (+3.05, 0, 1.55)
    method: str                        # "intersection" / "yolo_direct" / "yolo_verified"
```

## 算法详解

```
function infer_net_endpoints(H, net_top_line_coeffs, yolo_detection=None):
    # ─── Step 1: 投影立柱底部（cv2.perspectiveTransform）───
    net_bases = np.array([[-3.05, 0.0], [+3.05, 0.0]])
    base_pixels = project_points_batch(H, net_bases)
    left_base_px, right_base_px = base_pixels[0], base_pixels[1]

    # ─── Step 2: 构造垂直线 ───
    # 假设球网立柱在图像中垂直: x = base_x
    left_vertical = [1.0, 0.0, -left_base_px[0]]     # x = left_base_px.x
    right_vertical = [1.0, 0.0, -right_base_px[0]]   # x = right_base_px.x

    # ─── Step 3: 求交点 ───
    left_top_px = line_intersection(net_top_line_coeffs, left_vertical)
    right_top_px = line_intersection(net_top_line_coeffs, right_vertical)

    inferred_method = "intersection"

    # ─── Step 4: YOLO 交叉验证（如果可用）───
    if yolo_detection is not None:
        kp22 = yolo_detection.keypoints[22]    # 网左顶
        kp23 = yolo_detection.keypoints[23]    # 网右顶

        if kp22.visible and kp23.visible:
            dist_left = ||kp22.pixel_xy - left_top_px||
            dist_right = ||kp23.pixel_xy - right_top_px||

            if dist_left < 15 and dist_right < 15:
                # 一致：用 YOLO 的值（通常更准确）
                left_top_px = kp22.pixel_xy
                right_top_px = kp23.pixel_xy
                inferred_method = "yolo_verified"
            else:
                # 不一致：发出警告，仍用 YOLO 的值
                log.warning(f"端点推断与YOLO不一致: L={dist_left:.1f}px, R={dist_right:.1f}px")
                left_top_px = kp22.pixel_xy
                right_top_px = kp23.pixel_xy
                inferred_method = "yolo_direct"

    return NetEndpointResult(
        left_top_px, right_top_px,
        left_base_px, right_base_px,
        [-3.05, 0, 1.55], [+3.05, 0, 1.55],
        inferred_method
    )
```

### `line_intersection` 辅助函数（`utils/geometry.py`）

```
function line_intersection(line1_coeffs, line2_coeffs):
    # 两条直线 a1x+b1y+c1=0 和 a2x+b2y+c2=0 的交点
    a1, b1, c1 = line1_coeffs
    a2, b2, c2 = line2_coeffs
    det = a1*b2 - a2*b1
    if abs(det) < 1e-10:
        return None    # 平行线
    x = (b1*c2 - b2*c1) / det
    y = (a2*c1 - a1*c2) / det
    return [x, y]
```

## 关于"垂直"假设的说明

**适用条件**：相机近似水平放置（roll angle ≈ 0），这在手机拍摄中通常成立。

**失效情况**：相机有明显倾斜（roll > 5°）时，球网立柱在图像中不再垂直。

**迭代改进方案**：
1. 用初始的"垂直假设"得到球网端点 → 求解初始相机参数
2. 从相机参数计算垂直方向（Z轴）的 vanishing point
3. 将"垂直线"替换为"过底部像素指向 vanishing point 的线"
4. 重新计算交点 → 重新求解相机参数
5. 迭代直至收敛（通常1-2次即可）

## 设计要点

- 垂直假设是合理的初始近似，可迭代改进
- YOLO 交叉验证提供独立的检验手段
- 输出的3D坐标是固定值（球网尺寸已知），像素坐标是推断值

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 合成垂直交点 | 已知H、已知球网线，计算交点与真值比较 | 误差 < 1px |
| 真实帧推断 | 与手工标注比较 | 误差 < 10px/端点 |
| YOLO交叉一致 | YOLO检测到22/23，与推断比较 | 差异 < 15px |
| 平行线处理 | 球网线恰好垂直 | 返回合理值或异常 |
