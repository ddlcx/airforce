# 子模块2.3：双路融合与交叉验证

## 功能概述

利用 Hough 和 YOLO 两路独立检测结果的冗余性，通过交叉验证提高球网顶部线定位精度。根据两路结果的一致性程度（角度差、位置差），采取不同策略：一致时加权融合、轻微不一致时取更优者、严重不一致时诊断选择。

## 依赖关系

- **上游**：Hough 球网检测(2.1) → `NetTopLineResult`；YOLO 球网检测(2.2) → `NetTopLineResult`；Homography(1.2) → H
- **下游**：球网端点推断(2.4)

## 对应代码文件

`module2/net_top_fusion.py`

---

## 输入/输出数据结构

**输入**：Hough检测结果、YOLO检测结果（均可为None）、H矩阵、帧图像

**输出**：
```
FusionResult:
    fused_line: NetTopLineResult      # 融合后的球网顶线
    consistency_level: str            # "consistent" / "minor_inconsistent" / "conflict"
    angle_diff_deg: float             # 两路角度差（度）
    position_diff_px: float           # 两路位置差（像素）
    hough_result: NetTopLineResult    # Hough 原始结果（可能为None）
    yolo_result: NetTopLineResult     # YOLO 原始结果（可能为None）
    method_used: str                  # "weighted_fusion" / "primary_yolo" / "primary_hough" / "single_source"
```

## 算法详解

```
function fuse_net_top_detections(hough_result, yolo_result, H, frame):
    # ─── Step 1: 可用性判断 ───
    if hough_result is None and yolo_result is None:
        return None

    if hough_result is None:
        return FusionResult(yolo_result, "single_source", ..., method="single_source")

    if yolo_result is None:
        return FusionResult(hough_result, "single_source", ..., method="single_source")

    # ─── Step 2: 一致性度量 ───

    # (a) 角度差
    angle_hough = atan2(hough端点差y, hough端点差x)
    angle_yolo = atan2(yolo端点差y, yolo端点差x)
    Δθ = abs(angular_difference(angle_hough, angle_yolo))   # 度

    # (b) 位置差：在图像水平中点处比较两条线的y值
    x_mid = (frame.width) / 2
    y_hough = line_y_at_x(hough_result.line_coeffs, x_mid)
    y_yolo = line_y_at_x(yolo_result.line_coeffs, x_mid)
    Δd = abs(y_hough - y_yolo)    # 像素

    # ─── Step 3: 基于差异的决策 ───

    if Δθ < 3.0 and Δd < 8.0:
        # 一致：加权平均融合
        w_yolo = yolo_result.confidence / (yolo_result.confidence + hough_result.confidence)
        w_hough = 1.0 - w_yolo

        fused_coeffs = w_yolo * yolo_coeffs + w_hough * hough_coeffs
        fused_coeffs = normalize(fused_coeffs)
        fused_ep1 = w_yolo * yolo.ep1 + w_hough * hough.ep1
        fused_ep2 = w_yolo * yolo.ep2 + w_hough * hough.ep2

        fused_line = NetTopLineResult(fused_ep1, fused_ep2, fused_coeffs,
                                       max(两路confidence), ..., "fusion")
        return FusionResult(fused_line, "consistent", Δθ, Δd, ..., "weighted_fusion")

    elif Δθ < 8.0 and Δd < 20.0:
        # 轻微不一致：取置信度更高者
        primary = yolo_result if yolo_result.confidence > hough_result.confidence
                  else hough_result
        height_ok = validate_net_height(primary, H)

        return FusionResult(primary, "minor_inconsistent", Δθ, Δd, ...,
                            "primary_" + primary.source)

    else:
        # 严重不一致：诊断模式
        endpoints_hough = infer_net_endpoints(H, hough_result.line_coeffs)
        endpoints_yolo = infer_net_endpoints(H, yolo_result.line_coeffs)
        error_hough = compute_pnp_reprojection_error(endpoints_hough, ...)
        error_yolo = compute_pnp_reprojection_error(endpoints_yolo, ...)

        selected = hough_result if error_hough < error_yolo else yolo_result
        log.warning(f"球网检测严重不一致: Δθ={Δθ:.1f}°, Δd={Δd:.1f}px. "
                    f"选择 {selected.source} (重投影误差更小)")

        return FusionResult(selected, "conflict", Δθ, Δd, ..., "conflict_resolved")

    # ─── Step 4: 球网高度合理性验证（所有非None结果）───
    validate_net_height(fusion_result.fused_line, H)
```

### 球网高度合理性验证

```
function validate_net_height(net_line, H):
    pts = np.array([[-3.05, 0], [+3.05, 0], [-3.05, -6.70], [+3.05, -6.70]])
    pixels = project_points_batch(H, pts)
    base_left_px, base_right_px = pixels[0], pixels[1]
    left_corner, right_corner = pixels[2], pixels[3]

    d_left = point_to_line_distance(base_left_px, net_line.line_coeffs)
    d_right = point_to_line_distance(base_right_px, net_line.line_coeffs)

    pixel_per_meter_approx = np.linalg.norm(right_corner - left_corner) / 6.10
    h_estimated_left = d_left / pixel_per_meter_approx
    h_estimated_right = d_right / pixel_per_meter_approx

    if abs(h_estimated_left - 1.55) > 0.5 or abs(h_estimated_right - 1.55) > 0.5:
        log.warning("球网高度推断异常")
        return False
    return True
```

> 注意：此验证为粗略估计，因为像素/米比例在不同位置不同（透视效应），仅作为 sanity check。

## 设计要点

- 一致性判定采用双门控（角度差 + 位置差），避免单一指标误判
- 严重不一致时通过下游 PnP 重投影误差来仲裁（end-to-end 验证）
- 球网高度验证提供额外的物理合理性约束

## 测试方案

| 测试项 | 方法 | 通过标准 |
|--------|------|----------|
| 两路一致 | 构造Δθ<3°, Δd<8px的两路结果 | 输出method="weighted_fusion" |
| 轻微不一致 | 构造Δθ=5°, Δd=15px | 输出method="primary_xxx" |
| 严重不一致 | 构造Δθ=20° | 输出method="conflict_resolved"，有warning |
| 单路 | 一路为None | 输出method="single_source" |
| 两路都None | 两路都为None | 输出None |
| 真实帧统计 | 在20+帧上运行 | ≥80%的帧为"consistent" |
