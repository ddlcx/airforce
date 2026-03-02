# 误差传播管理

## 功能概述

定义各环节的误差预算、缓解策略和质量监控指标，确保误差不会在流水线中累积传播导致最终结果不可用。

## 依赖关系

本文档为跨模块的工程规范，所有子模块均需遵循。

## 对应代码文件

`utils/metrics.py`

---

## 误差预算

| 环节 | 预期误差 | 累积影响 |
|------|----------|----------|
| YOLO关键点检测 | 3-8px/点 | → H质量 |
| Homography | 2-5px 重投影 | → 球网底部投影精度 |
| 球网线检测(双路) | 2-5px, 1-3° | → 端点推断精度 |
| 球网端点推断 | 5-15px/端点 | → PnP（2/N权重低） |
| DLT/PnP | 2-8px 总重投影 | 最终输出 |
| 时间域平滑（视频模式） | 将 3-8px 抖动降至 <1px（稳态） | 显著提升叠加渲染稳定性 |

## 缓解策略

1. **RANSAC 逐级拒绝**：Homography 和 PnP 都使用 RANSAC，自动拒绝外点
2. **验证门控**：H 验证不通过 → 不进入模块2；P 验证不通过 → 不输出结果
3. **冗余设计**：22个地面点中最少4个即可算 H；双路球网检测互为备份
4. **LM精化**：PnP 后进行 Levenberg-Marquardt 优化，降低最终重投影误差
5. **交叉验证**：双路球网检测交叉验证；YOLO端点与推断端点交叉验证；H-P一致性检查
6. **两级标定**：DLT获得粗略内参 → PnP精化，避免内参初始假设偏差过大
7. **时间域平滑**：EMA 平滑关键点坐标，消除逐帧随机抖动；连续成功后锁定标定结果，消除渲染闪烁

## 质量监控指标

```
function compute_reprojection_stats(rvec, tvec, K, dist_coeffs, pts_3d, pts_2d):
    projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - pts_2d, axis=1)
    return {
        'mean': mean(errors),
        'median': median(errors),
        'max': max(errors),
        'std': std(errors),
        'rms': sqrt(mean(errors^2)),
        'p90': percentile(errors, 90),
        'p95': percentile(errors, 95),
    }

function compute_hp_consistency(H, P):
    H_from_P = P[:, [0, 1, 3]]
    H_norm = H / norm(H)
    H_P_norm = H_from_P / norm(H_from_P)
    return min(norm(H_norm - H_P_norm), norm(H_norm + H_P_norm))
```
