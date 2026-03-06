"""
全流程入口：模块1（球场检测与渲染）+ 模块2（球网检测与相机标定）。

串联 7 个步骤：
    Step 1: YOLO 检测 (1.1) → 26 关键点
    Step 2: Homography 计算 (1.2)
    Step 3: H 验证门控
    Step 4: 球场线渲染 (1.3)
    Step 5: 球网关键点提取 (2.2)
    Step 6: 相机标定 (2.5)
    Step 7: 投影验证 (2.6)

用法：
    python -m scripts.run_pipeline --input frame.jpg --output result.jpg
    python -m scripts.run_pipeline --input video.mp4 --output output.mp4
    python -m scripts.run_pipeline --input frame.jpg --output result.jpg --show-keypoints --show-net
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from config.court_config import COURT_KEYPOINTS_2D, COURT_LINE_SEGMENTS
from module1.court_renderer import draw_court_overlay, draw_keypoint_markers
from module1.homography import HomographyResult, compute_homography, validate_homography
from module1.yolo_detector import CourtDetectionResult, YoloPoseDetector
from module2.camera_calibration import (
    CameraCalibrationResult,
    build_3d_2d_correspondences,
    calibrate_frame,
)
from module2.net_top_detector_yolo import NetKeypointResult, extract_net_keypoints
from module2.projection_validation import validate_projection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── 数据结构 ────────────────────────────────────────────────


class FrameResult:
    """单帧处理结果，记录每个步骤的产出和状态。"""

    __slots__ = (
        "detection", "h_result", "h_metrics",
        "net_kpts", "calibration", "proj_metrics",
        "failed_step",
    )

    def __init__(self):
        self.detection: CourtDetectionResult | None = None
        self.h_result: HomographyResult | None = None
        self.h_metrics: dict | None = None
        self.net_kpts: NetKeypointResult | None = None
        self.calibration: CameraCalibrationResult | None = None
        self.proj_metrics: dict | None = None
        self.failed_step: str | None = None

    @property
    def h_ok(self) -> bool:
        return self.h_metrics is not None and self.h_metrics["overall_ok"]

    @property
    def proj_ok(self) -> bool:
        return self.proj_metrics is not None and self.proj_metrics["overall_ok"]

    @property
    def pipeline_complete(self) -> bool:
        return self.proj_metrics is not None


# ── 核心流水线 ──────────────────────────────────────────────


def process_frame(
    frame: np.ndarray,
    detector: YoloPoseDetector,
    min_conf: float,
) -> FrameResult:
    """执行单帧 7 步流水线。

    Args:
        frame: BGR 图像。
        detector: YOLO 检测器。
        min_conf: 关键点最低置信度。

    Returns:
        FrameResult 包含各步骤的产出。
    """
    result = FrameResult()
    h_img, w_img = frame.shape[:2]

    # ── Step 1: YOLO 检测 (1.1) ──
    detection = detector.detect(frame)
    if detection is None:
        result.failed_step = "Step1:YOLO检测"
        return result
    result.detection = detection

    # ── Step 2: Homography 计算 (1.2) ──
    h_result = compute_homography(detection, COURT_KEYPOINTS_2D, min_conf)
    if h_result is None:
        result.failed_step = "Step2:Homography"
        return result
    result.h_result = h_result

    # ── Step 3: H 验证门控 ──
    h_metrics = validate_homography(h_result)
    result.h_metrics = h_metrics
    if not h_metrics["overall_ok"]:
        result.failed_step = "Step3:H验证"
        return result

    # ── Step 4: 球场线渲染 (1.3) ──
    # 渲染在 render_frame 中统一执行，此处只做数据处理

    # ── Step 5: 球网关键点提取 (2.2) ──
    net_kpts = extract_net_keypoints(detection, min_conf)
    if net_kpts is None:
        result.failed_step = "Step5:球网关键点"
        return result
    result.net_kpts = net_kpts

    # ── Step 6: 相机标定 (2.5) ──
    calibration = calibrate_frame(
        detection, h_result, net_kpts, (w_img, h_img)
    )
    if calibration is None:
        result.failed_step = "Step6:相机标定"
        return result
    result.calibration = calibration

    # ── Step 7: 投影验证 (2.6) ──
    obj_pts, img_pts = build_3d_2d_correspondences(
        detection, net_kpts, min_conf
    )
    proj_metrics = validate_projection(
        calibration, obj_pts, img_pts, h_result.H, (w_img, h_img)
    )
    result.proj_metrics = proj_metrics

    return result


# ── 渲染 ──────────────────────────────────────────────────


def render_frame(
    frame: np.ndarray,
    result: FrameResult,
    color: tuple[int, int, int],
    show_keypoints: bool,
    show_net: bool,
) -> np.ndarray:
    """根据流水线结果渲染输出帧。"""
    output = frame

    # 球场线叠加 (Step 4)
    if result.h_result is not None:
        output = draw_court_overlay(
            output, result.h_result.H,
            COURT_KEYPOINTS_2D, COURT_LINE_SEGMENTS, color,
        )

    # 关键点标注
    if show_keypoints and result.detection is not None:
        output = draw_keypoint_markers(output, result.detection, 0.5)

    # 球网关键点标注
    if show_net and result.net_kpts is not None:
        output = _draw_net_overlay(output, result.net_kpts)

    return output


def _draw_net_overlay(
    frame: np.ndarray,
    net_kpts: NetKeypointResult,
) -> np.ndarray:
    """在帧上渲染球网顶线和端点。"""
    overlay = frame.copy()
    top_color = (0, 0, 255)
    base_color = (255, 0, 0)
    line_color = (0, 255, 255)

    ep1 = net_kpts.left_top_pixel.astype(int)
    ep2 = net_kpts.right_top_pixel.astype(int)
    cv2.line(overlay, tuple(ep1), tuple(ep2), line_color, 2, cv2.LINE_AA)

    for px in [net_kpts.left_top_pixel, net_kpts.right_top_pixel]:
        cv2.circle(overlay, tuple(px.astype(int)), 6, top_color, -1, cv2.LINE_AA)

    if net_kpts.left_base_pixel is not None:
        for px in [net_kpts.left_base_pixel, net_kpts.right_base_pixel]:
            cv2.circle(overlay, tuple(px.astype(int)), 6, base_color, -1, cv2.LINE_AA)

    return overlay


# ── 日志 ──────────────────────────────────────────────────


def log_frame_result(result: FrameResult, prefix: str = ""):
    """输出单帧处理结果。"""
    parts = [prefix] if prefix else []

    if result.failed_step:
        parts.append(f"中止于 {result.failed_step}")
        if result.detection:
            parts.append(f"可见关键点={result.detection.num_visible}")
        if result.h_metrics and not result.h_metrics["overall_ok"]:
            failed_checks = [
                k for k in ["reproj_ok", "inlier_ok", "det_ok", "cond_ok", "sv_ok"]
                if not result.h_metrics.get(k, True)
            ]
            parts.append(f"H失败项={failed_checks}")
        logger.info(" | ".join(parts))
        return

    # Homography
    if result.h_metrics:
        parts.append(f"H误差={result.h_metrics['reprojection_error']:.2f}px")

    # 标定
    if result.calibration:
        cal = result.calibration
        parts.append(f"标定={cal.method}")
        parts.append(f"标定误差={cal.reprojection_error:.2f}px")

    # 投影验证
    if result.proj_metrics:
        pm = result.proj_metrics
        ok_str = "通过" if pm["overall_ok"] else "未通过"
        parts.append(f"投影验证={ok_str}")
        if not pm["overall_ok"]:
            failed_checks = [
                k for k in ["reproj_ok", "cam_ok", "orth_ok",
                             "intrinsics_ok", "center_ok", "hp_ok"]
                if not pm.get(k, True)
            ]
            parts.append(f"失败项={failed_checks}")
        parts.append(f"相机高度={pm['cam_height']:.2f}m")
        parts.append(f"H-P={pm['hp_consistency']:.4f}")

    logger.info(" | ".join(parts))


# ── 图片模式 ──────────────────────────────────────────────


def process_image(args, detector, color):
    """处理单张图片。"""
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = cv2.imread(str(input_path))
    if frame is None:
        logger.error("无法读取图片: %s", input_path)
        sys.exit(1)

    logger.info("处理图片: %s (%dx%d)", input_path.name,
                frame.shape[1], frame.shape[0])

    result = process_frame(frame, detector, args.min_conf)
    output = render_frame(frame, result, color, args.show_keypoints, args.show_net)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    logger.info("结果已保存: %s", output_path)

    log_frame_result(result)


# ── 视频模式 ──────────────────────────────────────────────


def process_video(args, detector, color):
    """处理视频。"""
    input_path = Path(args.input)
    output_path = Path(args.output)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error("无法打开视频: %s", input_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("视频: %dx%d @ %.1f fps, 共 %d 帧", width, height, fps, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 统计
    frame_idx = 0
    stats = {
        "total": 0,
        "h_ok": 0,
        "proj_complete": 0,
        "proj_ok": 0,
        "fail_steps": {},
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame, detector, args.min_conf)
        output = render_frame(frame, result, color, args.show_keypoints, args.show_net)
        writer.write(output)

        # 统计
        stats["total"] += 1
        if result.h_ok:
            stats["h_ok"] += 1
        if result.pipeline_complete:
            stats["proj_complete"] += 1
        if result.proj_ok:
            stats["proj_ok"] += 1
        if result.failed_step:
            step = result.failed_step
            stats["fail_steps"][step] = stats["fail_steps"].get(step, 0) + 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            _log_video_progress(frame_idx, total, stats)

    cap.release()
    writer.release()

    logger.info("输出: %s", output_path)
    _log_video_summary(stats)


def _log_video_progress(frame_idx: int, total: int, stats: dict):
    """视频处理进度日志。"""
    n = stats["total"]
    logger.info(
        "进度: %d/%d 帧 (%.1f%%) | H通过: %d (%.1f%%) | "
        "标定完成: %d (%.1f%%) | 投影通过: %d (%.1f%%)",
        frame_idx, total, 100.0 * frame_idx / max(total, 1),
        stats["h_ok"], 100.0 * stats["h_ok"] / max(n, 1),
        stats["proj_complete"], 100.0 * stats["proj_complete"] / max(n, 1),
        stats["proj_ok"], 100.0 * stats["proj_ok"] / max(n, 1),
    )


def _log_video_summary(stats: dict):
    """视频处理结束汇总。"""
    n = stats["total"]
    logger.info("=" * 60)
    logger.info("视频处理完成")
    logger.info("  总帧数: %d", n)
    logger.info("  H验证通过: %d (%.1f%%)",
                stats["h_ok"], 100.0 * stats["h_ok"] / max(n, 1))
    logger.info("  标定完成: %d (%.1f%%)",
                stats["proj_complete"], 100.0 * stats["proj_complete"] / max(n, 1))
    logger.info("  投影验证通过: %d (%.1f%%)",
                stats["proj_ok"], 100.0 * stats["proj_ok"] / max(n, 1))
    if stats["fail_steps"]:
        logger.info("  中止分布:")
        for step, count in sorted(stats["fail_steps"].items(),
                                   key=lambda x: -x[1]):
            logger.info("    %s: %d (%.1f%%)", step, count,
                        100.0 * count / max(n, 1))
    logger.info("=" * 60)


# ── 入口 ──────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="全流程：球场检测 → Homography → 球网检测 → 相机标定 → 投影验证"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入视频或图片路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出视频或图片路径")
    parser.add_argument("--court-model", type=str, default=None,
                        help="球场模型权重路径")
    parser.add_argument("--net-model", type=str, default=None,
                        help="球网模型权重路径")
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="关键点最低置信度 (默认: 0.5)")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (e.g. cpu, mps, 0)")
    parser.add_argument("--color", type=str, default="0,255,0",
                        help="球场渲染颜色 BGR, 逗号分隔 (默认: 0,255,0)")
    parser.add_argument("--show-keypoints", action="store_true",
                        help="在画面上标注关键点")
    parser.add_argument("--show-net", action="store_true",
                        help="在画面上渲染球网关键点")
    return parser.parse_args()


def main():
    args = parse_args()
    color = tuple(int(c) for c in args.color.split(","))

    logger.info("加载 YOLO 模型...")
    detector = YoloPoseDetector(
        court_model_path=args.court_model,
        net_model_path=args.net_model,
        confidence_threshold=args.min_conf,
        device=args.device,
    )
    logger.info("模型加载完成")

    input_path = Path(args.input)
    is_image = input_path.suffix.lower() in IMAGE_EXTENSIONS

    if is_image:
        process_image(args, detector, color)
    else:
        process_video(args, detector, color)


if __name__ == "__main__":
    main()
