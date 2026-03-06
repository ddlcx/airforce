"""
模块2独立运行入口。

对视频或图片执行完整的球场检测 → 球网检测 → 相机标定 → 验证流程。

用法：
    python -m scripts.run_module2 --input video.mp4 --output output.mp4
    python -m scripts.run_module2 --input frame.jpg --output result.jpg
    python -m scripts.run_module2 --input frame.jpg --output result.jpg --show-net --show-keypoints
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
from module1.homography import compute_homography, validate_homography
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


def parse_args():
    parser = argparse.ArgumentParser(description="模块2：球网检测与相机标定")
    parser.add_argument("--input", type=str, required=True, help="输入视频或图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出视频或图片路径")
    parser.add_argument(
        "--court-model", type=str, default=None, help="球场模型权重路径"
    )
    parser.add_argument(
        "--net-model", type=str, default=None, help="球网模型权重路径"
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.5, help="关键点最低置信度 (默认: 0.5)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="推理设备 (e.g. cpu, mps, 0)"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="0,255,0",
        help="球场渲染颜色 BGR, 逗号分隔 (默认: 0,255,0)",
    )
    parser.add_argument(
        "--show-net", action="store_true", help="在画面上渲染球网关键点"
    )
    parser.add_argument(
        "--show-keypoints", action="store_true", help="在画面上标注关键点"
    )
    return parser.parse_args()


def process_frame(
    frame: np.ndarray,
    detector: YoloPoseDetector,
    min_conf: float,
    color: tuple[int, int, int],
    show_net: bool,
    show_keypoints: bool,
) -> tuple[np.ndarray, dict | None]:
    """处理单帧：检测 → Homography → 球网关键点 → 标定 → 验证。

    Returns:
        (output_frame, info_dict)
    """
    info = {}
    h, w = frame.shape[:2]

    # ── Module 1: 检测 + Homography ──
    detection = detector.detect(frame)
    if detection is None:
        return frame, None

    h_result = compute_homography(detection, COURT_KEYPOINTS_2D, min_conf)
    if h_result is None:
        output = frame
        if show_keypoints:
            output = draw_keypoint_markers(output, detection, min_conf)
        return output, None

    h_metrics = validate_homography(h_result)
    info["h_reproj"] = h_metrics["reprojection_error"]
    info["h_ok"] = h_metrics["overall_ok"]

    # ── Module 2: 球网关键点提取 ──
    net_kpts = extract_net_keypoints(detection, min_conf)

    # ── Module 2.5: 相机标定 ──
    calibration = None
    if net_kpts is not None:
        calibration = calibrate_frame(
            detection, h_result, net_kpts, (w, h)
        )
        if calibration is not None:
            info["cal_method"] = calibration.method
            info["cal_k_method"] = calibration.k_estimation_method
            info["cal_reproj"] = calibration.reprojection_error

    # ── Module 2.6: 投影验证 ──
    if calibration is not None and net_kpts is not None:
        obj_pts, img_pts = build_3d_2d_correspondences(detection, net_kpts, min_conf)
        proj_metrics = validate_projection(
            calibration, obj_pts, img_pts, h_result.H, (w, h)
        )
        info["proj_ok"] = proj_metrics["overall_ok"]
        info["proj_reproj"] = proj_metrics["mean_reproj"]
        info["cam_height"] = proj_metrics["cam_height"]
        info["cam_dist"] = proj_metrics["cam_dist"]
        info["hp_consistency"] = proj_metrics["hp_consistency"]

    # ── 渲染 ──
    output = draw_court_overlay(
        frame, h_result.H, COURT_KEYPOINTS_2D, COURT_LINE_SEGMENTS, color
    )

    if show_net and net_kpts is not None:
        output = _draw_net_keypoints(output, net_kpts)

    if show_keypoints:
        output = draw_keypoint_markers(output, detection, min_conf)

    return output, info


def _draw_net_keypoints(
    frame: np.ndarray,
    net_kpts: NetKeypointResult,
    top_color: tuple[int, int, int] = (0, 0, 255),
    base_color: tuple[int, int, int] = (255, 0, 0),
    line_color: tuple[int, int, int] = (0, 255, 255),
    radius: int = 6,
) -> np.ndarray:
    """在帧上渲染球网关键点和顶线。"""
    overlay = frame.copy()

    # 画球网顶线
    ep1 = net_kpts.left_top_pixel.astype(int)
    ep2 = net_kpts.right_top_pixel.astype(int)
    cv2.line(overlay, tuple(ep1), tuple(ep2), line_color, 2, cv2.LINE_AA)

    # 画顶部端点
    for px in [net_kpts.left_top_pixel, net_kpts.right_top_pixel]:
        center = tuple(px.astype(int))
        cv2.circle(overlay, center, radius, top_color, -1, cv2.LINE_AA)

    # 画底部端点
    if net_kpts.left_base_pixel is not None:
        for px in [net_kpts.left_base_pixel, net_kpts.right_base_pixel]:
            center = tuple(px.astype(int))
            cv2.circle(overlay, center, radius, base_color, -1, cv2.LINE_AA)

    return overlay


def process_image(args, detector, color):
    """处理单张图片。"""
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = cv2.imread(str(input_path))
    if frame is None:
        logger.error("无法读取图片: %s", input_path)
        sys.exit(1)

    output, info = process_frame(
        frame, detector, args.min_conf, color, args.show_net, args.show_keypoints
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    logger.info("结果已保存: %s", output_path)

    if info:
        _log_frame_info(info)


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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    success_count = 0

    logger.info("视频: %dx%d @ %.1f fps, 共 %d 帧", width, height, fps, total)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, info = process_frame(
            frame, detector, args.min_conf, color, args.show_net, args.show_keypoints
        )
        writer.write(output)

        if info is not None and info.get("proj_ok"):
            success_count += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(
                "进度: %d/%d 帧 (%.1f%%), 标定成功率 %.1f%%",
                frame_idx,
                total,
                100.0 * frame_idx / max(total, 1),
                100.0 * success_count / frame_idx,
            )

    cap.release()
    writer.release()

    logger.info(
        "完成: %d 帧, 标定成功 %d 帧 (%.1f%%)",
        frame_idx,
        success_count,
        100.0 * success_count / max(frame_idx, 1),
    )
    logger.info("输出: %s", output_path)


def _log_frame_info(info: dict):
    """输出帧处理详细信息。"""
    parts = []
    if "h_reproj" in info:
        parts.append(f"H误差={info['h_reproj']:.2f}px")
    if info.get("cal_method"):
        parts.append(f"标定={info['cal_method']}")
    if "cal_reproj" in info:
        parts.append(f"标定误差={info['cal_reproj']:.2f}px")
    if "proj_ok" in info:
        parts.append(f"验证={'通过' if info['proj_ok'] else '未通过'}")
    if "cam_height" in info:
        parts.append(f"相机高度={info['cam_height']:.2f}m")
    if "hp_consistency" in info:
        parts.append(f"H-P一致性={info['hp_consistency']:.4f}")

    logger.info("帧结果: %s", ", ".join(parts))


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

    input_path = Path(args.input)
    is_image = input_path.suffix.lower() in IMAGE_EXTENSIONS

    if is_image:
        process_image(args, detector, color)
    else:
        process_video(args, detector, color)


if __name__ == "__main__":
    main()
