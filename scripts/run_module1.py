"""
模块1独立运行入口。

对视频或图片执行球场检测 → Homography 计算 → 球场线条渲染。

用法：
    python -m scripts.run_module1 --input video.mp4 --output output.mp4
    python -m scripts.run_module1 --input frame.jpg --output result.jpg
    python -m scripts.run_module1 --input frame.jpg --output result.jpg --show-keypoints
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="模块1：球场检测与渲染")
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
        help="渲染颜色 BGR, 逗号分隔 (默认: 0,255,0)",
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
    show_keypoints: bool,
) -> tuple[np.ndarray, CourtDetectionResult | None, dict | None]:
    """处理单帧：检测 → Homography → 渲染。

    Returns:
        (output_frame, detection, metrics)
        metrics 中额外包含 'num_inliers' 和 'num_correspondences'。
    """
    # Step 1: 检测
    detection = detector.detect(frame)
    if detection is None:
        return frame, None, None

    # Step 2: 计算 Homography
    h_result = compute_homography(detection, COURT_KEYPOINTS_2D, min_conf)
    if h_result is None:
        output = frame
        if show_keypoints:
            output = draw_keypoint_markers(
                output, detection, min_conf, ground_only=True
            )
        return output, detection, None

    # Step 3: 验证
    metrics = validate_homography(h_result)
    metrics["num_inliers"] = h_result.num_inliers
    metrics["num_correspondences"] = h_result.num_correspondences

    # Step 4: 渲染（即使验证未通过也渲染，方便调试观察）
    output = draw_court_overlay(
        frame, h_result.H, COURT_KEYPOINTS_2D, COURT_LINE_SEGMENTS, color
    )
    if show_keypoints:
        # 模块1只标注地面关键点 (0-21)，球网关键点留给模块2
        output = draw_keypoint_markers(
            output, detection, min_conf, ground_only=True
        )

    return output, detection, metrics


def process_image(args, detector, color):
    """处理单张图片。"""
    input_path = Path(args.input)
    output_path = Path(args.output)

    frame = cv2.imread(str(input_path))
    if frame is None:
        logger.error("无法读取图片: %s", input_path)
        sys.exit(1)

    output, detection, metrics = process_frame(
        frame, detector, args.min_conf, color, args.show_keypoints
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    logger.info("结果已保存: %s", output_path)

    if detection:
        logger.info("检测到 %d 个可见关键点", detection.num_visible)
    if metrics:
        logger.info(
            "Homography: %d 内点 / %d 对应, 重投影误差 %.2f px, 验证 %s",
            metrics["num_inliers"],
            metrics["num_correspondences"],
            metrics["reprojection_error"],
            "通过" if metrics["overall_ok"] else "未通过",
        )


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

        output, detection, metrics = process_frame(
            frame, detector, args.min_conf, color, args.show_keypoints
        )
        writer.write(output)

        if metrics is not None and metrics["overall_ok"]:
            success_count += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info(
                "进度: %d/%d 帧 (%.1f%%), Homography 成功率 %.1f%%",
                frame_idx,
                total,
                100.0 * frame_idx / max(total, 1),
                100.0 * success_count / frame_idx,
            )

    cap.release()
    writer.release()

    logger.info(
        "完成: %d 帧, Homography 成功 %d 帧 (%.1f%%)",
        frame_idx,
        success_count,
        100.0 * success_count / max(frame_idx, 1),
    )
    logger.info("输出: %s", output_path)


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
