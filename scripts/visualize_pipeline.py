"""
模块1+2 流水线可视化脚本。

从训练数据集随机抽样图片，对每张图片生成包含 3 个子图的合成图：
  1. YOLO 场地关键点 (kp0-21) + 球场线叠加
  2. YOLO 球网关键点 (kp22-25)
  3. 球网关键点提取结果 (顶部/底部端点 + 连线)

用法：
    python -m scripts.visualize_pipeline --num 10 --output output/pipeline_vis
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from config.court_config import (
    COURT_KEYPOINTS_2D,
    COURT_LINE_SEGMENTS,
    NUM_GROUND_KEYPOINTS,
)
from module1.court_renderer import draw_court_overlay, project_points_batch
from module1.homography import compute_homography
from module1.yolo_detector import CourtDetectionResult, YoloPoseDetector
from module2.net_top_detector_yolo import extract_net_keypoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 颜色定义 (BGR) ──────────────────────────────────────────
COLOR_COURT_KP = (0, 255, 0)       # 绿 - 场地关键点
COLOR_NET_KP = (0, 0, 255)         # 红 - 球网关键点
COLOR_NET_TOP = (0, 0, 255)        # 红 - 球网顶部端点
COLOR_NET_BASE = (255, 0, 0)       # 蓝 - 球网底部端点
COLOR_NET_LINE = (0, 255, 255)     # 黄 - 球网顶线
COLOR_COURT_OVERLAY = (0, 200, 0)  # 暗绿 - 球场线条


def parse_args():
    parser = argparse.ArgumentParser(description="模块流水线可视化")
    parser.add_argument(
        "--num", type=int, default=10, help="抽样图片数量 (默认: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="output/pipeline_vis",
        help="输出目录 (默认: output/pipeline_vis)",
    )
    parser.add_argument(
        "--court-model", type=str, default=None, help="球场模型权重路径"
    )
    parser.add_argument(
        "--net-model", type=str, default=None, help="球网模型权重路径"
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.5, help="关键点最低置信度"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="推理设备"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--dataset", type=str, default="court",
        choices=["court", "net"],
        help="从哪个数据集采样 (默认: court)",
    )
    return parser.parse_args()


def _put_title(img: np.ndarray, title: str, bg_color=(0, 0, 0)):
    """在图片顶部加标题栏。"""
    h, w = img.shape[:2]
    bar_h = 32
    result = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    result[:bar_h] = bg_color
    result[bar_h:] = img
    cv2.putText(
        result, title, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def _draw_court_keypoints(
    frame: np.ndarray,
    detection: CourtDetectionResult,
    min_conf: float,
) -> np.ndarray:
    """子图1: 场地关键点 (kp0-21) + 球场线叠加。"""
    overlay = frame.copy()
    for kp in detection.keypoints:
        if kp.index >= NUM_GROUND_KEYPOINTS:
            continue
        if kp.confidence < min_conf:
            continue
        center = tuple(kp.pixel_xy.astype(int))
        cv2.circle(overlay, center, 5, COLOR_COURT_KP, -1, cv2.LINE_AA)
        cv2.putText(
            overlay, str(kp.index),
            (center[0] + 6, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return overlay


def _draw_net_keypoints(
    frame: np.ndarray,
    detection: CourtDetectionResult,
    min_conf: float,
) -> np.ndarray:
    """子图2: 球网关键点 (kp22-25)。"""
    overlay = frame.copy()

    net_labels = {22: "LTop", 23: "LBase", 24: "RTop", 25: "RBase"}
    net_colors = {22: COLOR_NET_TOP, 23: COLOR_NET_BASE, 24: COLOR_NET_TOP, 25: COLOR_NET_BASE}

    for idx in range(22, 26):
        kp = detection.keypoints[idx]
        if kp.confidence < min_conf:
            continue
        center = tuple(kp.pixel_xy.astype(int))
        cv2.circle(overlay, center, 6, net_colors[idx], -1, cv2.LINE_AA)
        label = f"{idx}:{net_labels[idx]} ({kp.confidence:.2f})"
        cv2.putText(
            overlay, label,
            (center[0] + 8, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # 顶线和底线连线
    kp22, kp24 = detection.keypoints[22], detection.keypoints[24]
    if kp22.confidence >= min_conf and kp24.confidence >= min_conf:
        cv2.line(overlay, tuple(kp22.pixel_xy.astype(int)),
                 tuple(kp24.pixel_xy.astype(int)), COLOR_NET_LINE, 2, cv2.LINE_AA)

    kp23, kp25 = detection.keypoints[23], detection.keypoints[25]
    if kp23.confidence >= min_conf and kp25.confidence >= min_conf:
        cv2.line(overlay, tuple(kp23.pixel_xy.astype(int)),
                 tuple(kp25.pixel_xy.astype(int)), (0, 200, 200), 2, cv2.LINE_AA)

    return overlay


def _draw_net_result(
    frame: np.ndarray,
    net_kpts,
    H: np.ndarray | None = None,
) -> np.ndarray:
    """子图3: 球网关键点提取结果 (顶部/底部端点 + 连线 + 地面参考线)。"""
    overlay = frame.copy()

    # 画球网地面线参考
    if H is not None:
        net_base = np.array([[-3.05, 0.0], [+3.05, 0.0]], dtype=np.float64)
        base_px = project_points_batch(H, net_base).astype(int)
        cv2.line(overlay, tuple(base_px[0]), tuple(base_px[1]), (0, 128, 0), 1, cv2.LINE_AA)

    if net_kpts is None:
        cv2.putText(
            overlay, "Net Keypoints: NOT DETECTED", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return overlay

    # 画球网顶线
    ep1 = net_kpts.left_top_pixel.astype(int)
    ep2 = net_kpts.right_top_pixel.astype(int)
    cv2.line(overlay, tuple(ep1), tuple(ep2), COLOR_NET_LINE, 2, cv2.LINE_AA)

    # 画顶部端点
    for px, label in [
        (net_kpts.left_top_pixel, "LT"),
        (net_kpts.right_top_pixel, "RT"),
    ]:
        center = tuple(px.astype(int))
        cv2.circle(overlay, center, 7, COLOR_NET_TOP, -1, cv2.LINE_AA)
        cv2.putText(
            overlay, label,
            (center[0] + 9, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # 画底部端点
    if net_kpts.left_base_pixel is not None:
        for px, label in [
            (net_kpts.left_base_pixel, "LB"),
            (net_kpts.right_base_pixel, "RB"),
        ]:
            center = tuple(px.astype(int))
            cv2.circle(overlay, center, 7, COLOR_NET_BASE, -1, cv2.LINE_AA)
            cv2.putText(
                overlay, label,
                (center[0] + 9, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # 画垂直连线 (顶到底)
        for top_px, base_px in [
            (net_kpts.left_top_pixel, net_kpts.left_base_pixel),
            (net_kpts.right_top_pixel, net_kpts.right_base_pixel),
        ]:
            p1 = tuple(top_px.astype(int))
            p2 = tuple(base_px.astype(int))
            cv2.line(overlay, p1, p2, (128, 128, 128), 1, cv2.LINE_AA)

    return overlay


def process_single_image(
    image_path: Path,
    detector: YoloPoseDetector,
    min_conf: float,
    output_dir: Path,
) -> bool:
    """处理单张图片，生成合成可视化。"""
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.warning("无法读取图片: %s", image_path)
        return False

    # ── YOLO 检测 ──
    detection = detector.detect(frame)
    if detection is None:
        logger.warning("YOLO 未检测到: %s", image_path.name)
        return False

    # ── Homography ──
    h_result = compute_homography(detection, COURT_KEYPOINTS_2D, min_conf)
    H = h_result.H if h_result is not None else None

    # ── 子图 1: 场地关键点 ──
    panel1 = _draw_court_keypoints(frame, detection, min_conf)
    if h_result is not None:
        panel1 = draw_court_overlay(
            panel1, H, COURT_KEYPOINTS_2D, COURT_LINE_SEGMENTS,
            COLOR_COURT_OVERLAY, thickness=1,
        )
    panel1 = _put_title(panel1, "1. Court Keypoints (kp0-21)")

    # ── 子图 2: 球网关键点 ──
    panel2 = _draw_net_keypoints(frame, detection, min_conf)
    panel2 = _put_title(panel2, "2. Net Keypoints (kp22-25)")

    # ── 子图 3: 球网关键点提取结果 ──
    net_kpts = extract_net_keypoints(detection, min_conf)
    panel3 = _draw_net_result(frame, net_kpts, H)
    panel3 = _put_title(panel3, "3. Net Keypoint Result")

    # ── 合成: 3 张横排 ──
    composite = np.hstack([panel1, panel2, panel3])

    # 保存
    out_name = image_path.stem + "_pipeline.jpg"
    out_path = output_dir / out_name
    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.info("已保存: %s", out_path)
    return True


def main():
    args = parse_args()
    random.seed(args.seed)

    # 查找数据集图片
    if args.dataset == "court":
        img_dir = Path("yolo/datasets/court/all/images")
    else:
        img_dir = Path("yolo/datasets/net/train/images")

    if not img_dir.exists():
        logger.error("数据集目录不存在: %s", img_dir)
        sys.exit(1)

    all_images = sorted(img_dir.glob("*.jpg"))
    if not all_images:
        logger.error("未找到图片: %s", img_dir)
        sys.exit(1)

    num = min(args.num, len(all_images))
    selected = random.sample(all_images, num)
    logger.info("从 %d 张图片中随机选取 %d 张", len(all_images), num)

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    logger.info("加载 YOLO 模型...")
    detector = YoloPoseDetector(
        court_model_path=args.court_model,
        net_model_path=args.net_model,
        confidence_threshold=args.min_conf,
        device=args.device,
    )

    success = 0
    for i, img_path in enumerate(selected, 1):
        logger.info("[%d/%d] 处理: %s", i, num, img_path.name)
        if process_single_image(img_path, detector, args.min_conf, output_dir):
            success += 1

    logger.info("完成: %d/%d 张成功, 输出目录: %s", success, num, output_dir)


if __name__ == "__main__":
    main()
