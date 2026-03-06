"""
球网 YOLO 检测精度审阅脚本。

从球网数据集随机抽样图片，用 YOLO 球网模型检测 4 个关键点 (kp22-25)，
在图片上标注检测结果（关键点位置、置信度、连线），供人工审阅。

用法：
    python -m scripts.inspect_net_detection --num 20 --output output/net_inspect
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np

from module1.yolo_detector import YoloPoseDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 球网关键点定义
NET_KP_LABELS = {
    22: "kp22:LTop",
    23: "kp23:LBase",
    24: "kp24:RTop",
    25: "kp25:RBase",
}

# 颜色 (BGR)
COLOR_TOP = (0, 0, 255)       # 红 - 顶部关键点
COLOR_BASE = (255, 0, 0)      # 蓝 - 底部关键点
COLOR_TOP_LINE = (0, 255, 255)  # 黄 - 顶线
COLOR_BASE_LINE = (0, 200, 200) # 暗黄 - 底线
COLOR_VERTICAL = (128, 128, 128) # 灰 - 垂直连线


def parse_args():
    parser = argparse.ArgumentParser(description="球网 YOLO 检测精度审阅")
    parser.add_argument("--num", type=int, default=20, help="抽样数量")
    parser.add_argument("--output", type=str, default="output/net_inspect", help="输出目录")
    parser.add_argument("--net-model", type=str, default=None, help="球网模型路径")
    parser.add_argument("--device", type=str, default=None, help="推理设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--min-conf", type=float, default=0.0, help="显示的最低置信度 (默认: 0, 全部显示)")
    return parser.parse_args()


def draw_net_detection(frame: np.ndarray, detection, min_conf: float) -> np.ndarray:
    """在图片上标注球网关键点检测结果。"""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    kps = {}
    for idx in [22, 23, 24, 25]:
        kp = detection.keypoints[idx]
        kps[idx] = kp

    # 画连线（先画线再画点，这样点在最上层）
    # 顶线 kp22-kp24
    if kps[22].confidence >= min_conf and kps[24].confidence >= min_conf:
        p1 = tuple(kps[22].pixel_xy.astype(int))
        p2 = tuple(kps[24].pixel_xy.astype(int))
        cv2.line(overlay, p1, p2, COLOR_TOP_LINE, 2, cv2.LINE_AA)

    # 底线 kp23-kp25
    if kps[23].confidence >= min_conf and kps[25].confidence >= min_conf:
        p1 = tuple(kps[23].pixel_xy.astype(int))
        p2 = tuple(kps[25].pixel_xy.astype(int))
        cv2.line(overlay, p1, p2, COLOR_BASE_LINE, 2, cv2.LINE_AA)

    # 垂直连线 kp22-kp23, kp24-kp25
    for top_idx, base_idx in [(22, 23), (24, 25)]:
        if kps[top_idx].confidence >= min_conf and kps[base_idx].confidence >= min_conf:
            p1 = tuple(kps[top_idx].pixel_xy.astype(int))
            p2 = tuple(kps[base_idx].pixel_xy.astype(int))
            cv2.line(overlay, p1, p2, COLOR_VERTICAL, 1, cv2.LINE_AA)

    # 画关键点
    for idx in [22, 23, 24, 25]:
        kp = kps[idx]
        color = COLOR_TOP if idx in [22, 24] else COLOR_BASE
        center = tuple(kp.pixel_xy.astype(int))
        conf = kp.confidence

        # 关键点圆圈
        if conf >= min_conf:
            cv2.circle(overlay, center, 8, color, -1, cv2.LINE_AA)
            # 白色边框
            cv2.circle(overlay, center, 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # 低置信度用空心圆
            cv2.circle(overlay, center, 8, color, 1, cv2.LINE_AA)

        # 标签: kp名称 + 置信度
        label = f"{NET_KP_LABELS[idx]} ({conf:.2f})"
        # 确定标签位置（避免重叠）
        offset_x = 12
        offset_y = -10 if idx in [22, 24] else 20
        label_pos = (center[0] + offset_x, center[1] + offset_y)

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            overlay,
            (label_pos[0] - 2, label_pos[1] - th - 2),
            (label_pos[0] + tw + 2, label_pos[1] + 4),
            (0, 0, 0), -1,
        )
        cv2.putText(
            overlay, label, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # 汇总信息栏
    confs = [kps[i].confidence for i in [22, 23, 24, 25]]
    visible_count = sum(1 for c in confs if c >= 0.5)
    info = f"Visible: {visible_count}/4 | Conf: {confs[0]:.2f}, {confs[1]:.2f}, {confs[2]:.2f}, {confs[3]:.2f}"
    cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(
        overlay, info, (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return overlay


def main():
    args = parse_args()
    random.seed(args.seed)

    img_dir = Path("yolo/datasets/net/train/images")
    if not img_dir.exists():
        logger.error("数据集目录不存在: %s", img_dir)
        return

    all_images = sorted(img_dir.glob("*.jpg"))
    if not all_images:
        logger.error("未找到图片")
        return

    num = min(args.num, len(all_images))
    selected = random.sample(all_images, num)
    logger.info("从 %d 张图片中随机选取 %d 张", len(all_images), num)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    logger.info("加载 YOLO 球网模型...")
    detector = YoloPoseDetector(
        net_model_path=args.net_model,
        confidence_threshold=0.1,  # 低阈值以显示所有检测
        device=args.device,
    )

    for i, img_path in enumerate(selected, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("无法读取: %s", img_path.name)
            continue

        detection = detector.detect(frame)
        if detection is None:
            logger.warning("[%d/%d] 未检测到: %s", i, num, img_path.name)
            # 保存原图 + 未检测标记
            cv2.putText(
                frame, "NO DETECTION", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
            )
            out_path = output_dir / f"{i:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            continue

        result = draw_net_detection(frame, detection, args.min_conf)

        out_path = output_dir / f"{i:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 日志
        confs = [detection.keypoints[idx].confidence for idx in [22, 23, 24, 25]]
        logger.info(
            "[%d/%d] %s → conf=[%.2f, %.2f, %.2f, %.2f]",
            i, num, img_path.name, *confs,
        )

    logger.info("完成, 输出目录: %s", output_dir)


if __name__ == "__main__":
    main()
