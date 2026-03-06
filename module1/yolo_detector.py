"""
YOLO Pose 推理封装。

封装球场（22关键点）和球网（4关键点）两个独立 YOLO Pose 模型的推理逻辑，
将数据集索引转换为全局 plan 索引，合并输出统一的 26 关键点检测结果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config.court_config import NUM_GROUND_KEYPOINTS, NUM_NET_KEYPOINTS, NUM_TOTAL_KEYPOINTS

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class KeypointDetection:
    """单个关键点的检测结果。"""

    index: int              # 0-25 全局索引 (plan 排序)
    pixel_xy: np.ndarray    # shape (2,)
    confidence: float       # 0.0-1.0
    visible: bool           # confidence >= threshold


@dataclass
class CourtDetectionResult:
    """合并的 26 关键点检测结果。"""

    keypoints: list[KeypointDetection]    # 长度 26, 按全局索引排列
    bbox: np.ndarray                       # shape (4,) [x1, y1, x2, y2]
    bbox_confidence: float
    num_visible: int
    smoothed: bool = False

    def get_visible_keypoints(self, min_conf: float = 0.5) -> list[KeypointDetection]:
        """返回置信度 >= min_conf 的关键点列表。"""
        return [kp for kp in self.keypoints if kp.confidence >= min_conf]

    def to_pixel_array(
        self, min_conf: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (indices, pixel_coords) 数组对。

        Returns:
            indices: shape (M,) int 数组，全局关键点索引。
            pixel_coords: shape (M, 2) float 数组，像素坐标。
        """
        visible = self.get_visible_keypoints(min_conf)
        if not visible:
            return np.array([], dtype=int), np.zeros((0, 2), dtype=np.float64)
        indices = np.array([kp.index for kp in visible], dtype=int)
        coords = np.array([kp.pixel_xy for kp in visible], dtype=np.float64)
        return indices, coords

    def get_ground_keypoints(
        self, min_conf: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回地面关键点 (索引 0-21) 的 (indices, pixel_coords)。"""
        indices, coords = self.to_pixel_array(min_conf)
        if len(indices) == 0:
            return indices, coords
        mask = indices < NUM_GROUND_KEYPOINTS
        return indices[mask], coords[mask]

    def get_net_keypoints(
        self, min_conf: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回球网关键点 (索引 22-25) 的 (indices, pixel_coords)。"""
        indices, coords = self.to_pixel_array(min_conf)
        if len(indices) == 0:
            return indices, coords
        mask = indices >= NUM_GROUND_KEYPOINTS
        return indices[mask], coords[mask]


# ── 检测器 ────────────────────────────────────────────────


class YoloPoseDetector:
    """封装球场和球网 YOLO Pose 模型的推理逻辑。"""

    def __init__(
        self,
        court_model_path: str | Path | None = None,
        net_model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ):
        """初始化检测器，加载两个 YOLO 模型。

        Args:
            court_model_path: 球场模型权重路径。None 则使用默认路径。
            net_model_path: 球网模型权重路径。None 则使用默认路径。
            confidence_threshold: 关键点可见性判定阈值。
            device: 推理设备 (e.g. 'cpu', 'mps', '0')。None 则自动检测。
        """
        from ultralytics import YOLO

        project_root = Path(__file__).resolve().parent.parent
        if court_model_path is None:
            court_model_path = project_root / "yolo" / "weights" / "court_yolo.pt"
        if net_model_path is None:
            net_model_path = project_root / "yolo" / "weights" / "net_yolo.pt"

        logger.info("加载球场模型: %s", court_model_path)
        self._court_model = YOLO(str(court_model_path))
        logger.info("加载球网模型: %s", net_model_path)
        self._net_model = YOLO(str(net_model_path))

        self._conf_threshold = confidence_threshold
        self._device = device

        # 从 training 模块导入索引映射
        from training.keypoint_mapping import (
            DATASET_TO_PLAN_INDEX as COURT_DS_TO_PLAN,
        )
        from training.keypoint_mapping_net import (
            DATASET_TO_PLAN_INDEX as NET_DS_TO_PLAN,
        )

        self._court_ds_to_plan = COURT_DS_TO_PLAN
        self._net_ds_to_plan = NET_DS_TO_PLAN

    def detect(self, frame: np.ndarray) -> CourtDetectionResult | None:
        """对单帧执行球场 + 球网检测。

        Args:
            frame: BGR 图像, shape (H, W, 3)。

        Returns:
            CourtDetectionResult 或 None（两个模型都未检测到时）。
        """
        court_kpts, court_bbox, court_conf = self._run_model(
            self._court_model,
            frame,
            num_keypoints=NUM_GROUND_KEYPOINTS,
            ds_to_plan=self._court_ds_to_plan,
        )

        net_kpts, net_bbox, net_conf = self._run_model(
            self._net_model,
            frame,
            num_keypoints=NUM_NET_KEYPOINTS,
            ds_to_plan=self._net_ds_to_plan,
        )

        if court_kpts is None and net_kpts is None:
            return None

        return self._merge_results(
            court_kpts, court_bbox, court_conf,
            net_kpts, net_bbox, net_conf,
        )

    def _run_model(
        self,
        model,
        frame: np.ndarray,
        num_keypoints: int,
        ds_to_plan: dict[int, int],
    ) -> tuple[dict | None, np.ndarray | None, float]:
        """运行单个 YOLO 模型并返回 plan 索引下的关键点字典。

        Returns:
            (kpts_dict, bbox, bbox_confidence)
            kpts_dict: {plan_index: (pixel_xy, confidence)} 或 None。
        """
        predict_kwargs = {"conf": 0.25, "verbose": False}
        if self._device is not None:
            predict_kwargs["device"] = self._device

        results = model(frame, **predict_kwargs)

        if results is None or len(results) == 0:
            return None, None, 0.0

        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return None, None, 0.0
        if len(result.boxes) == 0:
            return None, None, 0.0

        # 取最高置信度检测
        box_confs = result.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(box_confs))
        best_conf = float(box_confs[best_idx])
        best_bbox = result.boxes.xyxy[best_idx].cpu().numpy()  # shape (4,)

        kpts_data = result.keypoints.data[best_idx].cpu().numpy()  # shape (N, 3)

        # 转换数据集索引到 plan 索引
        kpts_dict = {}
        for ds_idx in range(num_keypoints):
            if ds_idx not in ds_to_plan:
                continue
            plan_idx = ds_to_plan[ds_idx]
            x, y, conf = kpts_data[ds_idx]
            kpts_dict[plan_idx] = (np.array([x, y], dtype=np.float64), float(conf))

        return kpts_dict, best_bbox, best_conf

    def _merge_results(
        self,
        court_kpts: dict | None,
        court_bbox: np.ndarray | None,
        court_conf: float,
        net_kpts: dict | None,
        net_bbox: np.ndarray | None,
        net_conf: float,
    ) -> CourtDetectionResult:
        """合并球场和球网检测结果为 26 关键点。"""
        # 球场 bbox 为主（覆盖整个球场区域）
        bbox = court_bbox if court_bbox is not None else net_bbox
        bbox_conf = court_conf if court_bbox is not None else net_conf

        keypoints = []
        num_visible = 0

        for idx in range(NUM_TOTAL_KEYPOINTS):
            source = None
            if court_kpts and idx in court_kpts:
                source = court_kpts[idx]
            elif net_kpts and idx in net_kpts:
                source = net_kpts[idx]

            if source is not None:
                pixel_xy, conf = source
                visible = conf >= self._conf_threshold
                if visible:
                    num_visible += 1
            else:
                pixel_xy = np.array([0.0, 0.0])
                conf = 0.0
                visible = False

            keypoints.append(KeypointDetection(
                index=idx,
                pixel_xy=pixel_xy,
                confidence=conf,
                visible=visible,
            ))

        return CourtDetectionResult(
            keypoints=keypoints,
            bbox=bbox if bbox is not None else np.zeros(4),
            bbox_confidence=bbox_conf,
            num_visible=num_visible,
        )
