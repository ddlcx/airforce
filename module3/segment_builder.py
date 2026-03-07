"""
Segment builder: split detection sequence into flight segments by hit events.

Each segment represents one shuttle flight between consecutive hits.
"""

from typing import List

import numpy as np

from module3.result_types import (
    ShuttlecockDetection,
    HitEvent,
    RallySegment,
)


def build_segments(
    detections: List[ShuttlecockDetection],
    hit_events: List[HitEvent],
    fps: float,
    P: np.ndarray,
    K: np.ndarray,
    min_segment_length: int = 5,
) -> List[RallySegment]:
    """
    Split detection sequence into flight segments by hit events.

    Args:
        detections: all detections sorted by frame_idx
        hit_events: hit events sorted by frame_idx
        fps: video frame rate
        P: (3, 4) projection matrix
        K: (3, 3) intrinsic matrix
        min_segment_length: minimum frames for a valid segment

    Returns:
        list of RallySegment
    """
    if len(hit_events) < 2:
        return []

    # Build frame_idx -> detection lookup
    det_by_frame = {d.frame_idx: d for d in detections}

    segments = []
    for i in range(len(hit_events) - 1):
        h_start = hit_events[i]
        h_end = hit_events[i + 1]

        start_frame = h_start.frame_idx
        end_frame = h_end.frame_idx

        if end_frame - start_frame < min_segment_length:
            continue

        all_frames = list(range(start_frame, end_frame + 1))
        seg_detections = []

        for f in all_frames:
            if f in det_by_frame:
                seg_detections.append(det_by_frame[f])
            else:
                # Missing detection placeholder
                seg_detections.append(ShuttlecockDetection(
                    frame_idx=f,
                    pixel_x=0.0,
                    pixel_y=0.0,
                    visible=False,
                    confidence=0.0,
                ))

        segments.append(RallySegment(
            detections=seg_detections,
            all_frame_indices=all_frames,
            hit_start=h_start,
            hit_end=h_end,
            fps=fps,
            P=P.copy(),
            K=K.copy(),
        ))

    return segments
