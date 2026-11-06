"""
Vehicle Detection & Localization: Evaluation Metrics.

Includes:
- Intersection over Union (IoU) calculation for bounding box evaluation
- Batch IoU calculation
- Average Precision (AP / mAP) metric for Stanford Cars localization validation
"""

from typing import List, Sequence, Tuple, Union
import numpy as np

BoundingBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def compute_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    
    Bounding boxes are in (x1, y1, x2, y2) format:
      x1, y1: top-left coordinates
      x2, y2: bottom-right coordinates
      
    Args:
        box_a: First bounding box [x1, y1, x2, y2].
        box_b: Second bounding box [x1, y1, x2, y2].
        
    Returns:
        IoU score in range [0.0, 1.0].
    """
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("Bounding boxes must contain exactly 4 coordinates (x1, y1, x2, y2).")

    # Coordinates of intersection rectangle
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))

    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return float(np.clip(intersection_area / union_area, 0.0, 1.0))


def compute_batch_iou(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
) -> List[float]:
    """
    Computes pairwise IoU between corresponding predicted and ground-truth boxes.
    
    Args:
        pred_boxes: List of predicted bounding boxes [N, 4].
        gt_boxes: List of ground-truth bounding boxes [N, 4].
        
    Returns:
        List of float IoU values.
    """
    if len(pred_boxes) != len(gt_boxes):
        raise ValueError(f"Lengths mismatch: {len(pred_boxes)} pred boxes vs {len(gt_boxes)} gt boxes.")

    return [compute_iou(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]


def compute_map(
    pred_boxes_list: Sequence[Sequence[Sequence[float]]],
    gt_boxes_list: Sequence[Sequence[Sequence[float]]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculates Mean Average Precision (mAP) at a specified IoU threshold.
    
    Args:
        pred_boxes_list: Sequence of predictions per image, each pred being [x1, y1, x2, y2, score].
        gt_boxes_list: Sequence of ground-truth boxes per image [x1, y1, x2, y2].
        iou_threshold: Minimum IoU to count as a True Positive (default 0.50).
        
    Returns:
        mAP score in [0.0, 1.0].
    """
    all_scores = []
    all_tp = []
    total_gts = sum(len(gts) for gts in gt_boxes_list)

    if total_gts == 0:
        return 0.0

    for preds, gts in zip(pred_boxes_list, gt_boxes_list):
        matched_gt = set()
        # Sort predictions by confidence descending
        sorted_preds = sorted(preds, key=lambda x: x[4] if len(x) > 4 else 1.0, reverse=True)

        for pred in sorted_preds:
            p_box = pred[:4]
            score = pred[4] if len(pred) > 4 else 1.0
            all_scores.append(score)

            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt in enumerate(gts):
                iou = compute_iou(p_box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                all_tp.append(1)
                matched_gt.add(best_gt_idx)
            else:
                all_tp.append(0)

    if not all_scores:
        return 0.0

    # Sort all predictions across dataset by score descending
    indices = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[indices]
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-7)
    recalls = tp_cum / total_gts

    # 11-point interpolation or area under PR curve
    # Standard VOC-style AP interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = precisions[recalls >= t]
        p = np.max(prec_at_rec) if prec_at_rec.size > 0 else 0.0
        ap += p / 11.0

    return float(np.clip(ap, 0.0, 1.0))
