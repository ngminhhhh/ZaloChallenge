from DataPrepare import *
import json

def bbox_iou(b1, b2):
    xA = max(b1["x1"], b2["x1"])
    yA = max(b1["y1"], b2["y1"])
    xB = min(b1["x2"], b2["x2"])
    yB = min(b1["y2"], b2["y2"])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    area2 = (b2["x2"] - b2["y1"]) * (b2["y2"] - b2["y1"])
    area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union

def compute_st_iou(gt_boxes, pred_boxes):
    gt_by_frame  = {b["frame"]: b for b in gt_boxes}
    pr_by_frame  = {b["frame"]: b for b in pred_boxes}

    gt_frames    = set(gt_by_frame.keys())
    pr_frames    = set(pr_by_frame.keys())

    intersection = gt_frames & pr_frames  
    union        = gt_frames | pr_frames   

    if not union:
        return 0.0   

    num = 0.0
    for f in intersection:
        num += bbox_iou(gt_by_frame[f], pr_by_frame[f])

    den = float(len(union))

    return num / den