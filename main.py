from model.LTS import *
from DataPrepare import DataPrepare
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.ops as ops
import json

from score import *

def get_candidate_boxes(onehot_heatmap, prob_heatmap, min_area=50):
    onehot_heatmap_np = onehot_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    prob_heatmap_np = prob_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    contours, _ = cv2.findContours(onehot_heatmap_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores = [], []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w * h < min_area: continue

        score = float(prob_heatmap_np[y:y+h, x:x+w].mean())

        x1, y1 = x, y
        x2, y2 = x + w, y + h

        boxes.append([x1, y1, x2, y2])
        scores.append(score)

    if not boxes:
        return torch.empty((0, 4)), torch.empty((0,))

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    return boxes, scores


def compute_heatmap(feature_map, query_embs):
    fm = feature_map.to(device=device, dtype=torch.float32)
    qe = query_embs.to(device=device, dtype=torch.float32)

    # * L2 norm to compute cosine simlarity
    feature_map_norm = F.normalize(fm, p=2, dim=1)
    query_embs_norm = F.normalize(qe, p=2, dim=1)

    # * Compute similarity between image point vector and query vector 
    similarity_map = torch.einsum('bchw,nc->bnhw', feature_map_norm, query_embs_norm)

    # * Max pooling theo query
    heatmap, _ = similarity_map.max(dim=1, keepdim=True)

    return heatmap


def process_video(video_id: str, video_path: str, frame_encoder, query_embs: torch.Tensor, threshold=0.5):
    cap = cv2.VideoCapture(video_path)

    # * video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f"Video {video_id} properties:")
    print(f"fps = {fps}, frame size ({width},{height})")

    results = []
    frame_idx = 0

    # * Process bar 
    process_bar = tqdm(total=n_frames)

    # * Iterate video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: break

        # * Extract feature map from frame
        P3, P4, P5 = frame_encoder.extract_feature_map(frame)
        # P3, P4, P5 = P3.clone(), P4.clone(), P5.clone()
        feature_map = frame_encoder.fusion_feature_map(P3, P4, P5)

        # * Compute similarity heatmap
        heatmap = compute_heatmap(feature_map, query_embs) # (1, 1, H_fm, W_fm)

        # * Scale up to frame size
        frame_heatmap = F.interpolate(heatmap, size=(int(height), int(width)), mode="bilinear", align_corners=False)

        # * Sigmoid to have probabily of each pixel
        prob_frame_heatmap = torch.sigmoid(frame_heatmap)
        onehot_frame_heatmap = (prob_frame_heatmap > threshold)

        # * Get bbox
        boxes, scores = get_candidate_boxes(onehot_frame_heatmap, prob_frame_heatmap)
        
        # * Run NMS to filter overlaped bbox
        if boxes.numel() != 0:
            keep_idx = ops.nms(boxes, scores, 0.5)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

        # * Get best box
        if boxes.shape[0]:
            best_idx = scores.argmax()
            x1, y1, x2, y2 = boxes[best_idx]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            results.append({
                "frame": frame_idx, 
                "x1": x1, 
                "y1": y1, 
                "x2": x2,
                "y2": y2
            })

        frame_idx += 1
        process_bar.update(1)

    process_bar.close()
    cap.release()

    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataPrepare(root_dir="observing")
    test_set = dataloader.prepare_train_data()

    image_encoder = CLIPEncoder(device=device)
    frame_encoder = YOLOv8mEncoder(device=device)

    test_results = []

    for sample in test_set:
        video_id = sample["video_id"]
        sample_annotation = dataloader.get_annotations_for_sample(video_id)
        query_images = sample["query_images"]
        query_video = sample["query_video"]

        # * Query embedding
        query_embs = []
        for img_path in query_images:
            query_embs.append(image_encoder.extract_features(image_path=img_path))
        query_embs = torch.cat(query_embs, dim=0)
        
        bboxes = process_video(video_id=video_id, video_path=query_video, frame_encoder=frame_encoder, query_embs=query_embs)

        sample_detection = {
            "video_id": video_id, 
            "detections": [
                {
                    "bboxes": bboxes
                }
            ]
        }

        with open(f'./test_prediction/{video_id}.json', 'w', encoding='utf-8') as f:
            json.dump(sample_detection, f, ensure_ascii=False, indent=4)

        test_results.append(sample_detection)

    with open('./test_prediction/test_prediction.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)




