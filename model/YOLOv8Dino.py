import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pathlib as Path
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=256):
        super().__init__()

        # * Projection head
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        embedding = self.encoder
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()

class YOLOv8Dino:
    def __init__(self, 
                yolo_model_path:str='yolov8n.pt', 
                dinov2_model_name:str='facebook/dinov2-base', 
                similarity_threshold: float = 0.75, 
                confidence_threshold: float = 0.25,
                device: str='cuda'):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

        self.yolo_model = YOLO(yolo_model_path)

        self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        self.dinov2_model = AutoModel.from_pretrained(dinov2_model_name).to(device)
        self.dinov2_model.eval()

        self.support_features = None

    def extract_features(self, image: Image.Image, is_normalized:bool=False) -> torch.Tensor:
        '''
            get features embedding from DINOv2
        '''
        inputs = self.dinov2_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

        if is_normalized:
            features = F.normalize(features, p=2, dim=1)

        return features

    def get_support_embeddings(self, support_images:List[str]) -> torch.Tensor:
        support_embddings = []

        for img_path in support_images:
            img = Image.open(img_path).convert('RGB')
            feats = self.extract_features(img)
            support_embddings.append(feats)

        return support_embddings

    def get_region_embeddings(self, yolo_results, frame) -> torch.Tensor:
        # * Get detected region
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        confidences = yolo_results.boxes.conf.cpu().numpy()

        detected_regions = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0: detected_regions.append(cropped)

        # * Extract features from detected region
        all_feats = []
        for region in detected_regions:
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(region_rgb)
            feats = self.extract_features(pil_img, is_normalized=False)
            all_feats.append(feats)

        all_feats = torch.cat(all_feats, dim=0)

        return all_feats
    
    def process_video(extract_model, simemese_model, video_path: str, support_images: List[str]):
        support_embeddings = extract_model.get_support_embeddings(support_images) # (3, 768)

        cap = cv2.VideoCapture(video_path)

        # * Video property
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        results = []
        frame_idx = 0

        process_bar = tqdm(total=n_frames)

        # * Iterate video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            yolo_results = extract_model.yolo_model(frame, conf=extract_model.confidence_threshold, verbose=False)[0]

            if len(yolo_results.boxes) == 0: continue
            
            region_embeddings = extract_model.get_region_embeddings(yolo_results, frame)

            # frame_res = {
            #     'frame_id': frame_idx,
            #     'detections': detections
            # }

            # results.append(frame_res)

            frame_idx +=1 
            process_bar.update(1)

        process_bar.close()
        cap.release()

        return results
