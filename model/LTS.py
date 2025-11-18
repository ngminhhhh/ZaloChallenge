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

class YOLOv8FeatureExtractor:
    def __init__(self, weight_path="yolov8n.pt", device="cuda", imgsz=640):
        self.yolo = YOLO(weight_path)
        self.device = device
        self.imgsz = imgsz

        self.detect_model = self.yolo.model.to(device)
        self.detect_layer = self.detect_model.model[-1]

        self.feature_maps = None

        def hook(module, inputs, output):
            feats = inputs[0]
            self.feature_maps = [f.detach().clone() for f in feats]

        self.hook_handle = self.detect_layer.register_forward_hook(hook)

    def extract_feature_map(self, frame_bgr):
        _ = self.yolo.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        if self.feature_maps is None:
            print("ERROR to extract feature map")

        return self.feature_maps
    
class DINOv2FeatureExtractor:
    def __init__(self, dinov2_model_name:str='facebook/dinov2-base', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        self.dinov2_model = AutoModel.from_pretrained(dinov2_model_name).to(device)
        self.dinov2_model.eval()


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
    
class YoloProj(nn.Module):
    def __init__(self, in_dim=144, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)
    
class DinoProj(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, s):
        return self.proj(s)
    
class MatchHead(nn.Module):
    def __init__(self, yolo_dim=144, dino_dim=768, latent_dim=256):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_head = YoloProj(yolo_dim, latent_dim).to(self.device)
        self.dino_head = DinoProj(dino_dim, latent_dim).to(self.device)

    def forward(self, P3, support_emb):
        B, C, H, W = P3.shape
        _, K, D = support_emb.shape

        # * Yolo projection
        F_y = self.yolo_head(P3)                 
        F_y = F_y / (F_y.norm(dim=1, keepdim=True) + 1e-6)

        # * Dino projection
        support_emb = support_emb.view(B*K, D)   
        z_s = self.dino_head(support_emb)        
        z_s = z_s / (z_s.norm(dim=1, keepdim=True) + 1e-6)
        z_s = z_s.view(B, K, -1)                 
        
        # * Compute similarity
        F_y_exp = F_y.unsqueeze(1)               
        z_s_exp = z_s.view(B, K, -1, 1, 1)       

        sim = (F_y_exp * z_s_exp).sum(dim=2)     

        sim_agg_max = sim.max(dim=1).values      
        sim_agg_mean = sim.mean(dim=1)          

        return sim_agg_max