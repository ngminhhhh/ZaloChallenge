import torch 
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict

class DINOv2Sim:
    def __init__(self, model_name:str='dinov2_vitb14', device:str='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"DINOv2 model loaded on {self.device}")


    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(img_tensor) # * (1, feature_dim=768)
        features = F.normalize(features, p=2, dim=1) # * L2 normalize

        return features


    @torch.no_grad()
    def extract_feature_map(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        feats = self.model.forward_features(img_tensor)

        patch_tokens = feats["x_norm_patchtokens"] # * (1, N, C)

        B, N, C = patch_tokens.shape

        h = w = int(N ** 0.5) 
        patch_tokens = patch_tokens.view(B, h, w, C)

        fmap = patch_tokens.permute(0, 3, 1, 2).contiguous()
        fmap = F.normalize(fmap, p=2, dim=1)

        return fmap[0] # * (C=768. H'=16, W'=16)


    def compute_similarity(self, query_features: torch.Tensor, support_embeddings: torch.Tensor) -> float:
        S = torch.einsum('nc,chw->nhw', support_embeddings, query_features)
        S, _ = torch.max(S, dim=0)

        return S


    def compute_support_embeddings(self, support_images: List[str]) -> torch.Tensor:
        support_features = []

        for img_path in support_images:
            img = Image.open(img_path).convert('RGB')
            features = self.extract_features(img)
            support_features.append(features)

        support_embedding = torch.cat(support_features, dim=0) 

        return support_embedding # * (num_support=3, dim=768)


    def mask_heatmap(self, S: torch.Tensor, threshold: float):
        S_cpu = S.detach().cpu()
        mask_torch = (S_cpu > threshold).to(torch.uint8) # {0, 1}
        mask_cv2 = (mask_torch.numpy() * 255).astype(np.uint8) # {0, 255}

        return mask_torch, mask_cv2

    def get_bboxes(self, cv_map:np.ndarray, heatmap:torch.Tensor, min_area: int=5):
        Hf, Wf = cv_map.shape
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv_map, connectivity=8
        )
        S = heatmap.detach().cpu().numpy()
        bboxes = []

        for label_id in range(1, num_labels):
            x, y, w, h, area = stats[label_id]
            if area < min_area: continue

            x1, y1 = x, y
            x2, y2 = x + w - 1, y + h - 1

            mask_component = (labels == label_id)
            comp_scores = S[mask_component]
            score = float(comp_scores.max())

            bboxes.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "score": score,
            })

        return bboxes
    
    def scale_bboxes(self, bboxes, img_w, img_h, f_w, f_h):
        scale_x = img_w / f_w
        scale_y = img_h / f_h

        bboxes_img = []
        for b in bboxes:
            x1_img = int(b["x1"] * scale_x)
            y1_img = int(b["y1"] * scale_y)
            x2_img = int((b["x2"] + 1) * scale_x)  
            y2_img = int((b["y2"] + 1) * scale_y)

            bboxes_img.append({
                "x1": x1_img,
                "y1": y1_img,
                "x2": x2_img,
                "y2": y2_img,
                "score": b["score"],
            })

        return bboxes_img

    def process_video(self, video_path: str, support_images: List[str], similarity_threshold: float=0.1, stride:int=1) -> List[Dict]:
        support_embeddings = self.compute_support_embeddings(support_images)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        detections = []
        frame_idx = 0

        # * Iterate video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            # * Convert frame to images
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # * Extract features
            query_features = self.extract_feature_map(pil_image)
            
            # * Compute heatmap
            heatmap = self.compute_similarity(query_features, support_embeddings) # * (w=16, h=16)
            torch_map, cv_map = self.mask_heatmap(heatmap, similarity_threshold)

            bbox_feat = self.get_bboxes(cv_map, heatmap)

            img_w, img_h = pil_image.size
            f_w, f_h = heatmap.shape
            bbox_img = self.scale_bboxes(bbox_feat, img_w, img_h, f_w, f_h)

            if bbox_img:
                res = max(bbox_img, key=lambda b:b["score"])

                detections.append({
                    "frame": frame_idx, 
                    "x1": res["x1"],
                    "y1": res["y1"],
                    "x2": res["x2"],
                    "y2": res["y2"]
                })

            frame_idx += 1

        cap.release()

        return detections
    
    def create_annotations(self, video_id, detections):
        annotations = {
            "video_id": video_id,
            "detections": [
                {
                    "bboxes": detections
                }
            ]
        }

        return annotations
    



