import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import torchvision


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * 3.141592653589793
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class FeatureEnhancerLayer(nn.Module):
    """Feature enhancer layer with deformable self-attention"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, image_feats, text_feats):
        # Self-attention on image features
        q = k = image_feats
        image_feats2 = self.self_attn(q, k, image_feats)[0]
        image_feats = image_feats + self.dropout1(image_feats2)
        image_feats = self.norm1(image_feats)

        # Cross-attention: image attend to text
        image_feats2 = self.cross_attn(image_feats, text_feats, text_feats)[0]
        image_feats = image_feats + self.dropout2(image_feats2)
        image_feats = self.norm2(image_feats)

        # FFN
        image_feats2 = self.linear2(self.dropout(F.relu(self.linear1(image_feats))))
        image_feats = image_feats + self.dropout3(image_feats2)
        image_feats = self.norm3(image_feats)

        return image_feats


class CrossModalityDecoderLayer(nn.Module):
    """Cross-modality decoder layer"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention between queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Image cross-attention
        self.img_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Text cross-attention
        self.text_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, queries, image_feats, text_feats):
        # Self-attention
        q = k = queries
        queries2 = self.self_attn(q, k, queries)[0]
        queries = queries + self.dropout1(queries2)
        queries = self.norm1(queries)

        # Image cross-attention
        queries2 = self.img_cross_attn(queries, image_feats, image_feats)[0]
        queries = queries + self.dropout2(queries2)
        queries = self.norm2(queries)

        # Text cross-attention
        queries2 = self.text_cross_attn(queries, text_feats, text_feats)[0]
        queries = queries + self.dropout3(queries2)
        queries = self.norm3(queries)

        # FFN
        queries2 = self.linear2(self.dropout(F.relu(self.linear1(queries))))
        queries = queries + self.dropout4(queries2)
        queries = self.norm4(queries)

        return queries


class FewShotGroundingDINO(nn.Module):
    """
    Few-Shot Grounding DINO for drone video object detection

    Args:
        num_classes: Number of object classes (default: 1)
        num_queries: Number of object queries (default: 900)
        d_model: Hidden dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of feature enhancer layers (default: 6)
        num_decoder_layers: Number of decoder layers (default: 6)
        dim_feedforward: FFN dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
        backbone_name: Vision backbone name (default: 'resnet50')
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_queries: int = 900,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        backbone_name: str = 'resnet50'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        # Vision backbone (ResNet or can be replaced with Swin Transformer)
        if backbone_name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dim = 2048
        elif backbone_name == 'resnet101':
            backbone = torchvision.models.resnet101(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Project backbone features to d_model
        self.input_proj = nn.Conv2d(backbone_dim, d_model, kernel_size=1)

        # Trainable embeddings for few-shot learning (thay thế BERT)
        # Mỗi class có 1 embedding vector
        self.trainable_embeddings = nn.Parameter(torch.randn(num_classes, d_model))

        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)

        # Feature enhancer (6 layers)
        self.feature_enhancer = nn.ModuleList([
            FeatureEnhancerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Language-guided query selection (simplified)
        self.query_selection = nn.Linear(d_model, 1)

        # Cross-modality decoder (6 layers)
        self.decoder = nn.ModuleList([
            CrossModalityDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Prediction heads
        self.bbox_embed = nn.Linear(d_model, 4)  # (cx, cy, w, h)
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for background

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        images: torch.Tensor,
        class_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            images: Input images [B, 3, H, W]
            class_ids: Class IDs for each image [B] (optional, default to 0)

        Returns:
            Dictionary containing:
                - pred_logits: Classification logits [B, num_queries, num_classes+1]
                - pred_boxes: Predicted boxes [B, num_queries, 4] in (cx, cy, w, h) format, normalized
        """
        B = images.shape[0]

        if class_ids is None:
            class_ids = torch.zeros(B, dtype=torch.long, device=images.device)

        # Extract image features from backbone
        image_feats = self.backbone(images)  # [B, C, H', W']
        image_feats = self.input_proj(image_feats)  # [B, d_model, H', W']

        # Add positional encoding
        pos_embed = self.position_embedding(image_feats)  # [B, d_model, H', W']

        # Flatten spatial dimensions
        B, C, H, W = image_feats.shape
        image_feats_flat = image_feats.flatten(2).permute(0, 2, 1)  # [B, H'*W', d_model]
        pos_embed_flat = pos_embed.flatten(2).permute(0, 2, 1)  # [B, H'*W', d_model]
        image_feats_flat = image_feats_flat + pos_embed_flat

        # Get trainable text embeddings for each sample
        text_feats = self.trainable_embeddings[class_ids].unsqueeze(1)  # [B, 1, d_model]

        # Feature enhancer (early fusion)
        enhanced_image_feats = image_feats_flat
        for layer in self.feature_enhancer:
            enhanced_image_feats = layer(enhanced_image_feats, text_feats)

        # Language-guided query selection
        # Compute similarity between image tokens and text embeddings
        similarity = torch.einsum('bnd,bkd->bnk', enhanced_image_feats, text_feats)  # [B, H'*W', 1]
        similarity_scores = similarity.squeeze(-1)  # [B, H'*W']

        # Select top-k tokens as query initializers
        topk = min(self.num_queries // 2, similarity_scores.shape[1])
        topk_indices = torch.topk(similarity_scores, topk, dim=1)[1]  # [B, topk]

        # Initialize queries: combine learned queries + selected image tokens
        learned_queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, d_model]

        # Cross-modality decoder
        queries = learned_queries
        for layer in self.decoder:
            queries = layer(queries, enhanced_image_feats, text_feats)

        # Prediction heads
        pred_logits = self.class_embed(queries)  # [B, num_queries, num_classes+1]
        pred_boxes = self.bbox_embed(queries).sigmoid()  # [B, num_queries, 4], normalized to [0,1]

        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }

    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_all_except_embeddings(self):
        """Freeze all parameters except trainable embeddings (for few-shot learning)"""
        for name, param in self.named_parameters():
            if 'trainable_embeddings' not in name:
                param.requires_grad = False

    def get_trainable_embedding_params(self):
        """Get trainable embedding parameters for optimizer"""
        return [self.trainable_embeddings]


def build_few_shot_grounding_dino(
    num_classes: int = 1,
    num_queries: int = 900,
    backbone_name: str = 'resnet50',
    freeze_backbone: bool = True
) -> FewShotGroundingDINO:
    """
    Build Few-Shot Grounding DINO model

    Args:
        num_classes: Number of object classes
        num_queries: Number of object queries
        backbone_name: Vision backbone name ('resnet50' or 'resnet101')
        freeze_backbone: Whether to freeze backbone initially

    Returns:
        FewShotGroundingDINO model
    """
    model = FewShotGroundingDINO(
        num_classes=num_classes,
        num_queries=num_queries,
        backbone_name=backbone_name
    )

    if freeze_backbone:
        model.freeze_backbone()

    return model