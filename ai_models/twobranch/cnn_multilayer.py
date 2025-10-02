#!/usr/bin/env python3
"""
🧠 Enhanced Multi-Layer CNN for Amulet Recognition
ระบบ CNN หลายชั้นขั้นสูง พร้อมฟีเจอร์การสกัดคุณลักษณะระดับต่างๆ

Features:
- Multi-Scale Feature Extraction (การสกัดคุณลักษณะหลายระดับ)  
- Attention Mechanism (กลไกความสนใจ)
- Feature Pyramid Network (เครือข่ายปิรามิดคุณลักษณะ)
- Advanced Pooling Strategies (กลยุทธ์การรวมข้อมูลขั้นสุง)
- Edge Enhancement (การเพิ่มคุณภาพขอบ)
- Texture Analysis (การวิเคราะห์พื้นผิว)
- Spatial Attention (ความสนใจเชิงพื้นที่)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

class MultiScaleFeatureExtractor(nn.Module):
    """การสกัดคุณลักษณะแบบหลายระดับ"""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Layer 1: Edge and Line Detection (การตรวจหาขอบและเส้น)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Corner and Texture Detection (การตรวจหามุมและพื้นผิว)
        self.texture_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3: Pattern Recognition (การจดจำรูปแบบ)
        self.pattern_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Layer 4: Complex Feature Detection (การตรวจหาคุณลักษณะซับซ้อน)
        self.complex_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Pooling layers with different strategies
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        
        # Level 1: Edge features
        edge_feat = self.edge_conv(x)
        features['edges'] = edge_feat
        edge_pooled = self.max_pool(edge_feat)
        
        # Level 2: Texture features
        texture_feat = self.texture_conv(edge_pooled)
        features['textures'] = texture_feat
        texture_pooled = self.max_pool(texture_feat)
        
        # Level 3: Pattern features
        pattern_feat = self.pattern_conv(texture_pooled)
        features['patterns'] = pattern_feat
        pattern_pooled = self.max_pool(pattern_feat)
        
        # Level 4: Complex features
        complex_feat = self.complex_conv(pattern_pooled)
        features['complex'] = complex_feat
        
        return features

class SpatialAttention(nn.Module):
    """กลไกความสนใจเชิงพื้นที่"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate attention map
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention

class ChannelAttention(nn.Module):
    """กลไกความสนใจเชิงช่องสัญญาณ"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Average pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        
        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class FeaturePyramidNetwork(nn.Module):
    """เครือข่ายปิรามิดคุณลักษณะ"""
    
    def __init__(self, feature_dims: List[int]):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for dim in feature_dims:
            self.lateral_convs.append(
                nn.Conv2d(dim, 256, kernel_size=1)
            )
            self.fpn_convs.append(
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
            )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Build FPN from top to bottom
        fpn_features = []
        
        # Start from the highest level
        prev_feature = self.lateral_convs[-1](features[-1])
        fpn_features.append(self.fpn_convs[-1](prev_feature))
        
        # Propagate to lower levels
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            
            # Upsample previous level
            upsampled = F.interpolate(prev_feature, size=lateral.shape[-2:], 
                                    mode='nearest')
            
            # Combine
            combined = lateral + upsampled
            fpn_feature = self.fpn_convs[i](combined)
            fpn_features.insert(0, fpn_feature)
            prev_feature = combined
            
        return fpn_features

class EnhancedPooling(nn.Module):
    """การรวมข้อมูลขั้นสูง"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Learnable pooling
        self.learnable_pool = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Different pooling strategies
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # Learnable attention-based pooling
        attention = self.learnable_pool(x)
        attended = x * attention
        learned_out = torch.sum(attended.view(b, c, -1), dim=2, keepdim=True)
        learned_out = learned_out.view(b, c, 1, 1)
        
        # Combine all pooling strategies
        combined = avg_out + max_out + learned_out
        return combined.flatten(1)

class EnhancedMultiLayerCNN(nn.Module):
    """ระบบ CNN หลายชั้นขั้นสูงสำหรับพระเครื่อง"""
    
    def __init__(self, num_classes: int = 10, 
                 embedding_dim: int = 512,
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 use_fpn: bool = True):
        super().__init__()
        
        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor()
        
        # Attention modules
        if use_attention:
            self.spatial_attention = nn.ModuleDict({
                'edges': SpatialAttention(32),
                'textures': SpatialAttention(64),
                'patterns': SpatialAttention(128),
                'complex': SpatialAttention(256)
            })
            
            self.channel_attention = nn.ModuleDict({
                'edges': ChannelAttention(32),
                'textures': ChannelAttention(64),
                'patterns': ChannelAttention(128),
                'complex': ChannelAttention(256)
            })
        else:
            self.spatial_attention = None
            self.channel_attention = None
            
        # Feature Pyramid Network
        if use_fpn:
            self.fpn = FeaturePyramidNetwork([32, 64, 128, 256])
        else:
            self.fpn = None
            
        # Enhanced pooling
        self.enhanced_pooling = EnhancedPooling(256)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        features = self.feature_extractor(x)
        
        # Apply attention if available
        if self.spatial_attention is not None:
            for key in features:
                if key in self.spatial_attention:
                    features[key] = self.spatial_attention[key](features[key])
                    features[key] = self.channel_attention[key](features[key])
        
        # Apply FPN if available
        if self.fpn is not None:
            feature_list = [features['edges'], features['textures'], 
                          features['patterns'], features['complex']]
            fpn_features = self.fpn(feature_list)
            # Use the most complex FPN feature
            final_feature = fpn_features[-1]
        else:
            final_feature = features['complex']
        
        # Enhanced pooling
        pooled = self.enhanced_pooling(final_feature)
        
        # Classification
        logits = self.classifier(pooled)
        
        return {
            'logits': logits,
            'features': features,
            'pooled': pooled,
            'final_feature': final_feature
        }

class TwoBranchEnhancedCNN(nn.Module):
    """Two-Branch Enhanced CNN สำหรับภาพหน้า-หลัง"""
    
    def __init__(self, num_classes: int = 10,
                 embedding_dim: int = 512,
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 use_fpn: bool = True,
                 fusion_method: str = 'concat'):
        super().__init__()
        
        # Shared enhanced CNN
        self.shared_cnn = EnhancedMultiLayerCNN(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_attention=use_attention,
            use_fpn=use_fpn
        )
        
        # Branch-specific processing
        self.front_branch = nn.Sequential(
            nn.Linear(256, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.back_branch = nn.Sequential(
            nn.Linear(256, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            fusion_dim = embedding_dim
        elif fusion_method == 'add':
            fusion_dim = embedding_dim // 2
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
            
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
    def forward(self, front: torch.Tensor, back: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Process both branches
        front_out = self.shared_cnn(front)
        back_out = self.shared_cnn(back)
        
        # Branch-specific processing
        front_processed = self.front_branch(front_out['pooled'])
        back_processed = self.back_branch(back_out['pooled'])
        
        # Fusion
        if self.fusion_method == 'concat':
            fused = torch.cat([front_processed, back_processed], dim=1)
        elif self.fusion_method == 'add':
            fused = front_processed + back_processed
            
        # Final classification
        final_logits = self.fusion_classifier(fused)
        
        return {
            'logits': final_logits,
            'front_logits': front_out['logits'],
            'back_logits': back_out['logits'],
            'front_features': front_out['features'],
            'back_features': back_out['features'],
            'front_processed': front_processed,
            'back_processed': back_processed,
            'fused': fused
        }

# Factory functions
def create_enhanced_cnn(num_classes: int = 10, 
                       model_type: str = 'two_branch',
                       **kwargs) -> nn.Module:
    """สร้างโมเดล Enhanced CNN"""
    
    if model_type == 'single':
        return EnhancedMultiLayerCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'two_branch':
        return TwoBranchEnhancedCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """นับจำนวน parameters ในโมเดล"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module) -> Dict:
    """สรุปข้อมูลโมเดล"""
    total_params = count_parameters(model)
    
    return {
        'total_parameters': total_params,
        'total_parameters_millions': total_params / 1e6,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'model_type': type(model).__name__
    }

if __name__ == "__main__":
    # Test the enhanced CNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Single branch model
    print("=== Single Branch Enhanced CNN ===")
    single_model = create_enhanced_cnn(num_classes=10, model_type='single')
    single_summary = model_summary(single_model)
    print(f"Parameters: {single_summary['total_parameters_millions']:.2f}M")
    print(f"Model size: {single_summary['model_size_mb']:.2f}MB")
    
    # Two branch model
    print("\n=== Two Branch Enhanced CNN ===")
    two_branch_model = create_enhanced_cnn(num_classes=10, model_type='two_branch')
    two_branch_summary = model_summary(two_branch_model)
    print(f"Parameters: {two_branch_summary['total_parameters_millions']:.2f}M")
    print(f"Model size: {two_branch_summary['model_size_mb']:.2f}MB")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    batch_size = 2
    img_size = 224
    
    front_img = torch.randn(batch_size, 3, img_size, img_size)
    back_img = torch.randn(batch_size, 3, img_size, img_size)
    
    with torch.no_grad():
        output = two_branch_model(front_img, back_img)
        print(f"Output shape: {output['logits'].shape}")
        print(f"Front features keys: {list(output['front_features'].keys())}")
        print("✅ Enhanced Multi-Layer CNN is ready!")