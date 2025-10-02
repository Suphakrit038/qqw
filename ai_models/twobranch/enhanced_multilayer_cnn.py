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
import numpy as np
import math

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction module"""
    
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        
        # Different kernel sizes for multi-scale extraction
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Extract features at different scales
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        f7 = self.conv7x7(x)
        
        # Concatenate multi-scale features
        out = torch.cat([f1, f3, f5, f7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-level feature fusion"""
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
            
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from different levels
        Returns:
            List of enhanced feature maps
        """
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest')
            
        # Apply FPN convolutions
        fpn_outs = [fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)]
        
        return fpn_outs

class EnhancedPooling(nn.Module):
    """Enhanced pooling with multiple strategies"""
    
    def __init__(self, in_channels: int, pool_type: str = 'mixed'):
        super().__init__()
        self.pool_type = pool_type
        
        if pool_type == 'mixed':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        elif pool_type == 'gem':
            self.gem_pool = GeneralizedMeanPooling()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            
    def forward(self, x):
        if self.pool_type == 'mixed':
            avg_out = self.avg_pool(x)
            max_out = self.max_pool(x)
            out = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(out)
        elif self.pool_type == 'gem':
            out = self.gem_pool(x)
        else:
            out = self.pool(x)
            
        return out.flatten(1)

class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling"""
    
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                           kernel_size=x.size()[2:]).pow(1./self.p)

class EnhancedMultiLayerCNN(nn.Module):
    """Enhanced Multi-Layer CNN with advanced features"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 backbone: str = 'resnet50',
                 use_attention: bool = True,
                 use_fpn: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_fpn = use_fpn
        
        # Load backbone
        self.backbone = self._create_backbone(backbone)
        
        # Get feature dimensions
        backbone_channels = self._get_backbone_channels(backbone)
        
        # Multi-scale feature extraction
        self.multi_scale_extractor = MultiScaleFeatureExtractor(
            backbone_channels[-1], 512
        )
        
        # Attention mechanisms
        if use_attention:
            self.channel_attention = ChannelAttention(512)
            self.spatial_attention = SpatialAttention()
            
        # Feature Pyramid Network
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(backbone_channels, 256)
            final_channels = 256 * len(backbone_channels)
        else:
            final_channels = 512
            
        # Enhanced pooling
        self.enhanced_pooling = EnhancedPooling(final_channels, 'mixed')
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_channels, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def _create_backbone(self, backbone: str):
        """Create backbone network"""
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            return nn.Sequential(*list(model.children())[:-2])
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=True)
            return model.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
    def _get_backbone_channels(self, backbone: str) -> List[int]:
        """Get channel dimensions for each level"""
        if backbone == 'resnet50':
            return [256, 512, 1024, 2048]
        elif backbone == 'efficientnet_b0':
            return [40, 80, 192, 320]
        else:
            return [256, 512, 1024, 2048]
            
    def forward(self, x):
        # Extract backbone features
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 5, 6, 7]:  # Collect intermediate features
                features.append(x)
                
        # Multi-scale feature extraction
        x = self.multi_scale_extractor(x)
        
        # Apply attention
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            
        # Feature pyramid network
        if self.use_fpn and len(features) > 1:
            fpn_features = self.fpn(features)
            # Global average pooling for each level
            pooled_features = []
            for feat in fpn_features:
                pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                pooled_features.append(pooled)
            x = torch.cat(pooled_features, dim=1)
        else:
            # Enhanced pooling
            x = self.enhanced_pooling(x)
            
        # Classification
        x = self.classifier(x)
        
        return x

class TwoBranchEnhancedCNN(nn.Module):
    """Two-branch enhanced CNN for front and back image processing"""
    
    def __init__(self, 
                 num_classes: int = 10,
                 backbone: str = 'resnet50',
                 fusion_method: str = 'concat',
                 **kwargs):
        super().__init__()
        
        # Shared enhanced CNN for both branches
        self.front_branch = EnhancedMultiLayerCNN(
            num_classes=num_classes, 
            backbone=backbone,
            **kwargs
        )
        self.back_branch = EnhancedMultiLayerCNN(
            num_classes=num_classes,
            backbone=backbone, 
            **kwargs
        )
        
        self.fusion_method = fusion_method
        
        # Feature fusion
        if fusion_method == 'concat':
            self.fusion_fc = nn.Sequential(
                nn.Linear(num_classes * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        elif fusion_method == 'attention':
            self.attention_weights = nn.Linear(num_classes * 2, 2)
            
    def forward(self, front_img, back_img):
        # Process both images
        front_features = self.front_branch(front_img)
        back_features = self.back_branch(back_img)
        
        # Fusion
        if self.fusion_method == 'concat':
            combined = torch.cat([front_features, back_features], dim=1)
            output = self.fusion_fc(combined)
        elif self.fusion_method == 'attention':
            combined = torch.cat([front_features, back_features], dim=1)
            attention_weights = F.softmax(self.attention_weights(combined), dim=1)
            output = (attention_weights[:, 0:1] * front_features + 
                     attention_weights[:, 1:2] * back_features)
        else:  # average
            output = (front_features + back_features) / 2
            
        return output

# Factory functions
def create_enhanced_cnn(num_classes: int = 10, 
                       model_type: str = 'two_branch',
                       **kwargs) -> nn.Module:
    """Factory function to create enhanced CNN models"""
    
    if model_type == 'single':
        return EnhancedMultiLayerCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'two_branch':
        return TwoBranchEnhancedCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """Print model summary"""
    from torchsummary import summary
    try:
        summary(model, input_size)
    except:
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if __name__ == "__main__":
    # Test the models
    print("🧪 Testing Enhanced Multi-Layer CNN...")
    
    # Single branch model
    model_single = create_enhanced_cnn(num_classes=10, model_type='single')
    print(f"✅ Single branch model created: {sum(p.numel() for p in model_single.parameters()):,} parameters")
    
    # Two branch model
    model_two_branch = create_enhanced_cnn(num_classes=10, model_type='two_branch')
    print(f"✅ Two branch model created: {sum(p.numel() for p in model_two_branch.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 4
    front_img = torch.randn(batch_size, 3, 224, 224)
    back_img = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model_two_branch(front_img, back_img)
        print(f"✅ Forward pass successful: {output.shape}")
    
    print("🎉 Enhanced Multi-Layer CNN ready for use!")