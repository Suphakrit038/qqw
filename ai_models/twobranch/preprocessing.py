#!/usr/bin/env python3
"""
🎨 Advanced Image Preprocessing for Enhanced CNN
ระบบประมวลผลภาพขั้นสูงสำหรับ Enhanced CNN

Features:
- Edge Enhancement (การเพิ่มคุณภาพขอบ)
- Texture Analysis (การวิเคราะห์พื้นผิว)
- Multi-Scale Preprocessing (การประมวลผลหลายระดับ)
- Adaptive Contrast Enhancement (การเพิ่มคอนทราสต์แบบปรับตัว)
- Noise Reduction (การลดสัญญาณรบกวน)
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2

@dataclass
class EnhancedPreprocessConfig:
    """การตั้งค่าการประมวลผลภาพขั้นสูง"""
    img_size: Tuple[int, int] = (224, 224)
    
    # Edge enhancement settings
    enhance_edges: bool = True
    edge_enhancement_strength: float = 0.3
    
    # Texture analysis
    enhance_texture: bool = True
    texture_strength: float = 0.2
    
    # Contrast enhancement
    adaptive_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Noise reduction
    denoise: bool = True
    denoise_strength: float = 10.0
    
    # Multi-scale processing
    multi_scale: bool = True
    scale_factors: List[float] = None
    
    # Normalization
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [0.8, 1.0, 1.2]

class EdgeEnhancer:
    """ตัวเพิ่มคุณภาพขอบ"""
    
    def __init__(self, strength: float = 0.3):
        self.strength = strength
        
        # Different edge detection kernels
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        self.laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        self.unsharp_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """เพิ่มคุณภาพขอบในภาพ"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply different edge detection methods
        edges_sobel_x = cv2.filter2D(gray, -1, self.sobel_x)
        edges_sobel_y = cv2.filter2D(gray, -1, self.sobel_y)
        edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
        
        edges_laplacian = cv2.filter2D(gray, -1, self.laplacian)
        
        # Combine edge information
        combined_edges = np.maximum(edges_sobel, np.abs(edges_laplacian))
        
        # Normalize edges
        if combined_edges.max() > 0:
            combined_edges = combined_edges / combined_edges.max()
        
        # Convert back to 3-channel if needed
        if len(image.shape) == 3:
            edges_3ch = np.stack([combined_edges] * 3, axis=-1)
            enhanced = image.astype(np.float32) + self.strength * edges_3ch * 255
        else:
            enhanced = image.astype(np.float32) + self.strength * combined_edges * 255
            
        return np.clip(enhanced, 0, 255).astype(np.uint8)

class TextureEnhancer:
    """ตัวเพิ่มคุณภาพพื้นผิว"""
    
    def __init__(self, strength: float = 0.2):
        self.strength = strength
    
    def enhance_texture(self, image: np.ndarray) -> np.ndarray:
        """เพิ่มคุณภาพพื้นผิวในภาพ"""
        # Convert to LAB color space for better texture enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image.copy()
        
        # Apply unsharp masking for texture enhancement
        gaussian_blur = cv2.GaussianBlur(l_channel, (0, 0), 1.0)
        unsharp = cv2.addWeighted(l_channel, 1.0 + self.strength, gaussian_blur, -self.strength, 0)
        
        if len(image.shape) == 3:
            lab[:, :, 0] = unsharp
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = unsharp
            
        return enhanced

class AdaptiveContrastEnhancer:
    """ตัวเพิ่มคอนทราสต์แบบปรับตัว"""
    
    def __init__(self, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """เพิ่มคอนทราสต์แบบปรับตัว"""
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = self.clahe.apply(image)
            
        return enhanced

class NoiseReducer:
    """ตัวลดสัญญาณรบกวน"""
    
    def __init__(self, strength: float = 10.0):
        self.strength = strength
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """ลดสัญญาณรบกวนในภาพ"""
        if len(image.shape) == 3:
            # Use Non-local Means Denoising for color images
            denoised = cv2.fastNlMeansDenoisingColored(image, None, self.strength, self.strength, 7, 21)
        else:
            # Use grayscale denoising
            denoised = cv2.fastNlMeansDenoising(image, None, self.strength, 7, 21)
            
        return denoised

class MultiScaleProcessor:
    """ตัวประมวลผลแบบหลายระดับ"""
    
    def __init__(self, scale_factors: List[float] = None):
        self.scale_factors = scale_factors or [0.8, 1.0, 1.2]
    
    def process_multi_scale(self, image: np.ndarray, 
                          target_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """ประมวลผลภาพในหลายระดับ"""
        processed = {}
        
        for i, scale in enumerate(self.scale_factors):
            # Calculate new size
            new_h = int(target_size[0] * scale)
            new_w = int(target_size[1] * scale)
            
            # Resize image
            if scale != 1.0:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                # Resize back to target size
                processed_scale = cv2.resize(resized, target_size, interpolation=cv2.INTER_CUBIC)
            else:
                processed_scale = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
            
            processed[f'scale_{scale}'] = processed_scale
            
        return processed

class EnhancedPreprocessor:
    """ตัวประมวลผลภาพขั้นสูง"""
    
    def __init__(self, config: EnhancedPreprocessConfig = None):
        self.config = config or EnhancedPreprocessConfig()
        
        # Initialize enhancers
        self.edge_enhancer = EdgeEnhancer(self.config.edge_enhancement_strength)
        self.texture_enhancer = TextureEnhancer(self.config.texture_strength)
        self.contrast_enhancer = AdaptiveContrastEnhancer(
            self.config.clahe_clip_limit, 
            self.config.clahe_tile_size
        )
        self.noise_reducer = NoiseReducer(self.config.denoise_strength)
        self.multi_scale_processor = MultiScaleProcessor(self.config.scale_factors)
        
        # Create augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """สร้าง pipeline การ augment ข้อมูล"""
        transforms = []
        
        # Basic geometric transforms
        transforms.extend([
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            )
        ])
        
        # Color and lighting transforms
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3)
        ])
        
        # Resize and normalize
        transforms.extend([
            A.Resize(height=self.config.img_size[0], width=self.config.img_size[1]),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def preprocess_single(self, image: Union[np.ndarray, Image.Image, str], 
                         training: bool = False) -> Dict[str, torch.Tensor]:
        """ประมวลผลภาพเดี่ยว"""
        
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass  # Already RGB
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        original_image = image.copy()
        
        # Apply enhancements
        if self.config.denoise:
            image = self.noise_reducer.reduce_noise(image)
        
        if self.config.adaptive_contrast:
            image = self.contrast_enhancer.enhance_contrast(image)
        
        if self.config.enhance_texture:
            image = self.texture_enhancer.enhance_texture(image)
            
        if self.config.enhance_edges:
            image = self.edge_enhancer.enhance_edges(image)
        
        results = {'enhanced': image}
        
        # Multi-scale processing
        if self.config.multi_scale:
            multi_scale_results = self.multi_scale_processor.process_multi_scale(
                image, self.config.img_size
            )
            results.update(multi_scale_results)
        
        # Apply augmentation pipeline if training
        if training:
            # Apply to enhanced image
            augmented = self.augmentation_pipeline(image=image)
            results['tensor'] = augmented['image']
        else:
            # Simple resize and normalize for inference
            resized = cv2.resize(image, self.config.img_size)
            if self.config.normalize:
                normalized = resized.astype(np.float32) / 255.0
                normalized = (normalized - np.array(self.config.mean)) / np.array(self.config.std)
                tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
            else:
                tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
            results['tensor'] = tensor
        
        return results
    
    def preprocess_pair(self, front_image: Union[np.ndarray, Image.Image, str],
                       back_image: Union[np.ndarray, Image.Image, str],
                       training: bool = False) -> Dict[str, torch.Tensor]:
        """ประมวลผลคู่ภาพ (หน้า-หลัง)"""
        
        front_results = self.preprocess_single(front_image, training=training)
        back_results = self.preprocess_single(back_image, training=training)
        
        return {
            'front': front_results['tensor'],
            'back': back_results['tensor'],
            'front_enhanced': front_results['enhanced'],
            'back_enhanced': back_results['enhanced']
        }
    
    def batch_preprocess(self, image_pairs: List[Tuple[str, str]],
                        training: bool = False) -> Dict[str, torch.Tensor]:
        """ประมวลผลภาพเป็นชุด"""
        
        front_tensors = []
        back_tensors = []
        
        for front_path, back_path in image_pairs:
            pair_results = self.preprocess_pair(front_path, back_path, training=training)
            front_tensors.append(pair_results['front'].unsqueeze(0))
            back_tensors.append(pair_results['back'].unsqueeze(0))
        
        return {
            'front_batch': torch.cat(front_tensors, dim=0),
            'back_batch': torch.cat(back_tensors, dim=0)
        }

# Utility functions
def visualize_preprocessing_steps(preprocessor: EnhancedPreprocessor, 
                                 image_path: str, 
                                 save_path: Optional[str] = None):
    """แสดงขั้นตอนการประมวลผลภาพ"""
    import matplotlib.pyplot as plt
    
    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Process with individual steps
    denoised = preprocessor.noise_reducer.reduce_noise(original.copy())
    contrast_enhanced = preprocessor.contrast_enhancer.enhance_contrast(denoised.copy())
    texture_enhanced = preprocessor.texture_enhancer.enhance_texture(contrast_enhanced.copy())
    edge_enhanced = preprocessor.edge_enhancer.enhance_edges(texture_enhanced.copy())
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denoised)
    axes[0, 1].set_title('Denoised')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(contrast_enhanced)
    axes[0, 2].set_title('Contrast Enhanced')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(texture_enhanced)
    axes[1, 0].set_title('Texture Enhanced')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edge_enhanced)
    axes[1, 1].set_title('Edge Enhanced')
    axes[1, 1].axis('off')
    
    # Show multi-scale processing
    multi_scale = preprocessor.multi_scale_processor.process_multi_scale(
        edge_enhanced, (224, 224)
    )
    axes[1, 2].imshow(multi_scale['scale_1.2'])
    axes[1, 2].set_title('Multi-scale (1.2x)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Test the enhanced preprocessor
    config = EnhancedPreprocessConfig()
    preprocessor = EnhancedPreprocessor(config)
    
    print("=== Enhanced Preprocessor Test ===")
    print(f"Image size: {config.img_size}")
    print(f"Edge enhancement: {config.enhance_edges}")
    print(f"Texture enhancement: {config.enhance_texture}")
    print(f"Adaptive contrast: {config.adaptive_contrast}")
    print(f"Noise reduction: {config.denoise}")
    print(f"Multi-scale: {config.multi_scale}")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    result = preprocessor.preprocess_single(dummy_image, training=False)
    print(f"\\nOutput tensor shape: {result['tensor'].shape}")
    print(f"Enhanced image shape: {result['enhanced'].shape}")
    
    if config.multi_scale:
        for key in result:
            if key.startswith('scale_'):
                print(f"{key} shape: {result[key].shape}")
    
    print("✅ Enhanced Preprocessor is ready!")