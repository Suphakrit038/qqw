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
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2

@dataclass
class EnhancedPreprocessConfig:
    """Configuration for enhanced preprocessing"""
    
    # Basic settings
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Edge enhancement
    use_edge_enhancement: bool = True
    edge_method: str = 'canny'  # 'canny', 'sobel', 'laplacian'
    edge_alpha: float = 0.3
    
    # Texture enhancement
    use_texture_enhancement: bool = True
    texture_method: str = 'gabor'  # 'gabor', 'lbp', 'glcm'
    
    # Adaptive contrast
    use_adaptive_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
    # Noise reduction
    use_noise_reduction: bool = True
    noise_method: str = 'bilateral'  # 'bilateral', 'gaussian', 'median'
    
    # Multi-scale processing
    use_multi_scale: bool = True
    scale_factors: List[float] = (0.8, 1.0, 1.2)
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.8

class EdgeEnhancer:
    """Edge enhancement module"""
    
    def __init__(self, method: str = 'canny', alpha: float = 0.3):
        self.method = method
        self.alpha = alpha
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges in the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if self.method == 'canny':
            edges = cv2.Canny(gray, 50, 150)
        elif self.method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif self.method == 'laplacian':
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            return image
            
        # Convert edges to 3-channel if needed
        if len(image.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
        # Blend with original image
        enhanced = cv2.addWeighted(image, 1-self.alpha, edges, self.alpha, 0)
        return enhanced

class TextureEnhancer:
    """Texture enhancement module"""
    
    def __init__(self, method: str = 'gabor'):
        self.method = method
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance texture features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if self.method == 'gabor':
            enhanced = self._gabor_enhancement(gray)
        elif self.method == 'lbp':
            enhanced = self._lbp_enhancement(gray)
        else:
            return image
            
        # Convert back to 3-channel if needed
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
        return enhanced
        
    def _gabor_enhancement(self, gray: np.ndarray) -> np.ndarray:
        """Apply Gabor filter for texture enhancement"""
        filtered_images = []
        
        # Apply Gabor filters with different orientations
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 
                                      2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            filtered_images.append(filtered)
            
        # Combine all filtered images
        enhanced = np.maximum.reduce(filtered_images)
        return enhanced
        
    def _lbp_enhancement(self, gray: np.ndarray) -> np.ndarray:
        """Apply Local Binary Pattern for texture enhancement"""
        # Simple LBP implementation
        rows, cols = gray.shape
        enhanced = np.zeros_like(gray)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = gray[i, j]
                binary_string = ''
                
                # 8-neighborhood
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                    
                enhanced[i, j] = int(binary_string, 2)
                
        return enhanced

class AdaptiveContrastEnhancer:
    """Adaptive contrast enhancement module"""
    
    def __init__(self, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast adaptively using CLAHE"""
        if len(image.shape) == 3:
            # Apply CLAHE to each channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
            enhanced = clahe.apply(image)
            
        return enhanced

class NoiseReducer:
    """Noise reduction module"""
    
    def __init__(self, method: str = 'bilateral'):
        self.method = method
        
    def reduce(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise in the image"""
        if self.method == 'bilateral':
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        elif self.method == 'gaussian':
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif self.method == 'median':
            denoised = cv2.medianBlur(image, 5)
        else:
            return image
            
        return denoised

class MultiScaleProcessor:
    """Multi-scale processing module"""
    
    def __init__(self, scale_factors: List[float] = [0.8, 1.0, 1.2]):
        self.scale_factors = scale_factors
        
    def process(self, image: np.ndarray, target_size: Tuple[int, int]) -> List[np.ndarray]:
        """Process image at multiple scales"""
        processed_images = []
        
        for scale in self.scale_factors:
            # Calculate scaled size
            scaled_size = (int(target_size[0] * scale), int(target_size[1] * scale))
            
            # Resize image
            scaled_image = cv2.resize(image, scaled_size)
            
            # Resize back to target size
            final_image = cv2.resize(scaled_image, target_size)
            processed_images.append(final_image)
            
        return processed_images

class EnhancedPreprocessor:
    """Main enhanced preprocessing class"""
    
    def __init__(self, config: EnhancedPreprocessConfig = None):
        self.config = config or EnhancedPreprocessConfig()
        
        # Initialize enhancement modules
        if self.config.use_edge_enhancement:
            self.edge_enhancer = EdgeEnhancer(
                self.config.edge_method, 
                self.config.edge_alpha
            )
            
        if self.config.use_texture_enhancement:
            self.texture_enhancer = TextureEnhancer(self.config.texture_method)
            
        if self.config.use_adaptive_contrast:
            self.contrast_enhancer = AdaptiveContrastEnhancer(
                self.config.clahe_clip_limit,
                self.config.clahe_grid_size
            )
            
        if self.config.use_noise_reduction:
            self.noise_reducer = NoiseReducer(self.config.noise_method)
            
        if self.config.use_multi_scale:
            self.multi_scale_processor = MultiScaleProcessor(self.config.scale_factors)
            
        # Setup augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline using Albumentations"""
        transforms = []
        
        if self.config.use_augmentation:
            transforms.extend([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.ElasticTransform(p=0.1),
                ], p=0.2),
                A.HueSaturationValue(p=0.3),
            ])
            
        # Always include resize and normalization
        transforms.extend([
            A.Resize(self.config.image_size[0], self.config.image_size[1]),
            A.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
        
    def preprocess(self, 
                   image: Union[np.ndarray, Image.Image], 
                   apply_augmentation: bool = False) -> torch.Tensor:
        """
        Main preprocessing function
        
        Args:
            image: Input image (numpy array or PIL Image)
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        elif len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Apply enhancements
        enhanced_image = image.copy()
        
        if self.config.use_noise_reduction:
            enhanced_image = self.noise_reducer.reduce(enhanced_image)
            
        if self.config.use_adaptive_contrast:
            enhanced_image = self.contrast_enhancer.enhance(enhanced_image)
            
        if self.config.use_edge_enhancement:
            enhanced_image = self.edge_enhancer.enhance(enhanced_image)
            
        if self.config.use_texture_enhancement:
            enhanced_image = self.texture_enhancer.enhance(enhanced_image)
            
        # Apply augmentation pipeline
        if apply_augmentation and self.config.use_augmentation:
            augmented = self.augmentation_pipeline(image=enhanced_image)
            return augmented['image']
        else:
            # Apply only resize and normalization
            basic_pipeline = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                ),
                ToTensorV2()
            ])
            processed = basic_pipeline(image=enhanced_image)
            return processed['image']
            
    def preprocess_pair(self, 
                       front_image: Union[np.ndarray, Image.Image],
                       back_image: Union[np.ndarray, Image.Image],
                       apply_augmentation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a pair of images (front and back)
        
        Args:
            front_image: Front view image
            back_image: Back view image
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Tuple of preprocessed tensors (front, back)
        """
        front_tensor = self.preprocess(front_image, apply_augmentation)
        back_tensor = self.preprocess(back_image, apply_augmentation)
        
        return front_tensor, back_tensor
        
    def preprocess_multi_scale(self, 
                              image: Union[np.ndarray, Image.Image]) -> List[torch.Tensor]:
        """
        Preprocess image at multiple scales
        
        Args:
            image: Input image
            
        Returns:
            List of preprocessed tensors at different scales
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if self.config.use_multi_scale:
            multi_scale_images = self.multi_scale_processor.process(
                image, self.config.image_size
            )
            
            tensors = []
            for scale_image in multi_scale_images:
                tensor = self.preprocess(scale_image, apply_augmentation=False)
                tensors.append(tensor)
                
            return tensors
        else:
            return [self.preprocess(image, apply_augmentation=False)]

# Utility functions
def visualize_preprocessing_steps(preprocessor: EnhancedPreprocessor, 
                                 image_path: str, 
                                 save_path: Optional[str] = None):
    """Visualize different preprocessing steps"""
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply different enhancements
    steps = {'Original': image}
    
    if preprocessor.config.use_noise_reduction:
        denoised = preprocessor.noise_reducer.reduce(image)
        steps['Noise Reduced'] = denoised
        
    if preprocessor.config.use_adaptive_contrast:
        contrast_enhanced = preprocessor.contrast_enhancer.enhance(image)
        steps['Contrast Enhanced'] = contrast_enhanced
        
    if preprocessor.config.use_edge_enhancement:
        edge_enhanced = preprocessor.edge_enhancer.enhance(image)
        steps['Edge Enhanced'] = edge_enhanced
        
    if preprocessor.config.use_texture_enhancement:
        texture_enhanced = preprocessor.texture_enhancer.enhance(image)
        steps['Texture Enhanced'] = texture_enhanced
        
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (title, img) in enumerate(steps.items()):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis('off')
            
    # Hide unused subplots
    for i in range(len(steps), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    # Test the enhanced preprocessor
    print("🧪 Testing Enhanced Preprocessor...")
    
    # Create config
    config = EnhancedPreprocessConfig(
        image_size=(224, 224),
        use_edge_enhancement=True,
        use_texture_enhancement=True,
        use_adaptive_contrast=True,
        use_noise_reduction=True,
        use_augmentation=True
    )
    
    # Create preprocessor
    preprocessor = EnhancedPreprocessor(config)
    print("✅ Enhanced preprocessor created")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Single image preprocessing
    processed_tensor = preprocessor.preprocess(dummy_image)
    print(f"✅ Single image processed: {processed_tensor.shape}")
    
    # Pair preprocessing
    front_tensor, back_tensor = preprocessor.preprocess_pair(dummy_image, dummy_image)
    print(f"✅ Image pair processed: {front_tensor.shape}, {back_tensor.shape}")
    
    # Multi-scale preprocessing
    multi_scale_tensors = preprocessor.preprocess_multi_scale(dummy_image)
    print(f"✅ Multi-scale processed: {len(multi_scale_tensors)} scales")
    
    print("🎉 Enhanced Preprocessor ready for use!")