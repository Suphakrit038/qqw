"""Image Processor Utility

ประมวลผลและปรับปรุงคุณภาพภาพ
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Any, Tuple
import cv2


class ImagePreprocessor:
    """คลาสสำหรับประมวลผลและปรับปรุงภาพ"""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'fair': 0.55,
            'poor': 0.40
        }
    
    def enhance_image(self, image: Image.Image, auto_adjust: bool = True) -> Tuple[Image.Image, Dict[str, Any]]:
        """ปรับปรุงคุณภาพภาพอัตโนมัติ"""
        
        enhanced_image = image.copy()
        enhancement_log = {
            'original_size': image.size,
            'enhancements_applied': [],
            'quality_before': 0,
            'quality_after': 0
        }
        
        try:
            # ประเมินคุณภาพเริ่มต้น
            quality_before = self.assess_image_quality(image)
            enhancement_log['quality_before'] = quality_before['overall_score']
            
            if auto_adjust:
                # ปรับความสว่าง
                if quality_before.get('brightness_score', 0.5) < 0.6:
                    enhanced_image = self._adjust_brightness(enhanced_image)
                    enhancement_log['enhancements_applied'].append('brightness')
                
                # ปรับคอนทราสต์
                if quality_before.get('contrast_score', 0.5) < 0.6:
                    enhanced_image = self._adjust_contrast(enhanced_image)
                    enhancement_log['enhancements_applied'].append('contrast')
                
                # ปรับความคมชัด
                if quality_before.get('sharpness_score', 0.5) < 0.6:
                    enhanced_image = self._adjust_sharpness(enhanced_image)
                    enhancement_log['enhancements_applied'].append('sharpness')
                
                # ลดสัญญาณรบกวน
                if quality_before.get('noise_score', 0.5) < 0.6:
                    enhanced_image = self._reduce_noise(enhanced_image)
                    enhancement_log['enhancements_applied'].append('noise_reduction')
            
            # ประเมินคุณภาพหลังปรับปรุง
            quality_after = self.assess_image_quality(enhanced_image)
            enhancement_log['quality_after'] = quality_after['overall_score']
            enhancement_log['final_size'] = enhanced_image.size
            
            return enhanced_image, enhancement_log
            
        except Exception as e:
            # ถ้าเกิดข้อผิดพลาด ส่งคืนภาพต้นฉบับ
            enhancement_log['error'] = str(e)
            return image, enhancement_log
    
    def assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """ประเมินคุณภาพภาพ"""
        
        try:
            # แปลงเป็น numpy array
            img_array = np.array(image)
            
            # ประเมินความสว่าง
            brightness_score = self._assess_brightness(img_array)
            
            # ประเมินคอนทราสต์
            contrast_score = self._assess_contrast(img_array)
            
            # ประเมินความคมชัด
            sharpness_score = self._assess_sharpness(img_array)
            
            # ประเมินสัญญาณรบกวน
            noise_score = self._assess_noise(img_array)
            
            # คำนวณคะแนนรวม
            overall_score = np.mean([brightness_score, contrast_score, sharpness_score, noise_score])
            
            # กำหนดระดับคุณภาพ
            quality_level = self._determine_quality_level(overall_score)
            
            return {
                'overall_score': overall_score,
                'quality_level': quality_level,
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
                'sharpness_score': sharpness_score,
                'noise_score': noise_score,
                'dimensions': image.size,
                'format': image.format
            }
            
        except Exception as e:
            return {
                'overall_score': 0.5,
                'quality_level': 'unknown',
                'error': str(e)
            }
    
    def _assess_brightness(self, img_array: np.ndarray) -> float:
        """ประเมินความสว่าง"""
        try:
            # แปลงเป็น grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # คำนวณความสว่างเฉลี่ย
            mean_brightness = np.mean(gray) / 255.0
            
            # คะแนนความสว่าง (ดีที่สุดที่ 0.4-0.7)
            if 0.4 <= mean_brightness <= 0.7:
                return 1.0
            elif 0.2 <= mean_brightness < 0.4 or 0.7 < mean_brightness <= 0.8:
                return 0.7
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    def _assess_contrast(self, img_array: np.ndarray) -> float:
        """ประเมินคอนทราสต์"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # คำนวณ standard deviation
            contrast = np.std(gray) / 255.0
            
            # ปรับให้เป็นคะแนน 0-1
            return min(contrast * 4, 1.0)
            
        except Exception:
            return 0.5
    
    def _assess_sharpness(self, img_array: np.ndarray) -> float:
        """ประเมินความคมชัด"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # ใช้ Laplacian operator
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # ปรับให้เป็นคะแนน 0-1
            return min(laplacian_var / 1000, 1.0)
            
        except Exception:
            return 0.5
    
    def _assess_noise(self, img_array: np.ndarray) -> float:
        """ประเมินสัญญาณรบกวน"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # ใช้ Gaussian blur เพื่อหาสัญญาณรบกวน
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.std(gray - blurred)
            
            # ปรับให้เป็นคะแนน (น้อยกว่า = ดีกว่า)
            noise_score = max(0, 1.0 - (noise / 50))
            return noise_score
            
        except Exception:
            return 0.5
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """กำหนดระดับคุณภาพ"""
        if overall_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif overall_score >= self.quality_thresholds['good']:
            return 'good'
        elif overall_score >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _adjust_brightness(self, image: Image.Image) -> Image.Image:
        """ปรับความสว่าง"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.2)  # เพิ่มความสว่าง 20%
    
    def _adjust_contrast(self, image: Image.Image) -> Image.Image:
        """ปรับคอนทราสต์"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.3)  # เพิ่มคอนทราสต์ 30%
    
    def _adjust_sharpness(self, image: Image.Image) -> Image.Image:
        """ปรับความคมชัด"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)  # เพิ่มความคมชัด 50%
    
    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """ลดสัญญาณรบกวน"""
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    def resize_for_analysis(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """ปรับขนาดภาพสำหรับการวิเคราะห์"""
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def normalize_image(self, image: Image.Image) -> np.ndarray:
        """ปรับค่าปกติของภาพสำหรับ ML model"""
        img_array = np.array(image)
        return img_array.astype(np.float32) / 255.0