#!/usr/bin/env python3
"""
🎨 Realistic Amulet Image Generator
ระบบสร้างภาพพระเครื่องจำลองที่เหมือนจริง

Features:
- Style Transfer (การถ่ายทอดสไตล์)
- Texture Synthesis (การสังเคราะห์พื้นผิว)
- 3D-like Rendering (การเรนเดอร์แบบ 3 มิติ)
- Age and Wear Effects (เอฟเฟคการเก่าและสึกหรอ)
- Lighting and Shadow Simulation (การจำลองแสงและเงา)
- Material Property Simulation (การจำลองคุณสมบัติวัสดุ)
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Dict, List, Tuple, Optional, Union
import random
import math
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AmuletGenerationConfig:
    """การตั้งค่าการสร้างภาพพระเครื่อง"""
    
    # Basic settings
    output_size: Tuple[int, int] = (512, 512)
    batch_size: int = 32
    
    # Style settings
    material_types: List[str] = None
    age_levels: List[str] = None
    lighting_conditions: List[str] = None
    
    # Quality settings
    texture_detail: float = 0.8
    surface_roughness: float = 0.3
    wear_intensity: float = 0.4
    patina_strength: float = 0.2
    
    # Color variation
    color_variation: float = 0.15
    hue_shift_range: Tuple[float, float] = (-10, 10)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    
    # 3D effects
    depth_strength: float = 0.6
    highlight_intensity: float = 0.4
    shadow_depth: float = 0.3
    
    # Noise and imperfections
    add_noise: bool = True
    noise_strength: float = 0.05
    add_scratches: bool = True
    scratch_density: float = 0.1
    add_dust: bool = True
    dust_density: float = 0.05
    
    def __post_init__(self):
        if self.material_types is None:
            self.material_types = ['bronze', 'brass', 'clay', 'plaster', 'stone']
        if self.age_levels is None:
            self.age_levels = ['new', 'aged', 'antique', 'ancient']
        if self.lighting_conditions is None:
            self.lighting_conditions = ['natural', 'warm', 'cool', 'dramatic']

class MaterialRenderer:
    """ตัวเรนเดอร์วัสดุต่างๆ"""
    
    def __init__(self):
        self.material_properties = {
            'bronze': {
                'base_color': (139, 69, 19),
                'metallic': 0.8,
                'roughness': 0.4,
                'patina_color': (0, 128, 128)
            },
            'brass': {
                'base_color': (181, 166, 66),
                'metallic': 0.9,
                'roughness': 0.3,
                'patina_color': (34, 139, 34)
            },
            'clay': {
                'base_color': (160, 82, 45),
                'metallic': 0.0,
                'roughness': 0.8,
                'patina_color': (139, 69, 19)
            },
            'plaster': {
                'base_color': (245, 245, 220),
                'metallic': 0.0,
                'roughness': 0.9,
                'patina_color': (139, 131, 120)
            },
            'stone': {
                'base_color': (128, 128, 128),
                'metallic': 0.1,
                'roughness': 0.7,
                'patina_color': (105, 105, 105)
            }
        }
    
    def apply_material(self, image: np.ndarray, material: str, 
                      wear_level: float = 0.3) -> np.ndarray:
        """ใส่คุณสมบัติวัสดุให้กับภาพ"""
        
        if material not in self.material_properties:
            material = 'bronze'  # Default
        
        props = self.material_properties[material]
        result = image.copy()
        
        # Apply base color tinting
        base_color = np.array(props['base_color'], dtype=np.float32) / 255.0
        
        # Convert to LAB for better color manipulation
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab = lab.astype(np.float32)
        
        # Adjust color channels based on material
        lab[:, :, 1] += (base_color[1] - 0.5) * 50  # A channel
        lab[:, :, 2] += (base_color[2] - 0.5) * 50  # B channel
        
        # Apply metallic properties
        if props['metallic'] > 0.5:
            # Increase specular highlights for metallic materials
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            highlights = np.where(gray > 200, 1.0, 0.0)
            metallic_boost = highlights * props['metallic'] * 0.3
            
            lab[:, :, 0] += metallic_boost * 255  # Increase lightness
        
        # Apply roughness (affects how diffuse the surface looks)
        if props['roughness'] > 0.5:
            # Add subtle noise for rough surfaces
            noise = np.random.normal(0, props['roughness'] * 5, lab[:, :, 0].shape)
            lab[:, :, 0] += noise
        
        # Convert back to RGB
        lab = np.clip(lab, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # Apply patina/aging effects
        if wear_level > 0.2:
            patina_color = np.array(props['patina_color'], dtype=np.float32) / 255.0
            
            # Create patina mask (more patina in crevices and edges)
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create patina areas
            kernel = np.ones((3, 3), np.uint8)
            patina_mask = cv2.dilate(edges, kernel, iterations=2)
            patina_mask = cv2.GaussianBlur(patina_mask, (5, 5), 0)
            patina_mask = patina_mask.astype(np.float32) / 255.0
            
            # Apply patina
            patina_strength = wear_level * 0.5
            for i in range(3):
                result[:, :, i] = (result[:, :, i].astype(np.float32) * (1 - patina_mask * patina_strength) + 
                                  patina_color[i] * 255 * patina_mask * patina_strength).astype(np.uint8)
        
        return result

class LightingSimulator:
    """ตัวจำลองแสงและเงา"""
    
    def __init__(self):
        self.lighting_setups = {
            'natural': {
                'direction': (0.3, -0.5, 0.8),
                'color': (255, 248, 220),
                'intensity': 0.8
            },
            'warm': {
                'direction': (0.5, -0.3, 0.8),
                'color': (255, 230, 150),
                'intensity': 0.9
            },
            'cool': {
                'direction': (-0.3, -0.4, 0.9),
                'color': (200, 220, 255),
                'intensity': 0.7
            },
            'dramatic': {
                'direction': (0.8, -0.6, 0.3),
                'color': (255, 240, 180),
                'intensity': 1.2
            }
        }
    
    def apply_lighting(self, image: np.ndarray, depth_map: Optional[np.ndarray] = None,
                      lighting: str = 'natural') -> np.ndarray:
        """ใส่เอฟเฟคแสงและเงา"""
        
        if lighting not in self.lighting_setups:
            lighting = 'natural'
        
        setup = self.lighting_setups[lighting]
        result = image.copy().astype(np.float32)
        
        # Generate depth map if not provided
        if depth_map is None:
            depth_map = self._generate_depth_map(image)
        
        # Calculate lighting based on depth and surface normals
        light_dir = np.array(setup['direction'])
        light_color = np.array(setup['color']) / 255.0
        intensity = setup['intensity']
        
        # Estimate surface normals from depth map
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Normalize gradients to get surface normals
        normals_x = -grad_x / 255.0
        normals_y = -grad_y / 255.0
        normals_z = np.ones_like(normals_x) * 0.5
        
        # Calculate dot product with light direction
        dot_product = (normals_x * light_dir[0] + 
                      normals_y * light_dir[1] + 
                      normals_z * light_dir[2])
        
        # Clamp to positive values (surfaces facing the light)
        lighting_factor = np.maximum(dot_product, 0.1) * intensity
        
        # Apply lighting to each color channel
        for i in range(3):
            result[:, :, i] *= lighting_factor
            result[:, :, i] *= light_color[i]
        
        # Add specular highlights for shiny surfaces
        specular = np.power(np.maximum(dot_product, 0), 8) * 0.3
        result += specular[:, :, np.newaxis] * 100
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _generate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """สร้างแผนที่ความลึกจากภาพ"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use gradient magnitude as depth approximation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Invert so that edges are "deeper"
        depth = 255 - gradient_magnitude
        depth = cv2.GaussianBlur(depth, (5, 5), 0)
        
        return depth.astype(np.uint8)

class AgeingEffectGenerator:
    """ตัวสร้างเอฟเฟคการเก่าและสึกหรอ"""
    
    def __init__(self):
        pass
    
    def add_wear_effects(self, image: np.ndarray, wear_level: str = 'aged') -> np.ndarray:
        """เพิ่มเอฟเฟคการสึกหรอ"""
        
        wear_configs = {
            'new': {'noise': 0.01, 'scratches': 0.0, 'darkening': 0.05},
            'aged': {'noise': 0.03, 'scratches': 0.1, 'darkening': 0.15},
            'antique': {'noise': 0.05, 'scratches': 0.2, 'darkening': 0.25},
            'ancient': {'noise': 0.08, 'scratches': 0.3, 'darkening': 0.4}
        }
        
        config = wear_configs.get(wear_level, wear_configs['aged'])
        result = image.copy().astype(np.float32)
        
        # Add noise
        if config['noise'] > 0:
            noise = np.random.normal(0, config['noise'] * 255, result.shape)
            result += noise
        
        # Add scratches
        if config['scratches'] > 0:
            result = self._add_scratches(result, config['scratches'])
        
        # Add general darkening/aging
        if config['darkening'] > 0:
            result *= (1 - config['darkening'])
        
        # Add subtle color shifts for aging
        if wear_level in ['antique', 'ancient']:
            # Shift towards warmer, darker tones
            result[:, :, 0] *= 0.95  # Reduce red slightly
            result[:, :, 1] *= 0.90  # Reduce green more
            result[:, :, 2] *= 0.85  # Reduce blue most
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _add_scratches(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """เพิ่มรอยขีดข่วน"""
        result = image.copy()
        h, w = result.shape[:2]
        
        # Number of scratches based on intensity
        num_scratches = int(intensity * 20)
        
        for _ in range(num_scratches):
            # Random scratch parameters
            start_x = random.randint(0, w-1)
            start_y = random.randint(0, h-1)
            length = random.randint(10, 50)
            angle = random.uniform(0, 2 * math.pi)
            
            # Calculate end point
            end_x = int(start_x + length * math.cos(angle))
            end_y = int(start_y + length * math.sin(angle))
            
            # Ensure end point is within bounds
            end_x = max(0, min(w-1, end_x))
            end_y = max(0, min(h-1, end_y))
            
            # Draw scratch as dark line
            scratch_color = random.randint(0, 50)  # Dark scratch
            cv2.line(result, (start_x, start_y), (end_x, end_y), 
                    (scratch_color, scratch_color, scratch_color), 
                    thickness=random.randint(1, 2))
        
        return result

class RealisticAmuletGenerator:
    """ตัวสร้างภาพพระเครื่องจำลองที่เหมือนจริง"""
    
    def __init__(self, config: AmuletGenerationConfig = None):
        self.config = config or AmuletGenerationConfig()
        
        self.material_renderer = MaterialRenderer()
        self.lighting_simulator = LightingSimulator()
        self.ageing_generator = AgeingEffectGenerator()
        
    def generate_realistic_amulet(self, base_image: Union[np.ndarray, str],
                                 amulet_class: str,
                                 variation_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """สร้างภาพพระเครื่องจำลองที่เหมือนจริง"""
        
        # Load base image if path is provided
        if isinstance(base_image, str):
            base_image = cv2.imread(base_image)
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        
        # Use default variation parameters if not provided
        if variation_params is None:
            variation_params = self._generate_random_variation_params()
        
        # Resize to working resolution
        working_size = (512, 512)
        image = cv2.resize(base_image, working_size)
        
        results = {'original': base_image}
        
        # Step 1: Apply material properties
        material = variation_params.get('material', random.choice(self.config.material_types))
        wear_level = variation_params.get('wear_level', 0.3)
        
        material_image = self.material_renderer.apply_material(image, material, wear_level)
        results['material_applied'] = material_image
        
        # Step 2: Apply lighting
        lighting = variation_params.get('lighting', random.choice(self.config.lighting_conditions))
        lit_image = self.lighting_simulator.apply_lighting(material_image, lighting=lighting)
        results['lighting_applied'] = lit_image
        
        # Step 3: Add aging effects
        age_level = variation_params.get('age_level', random.choice(self.config.age_levels))
        aged_image = self.ageing_generator.add_wear_effects(lit_image, age_level)
        results['aged'] = aged_image
        
        # Step 4: Add final details and imperfections
        final_image = self._add_final_details(aged_image, variation_params)
        results['final'] = final_image
        
        # Resize to target output size
        if self.config.output_size != working_size:
            for key in results:
                if key != 'original':
                    results[key] = cv2.resize(results[key], self.config.output_size)
        
        # Add metadata
        results['metadata'] = {
            'amulet_class': amulet_class,
            'material': material,
            'lighting': lighting,
            'age_level': age_level,
            'wear_level': wear_level,
            'generation_timestamp': datetime.now().isoformat(),
            'variation_params': variation_params
        }
        
        return results
    
    def _generate_random_variation_params(self) -> Dict:
        """สร้างพารามิเตอร์การปรับแต่งแบบสุ่ม"""
        return {
            'material': random.choice(self.config.material_types),
            'lighting': random.choice(self.config.lighting_conditions),
            'age_level': random.choice(self.config.age_levels),
            'wear_level': random.uniform(0.1, 0.6),
            'color_shift': {
                'hue': random.uniform(*self.config.hue_shift_range),
                'saturation': random.uniform(*self.config.saturation_range),
                'brightness': random.uniform(*self.config.brightness_range)
            },
            'noise_level': random.uniform(0.01, self.config.noise_strength),
            'scratch_density': random.uniform(0.0, self.config.scratch_density),
            'dust_density': random.uniform(0.0, self.config.dust_density)
        }
    
    def _add_final_details(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """เพิ่มรายละเอียดสุดท้าย"""
        result = image.copy().astype(np.float32)
        
        # Color variations
        color_shift = params.get('color_shift', {})
        if color_shift:
            # Convert to HSV for easier color manipulation
            hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Apply hue shift
            if 'hue' in color_shift:
                hsv[:, :, 0] += color_shift['hue']
                hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
            
            # Apply saturation scaling
            if 'saturation' in color_shift:
                hsv[:, :, 1] *= color_shift['saturation']
            
            # Apply brightness scaling  
            if 'brightness' in color_shift:
                hsv[:, :, 2] *= color_shift['brightness']
            
            hsv = np.clip(hsv, 0, [179, 255, 255])
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Add noise if specified
        noise_level = params.get('noise_level', 0)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, result.shape)
            result += noise
        
        # Add dust effects
        dust_density = params.get('dust_density', 0)
        if dust_density > 0:
            result = self._add_dust(result, dust_density)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _add_dust(self, image: np.ndarray, density: float) -> np.ndarray:
        """เพิ่มเอฟเฟคฝุ่น"""
        result = image.copy()
        h, w = result.shape[:2]
        
        # Number of dust particles
        num_particles = int(density * w * h / 1000)
        
        for _ in range(num_particles):
            # Random dust particle
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            size = random.randint(1, 3)
            
            # Dust color (light particles)
            dust_color = random.randint(200, 255)
            
            cv2.circle(result, (x, y), size, 
                      (dust_color, dust_color, dust_color), -1)
        
        return result
    
    def generate_batch(self, base_images: List[Union[np.ndarray, str]],
                      amulet_classes: List[str],
                      num_variations_per_image: int = 5) -> List[Dict]:
        """สร้างภาพพระเครื่องเป็นชุด"""
        
        results = []
        
        for i, (base_image, amulet_class) in enumerate(zip(base_images, amulet_classes)):
            for j in range(num_variations_per_image):
                # Generate different variation for each iteration
                variation_params = self._generate_random_variation_params()
                
                # Generate realistic amulet
                generated = self.generate_realistic_amulet(
                    base_image, amulet_class, variation_params
                )
                
                # Add batch information
                generated['metadata']['batch_id'] = i
                generated['metadata']['variation_id'] = j
                
                results.append(generated)
        
        return results
    
    def save_generated_dataset(self, generated_batch: List[Dict],
                              output_dir: Path,
                              save_intermediates: bool = False) -> Dict:
        """บันทึกชุดข้อมูลที่สร้าง"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        images_dir = output_dir / 'images'
        metadata_dir = output_dir / 'metadata'
        images_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)
        
        if save_intermediates:
            intermediates_dir = output_dir / 'intermediates'
            intermediates_dir.mkdir(exist_ok=True)
        
        saved_info = {
            'total_images': len(generated_batch),
            'classes': {},
            'files': []
        }
        
        for i, generated in enumerate(generated_batch):
            metadata = generated['metadata']
            amulet_class = metadata['amulet_class']
            
            # Count classes
            if amulet_class not in saved_info['classes']:
                saved_info['classes'][amulet_class] = 0
            saved_info['classes'][amulet_class] += 1
            
            # Save final image
            filename = f"{amulet_class}_{i:04d}.jpg"
            image_path = images_dir / filename
            
            final_image = generated['final']
            final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), final_bgr)
            
            # Save metadata
            metadata_path = metadata_dir / f"{amulet_class}_{i:04d}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save intermediates if requested
            if save_intermediates:
                for stage in ['material_applied', 'lighting_applied', 'aged']:
                    if stage in generated:
                        stage_filename = f"{amulet_class}_{i:04d}_{stage}.jpg"
                        stage_path = intermediates_dir / stage_filename
                        
                        stage_image = generated[stage]
                        stage_bgr = cv2.cvtColor(stage_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(stage_path), stage_bgr)
            
            saved_info['files'].append({
                'image': str(image_path),
                'metadata': str(metadata_path),
                'class': amulet_class
            })
        
        # Save summary
        summary_path = output_dir / 'generation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(saved_info, f, indent=2)
        
        return saved_info

# Utility functions for integration with existing codebase
def enhance_existing_dataset(dataset_dir: Path, 
                           output_dir: Path,
                           variations_per_image: int = 3) -> Dict:
    """ปรับปรุงชุดข้อมูลที่มีอยู่ให้เหมือนจริงมากขึ้น"""
    
    generator = RealisticAmuletGenerator()
    
    # Find all images in dataset
    image_paths = []
    class_names = []
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            for image_path in class_dir.glob('*.jpg'):
                image_paths.append(str(image_path))
                class_names.append(class_dir.name)
    
    # Generate enhanced dataset
    enhanced_batch = generator.generate_batch(
        image_paths, class_names, variations_per_image
    )
    
    # Save results
    save_info = generator.save_generated_dataset(
        enhanced_batch, output_dir, save_intermediates=True
    )
    
    return save_info

if __name__ == "__main__":
    # Test the realistic amulet generator
    print("=== Realistic Amulet Generator Test ===")
    
    # Create test configuration
    config = AmuletGenerationConfig(
        output_size=(256, 256),
        texture_detail=0.8,
        add_noise=True,
        add_scratches=True
    )
    
    generator = RealisticAmuletGenerator(config)
    
    print(f"Materials: {config.material_types}")
    print(f"Age levels: {config.age_levels}")
    print(f"Lighting conditions: {config.lighting_conditions}")
    
    # Test with dummy image
    dummy_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    
    # Generate realistic version
    result = generator.generate_realistic_amulet(
        dummy_image, 
        'phra_somdej'
    )
    
    print(f"\\nGenerated stages: {list(result.keys())}")
    print(f"Final image shape: {result['final'].shape}")
    print(f"Metadata: {result['metadata']['material']} material, {result['metadata']['age_level']} age")
    
    print("✅ Realistic Amulet Generator is ready!")
    print("\\n💡 Next steps:")
    print("1. Use enhance_existing_dataset() to improve current dataset")
    print("2. Generate new variations with generate_batch()")
    print("3. Integrate with training pipeline for better model accuracy")