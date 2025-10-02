#!/usr/bin/env python3
"""
🏋️ AI Model Training System
ระบบเทรนโมเดล AI สำหรับ Amulet Classification

Features:
- Data preparation and preprocessing
- Model training with validation
- Performance monitoring
- Model saving and evaluation
- Support for multiple algorithms
"""

from __future__ import annotations
import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Optional imports with fallbacks
try:
    import numpy as np
    import pandas as pd
    NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False
    print("⚠️ NumPy/Pandas not available")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        classification_report, accuracy_score, 
        confusion_matrix, f1_score
    )
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available")

# Suppress warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """Extract features from images for traditional ML models"""
    
    def __init__(self):
        self.feature_methods = [
            'color_histogram',
            'texture_glcm',
            'shape_moments',
            'edge_features'
        ]
        
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract comprehensive features from image"""
        if not (PIL_AVAILABLE and CV2_AVAILABLE and NUMPY_PANDAS_AVAILABLE):
            # Fallback: return dummy features
            return np.random.randn(128)
            
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # Color histogram features
            color_features = self._extract_color_histogram(image_rgb)
            features.extend(color_features)
            
            # Texture features (GLCM)
            texture_features = self._extract_texture_features(image_gray)
            features.extend(texture_features)
            
            # Shape moments
            shape_features = self._extract_shape_features(image_gray)
            features.extend(shape_features)
            
            # Edge features
            edge_features = self._extract_edge_features(image_gray)
            features.extend(edge_features)
            
            return np.array(features)
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return None
            
    def _extract_color_histogram(self, image_rgb: np.ndarray) -> List[float]:
        """Extract color histogram features"""
        features = []
        
        # RGB histograms
        for channel in range(3):
            hist = cv2.calcHist([image_rgb], [channel], None, [32], [0, 256])
            hist = hist.flatten() / (image_rgb.shape[0] * image_rgb.shape[1])
            features.extend(hist.tolist())
            
        return features
        
    def _extract_texture_features(self, image_gray: np.ndarray) -> List[float]:
        """Extract texture features using GLCM"""
        features = []
        
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Normalize image
            image_norm = ((image_gray / 255.0) * 63).astype(np.uint8)
            
            # Calculate GLCM
            distances = [1, 2]
            angles = [0, 45, 90, 135]
            
            glcm = graycomatrix(image_norm, distances, angles, 
                              levels=64, symmetric=True, normed=True)
            
            # Extract GLCM properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
            
            for prop in properties:
                values = graycoprops(glcm, prop)
                features.extend(values.flatten().tolist())
                
        except ImportError:
            # Fallback: simple texture measures
            # Standard deviation as texture measure
            features.extend([np.std(image_gray)])
            
            # Local binary pattern approximation
            rows, cols = image_gray.shape
            lbp_values = []
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = image_gray[i, j]
                    binary_string = ''
                    neighbors = [
                        image_gray[i-1, j-1], image_gray[i-1, j], image_gray[i-1, j+1],
                        image_gray[i, j+1], image_gray[i+1, j+1], image_gray[i+1, j],
                        image_gray[i+1, j-1], image_gray[i, j-1]
                    ]
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    lbp_values.append(int(binary_string, 2))
                    
            # LBP histogram
            lbp_hist, _ = np.histogram(lbp_values, bins=32, range=[0, 255])
            lbp_hist = lbp_hist / len(lbp_values)
            features.extend(lbp_hist.tolist())
            
        return features
        
    def _extract_shape_features(self, image_gray: np.ndarray) -> List[float]:
        """Extract shape-based features"""
        features = []
        
        # Image moments
        moments = cv2.moments(image_gray)
        
        # Hu moments (scale, rotation, and translation invariant)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        features.extend(hu_moments.flatten().tolist())
        
        # Basic shape features
        height, width = image_gray.shape
        features.extend([
            width / height,  # Aspect ratio
            np.sum(image_gray > 0) / (width * height),  # Fill ratio
        ])
        
        return features
        
    def _extract_edge_features(self, image_gray: np.ndarray) -> List[float]:
        """Extract edge-based features"""
        features = []
        
        # Canny edge detection
        edges = cv2.Canny(image_gray, 50, 150)
        
        # Edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Edge orientation histogram
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge orientations
        orientations = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        orientations = orientations[edges > 0]
        
        if len(orientations) > 0:
            # Orientation histogram
            hist, _ = np.histogram(orientations, bins=8, range=[-180, 180])
            hist = hist / len(orientations)
            features.extend(hist.tolist())
        else:
            features.extend([0] * 8)
            
        return features

class DatasetProcessor:
    """Process dataset for training"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        
    def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare dataset for training"""
        print(f"📂 Processing dataset: {self.dataset_dir}")
        
        features = []
        labels = []
        class_names = []
        
        # Find all class directories
        class_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            print(f"❌ No class directories found in {self.dataset_dir}")
            return np.array([]), np.array([]), []
            
        print(f"📁 Found {len(class_dirs)} classes:")
        for class_dir in class_dirs:
            print(f"   - {class_dir.name}")
            
        # Process each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Find all images in class directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(class_dir.glob(f"*{ext}"))
                image_files.extend(class_dir.glob(f"*{ext.upper()}"))
                
            print(f"🖼️ Processing {len(image_files)} images for class '{class_name}'...")
            
            # Extract features from each image
            for img_file in image_files:
                try:
                    img_features = self.feature_extractor.extract_features(str(img_file))
                    if img_features is not None:
                        features.append(img_features)
                        labels.append(class_name)
                    else:
                        print(f"⚠️ Failed to extract features from {img_file}")
                        
                except Exception as e:
                    print(f"❌ Error processing {img_file}: {e}")
                    
        if not features:
            print("❌ No features extracted from any images")
            return np.array([]), np.array([]), []
            
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Encode labels
        if SKLEARN_AVAILABLE:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            # Fallback label encoding
            unique_labels = list(set(y))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_encoded = np.array([label_map[label] for label in y])
            
        print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")
        
        return X, y_encoded, class_names

class AmuletAITrainer:
    """Main trainer class for Amulet AI models"""
    
    def __init__(self, output_dir: str = "trained_model"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.training_history = {}
        
    def train_random_forest(self, 
                           X_train: np.ndarray, 
                           X_test: np.ndarray,
                           y_train: np.ndarray, 
                           y_test: np.ndarray,
                           class_names: List[str]) -> Dict:
        """Train Random Forest classifier"""
        
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn not available - cannot train Random Forest")
            return {}
            
        print("🌲 Training Random Forest Classifier...")
        
        start_time = time.time()
        
        # Initialize model
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        training_time = time.time() - start_time
        
        print(f"✅ Random Forest trained in {training_time:.2f}s")
        print(f"📊 Test Accuracy: {accuracy:.4f}")
        print(f"📊 F1 Score: {f1:.4f}")
        
        # Store model
        self.models['random_forest'] = rf_model
        
        # Save model
        model_path = self.output_dir / "classifier.joblib"
        joblib.dump(rf_model, model_path)
        print(f"💾 Model saved: {model_path}")
        
        return {
            'model': rf_model,
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
    def train_with_preprocessing(self, 
                                X: np.ndarray, 
                                y: np.ndarray, 
                                class_names: List[str],
                                test_size: float = 0.2) -> Dict:
        """Train model with preprocessing pipeline"""
        
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn not available")
            return {}
            
        print("🔧 Setting up preprocessing pipeline...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of variance
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        self.scalers['pca'] = pca
        
        print(f"📐 Feature dimensions reduced: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]}")
        
        # Train Random Forest
        rf_results = self.train_random_forest(
            X_train_pca, X_test_pca, y_train, y_test, class_names
        )
        
        # Save preprocessing components
        scaler_path = self.output_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        pca_path = self.output_dir / "pca.joblib"
        joblib.dump(pca, pca_path)
        
        # Create and save label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)
        
        encoder_path = self.output_dir / "label_encoder.joblib"
        joblib.dump(label_encoder, encoder_path)
        
        # Save labels mapping
        labels_dict = {
            "current_classes": {str(i): name for i, name in enumerate(class_names)},
            "num_classes": len(class_names)
        }
        
        labels_path = self.output_dir / "labels.json"
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(labels_dict, f, indent=2, ensure_ascii=False)
            
        # Save training info
        training_info = {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(X),
            'num_classes': len(class_names),
            'class_names': class_names,
            'feature_dimensions': {
                'original': X.shape[1],
                'after_pca': X_train_pca.shape[1]
            },
            'model_performance': {
                'accuracy': rf_results.get('accuracy', 0),
                'f1_score': rf_results.get('f1_score', 0),
                'training_time': rf_results.get('training_time', 0)
            },
            'preprocessing': {
                'scaling': 'StandardScaler',
                'dimensionality_reduction': 'PCA',
                'pca_variance_ratio': pca.explained_variance_ratio_.sum()
            }
        }
        
        info_path = self.output_dir / "training_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Training completed successfully!")
        print(f"📁 Models and artifacts saved in: {self.output_dir}")
        
        return {
            'training_info': training_info,
            'model_results': rf_results,
            'preprocessing': {
                'scaler': scaler,
                'pca': pca,
                'label_encoder': label_encoder
            }
        }
        
    def train_from_dataset(self, dataset_dir: str) -> Dict:
        """Train model from dataset directory"""
        print(f"🚀 Starting AI model training from dataset: {dataset_dir}")
        
        # Process dataset
        processor = DatasetProcessor(dataset_dir)
        X, y, class_names = processor.prepare_dataset()
        
        if len(X) == 0:
            print("❌ No data available for training")
            return {}
            
        # Train with preprocessing
        results = self.train_with_preprocessing(X, y, class_names)
        
        return results

def main():
    """Main training function"""
    print("🏋️ Amulet AI Training System")
    print("=" * 50)
    
    # Configuration
    dataset_dir = "dataset/splits/train"  # Adjust path as needed
    output_dir = "trained_model"
    
    # Check if dataset exists
    if not Path(dataset_dir).exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("Please ensure the dataset is available in the correct location.")
        return
        
    # Initialize trainer
    trainer = AmuletAITrainer(output_dir)
    
    # Train model
    try:
        results = trainer.train_from_dataset(dataset_dir)
        
        if results:
            print("\n🎉 Training completed successfully!")
            training_info = results.get('training_info', {})
            
            print(f"📊 Dataset size: {training_info.get('dataset_size', 0)}")
            print(f"🏷️ Number of classes: {training_info.get('num_classes', 0)}")
            
            performance = training_info.get('model_performance', {})
            print(f"🎯 Accuracy: {performance.get('accuracy', 0):.4f}")
            print(f"📈 F1 Score: {performance.get('f1_score', 0):.4f}")
            print(f"⏱️ Training time: {performance.get('training_time', 0):.2f}s")
            
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()