#!/usr/bin/env python3
"""
Updated Model Loader for trained_model directory
โหลดโมเดลใหม่จาก trained_model
"""

import joblib
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class UpdatedAmuletClassifier:
    """อัปเดตเวอร์ชัน classifier ที่ใช้โมเดลใหม่"""
    
    def __init__(self, model_path: str = "trained_model"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.pca = None  # เพิ่ม PCA
        self.class_mapping = None
        self.model_info = None
        self.image_size = (224, 224)
        
    def load_model(self, model_path: Optional[str] = None):
        """โหลดโมเดลและ artifacts ทั้งหมด"""
        if model_path:
            self.model_path = Path(model_path)
            
        print(f"Loading model from: {self.model_path}")
        
        # โหลด model files
        try:
            # โหลด classifier
            classifier_path = self.model_path / "classifier.joblib"
            if classifier_path.exists():
                self.model = joblib.load(classifier_path)
                print("✅ Loaded classifier.joblib")
            else:
                raise FileNotFoundError(f"Classifier not found: {classifier_path}")
            
            # โหลด scaler
            scaler_path = self.model_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("✅ Loaded scaler.joblib")
            else:
                print("⚠️ Scaler not found, will use raw features")
            
            # โหลด label encoder
            encoder_path = self.model_path / "label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                print("✅ Loaded label_encoder.joblib")
            else:
                print("⚠️ Label encoder not found")
            
            # โหลด PCA
            pca_path = self.model_path / "pca.joblib"
            if pca_path.exists():
                self.pca = joblib.load(pca_path)
                print("✅ Loaded pca.joblib")
            else:
                print("⚠️ PCA not found")
            
            # โหลด model info
            info_path = self.model_path / "training_info.json"
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                # โหลด class mapping จาก labels.json
                labels_path = self.model_path / "labels.json"
                if labels_path.exists():
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        labels_data = json.load(f)
                    self.class_mapping = labels_data.get('current_classes', {})
                    # แปลง key จาก string index เป็น class name mapping
                    if self.class_mapping:
                        self.class_mapping = {v: int(k) for k, v in self.class_mapping.items()}
                self.image_size = tuple(self.model_info.get('image_size', [224, 224]))
                print("✅ Loaded training_info.json and labels.json")
            else:
                print("⚠️ Model info not found")
            
            print(f"🎯 Model loaded successfully!")
            if self.model_info:
                print(f"📊 Accuracy: {self.model_info.get('accuracy', 'Unknown')}")
                print(f"📁 Classes: {len(self.class_mapping) if self.class_mapping else 'Unknown'}")
            else:
                print(f"📁 Classes: {len(self.class_mapping) if self.class_mapping else 'Unknown'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """สกัดฟีเจอร์จากรูปภาพ (ตรงกับระบบเทรนนิ่ง)"""
        try:
            if not isinstance(image, np.ndarray):
                return None
            
            # แปลงเป็น RGB ถ้าจำเป็น
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.uint8 and np.max(image) > 1:
                    # ถ้าเป็น BGR (OpenCV) แปลงเป็น RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # ปรับขนาดรูปภาพ
            image_resized = cv2.resize(image, self.image_size)
            img_array = image_resized.astype(np.uint8)
            
            # Extract basic features (ตรงกับระบบเทรน)
            features = []
            
            # Color histogram features  
            for channel in range(3):  # RGB
                hist, _ = np.histogram(img_array[:,:,channel], bins=32, range=(0, 256))
                features.extend(hist)
            
            # Statistical features
            features.extend([
                np.mean(img_array),
                np.std(img_array),
                np.var(img_array),
                np.min(img_array),
                np.max(img_array)
            ])
            
            # Shape and texture features
            gray = np.mean(img_array, axis=2)
            
            # Edge density (simplified)
            # ใช้ Sobel edge detection
            sobel_x = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_density = np.mean(edges > np.mean(edges))
            features.append(edge_density)
            
            # Local binary pattern (simplified)
            lbp_features = self.compute_lbp_features(gray)
            features.extend(lbp_features)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def compute_lbp_features(self, gray_image, radius=1, n_points=8):
        """คำนวณ Local Binary Pattern features (ตรงกับระบบเทรน)"""
        try:
            height, width = gray_image.shape
            lbp_features = []
            
            # LBP computation (ทำทุกจุดแต่ลดขนาดรูปก่อน)
            # ลดขนาดรูปเพื่อลดเวลาการประมวลผล
            small_gray = cv2.resize(gray_image, (56, 56))  # ลดขนาด
            h, w = small_gray.shape
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = small_gray[i, j]
                    binary_string = ''
                    
                    # 8-neighborhood
                    neighbors = [
                        small_gray[i-1, j-1], small_gray[i-1, j], small_gray[i-1, j+1],
                        small_gray[i, j+1], small_gray[i+1, j+1], small_gray[i+1, j],
                        small_gray[i+1, j-1], small_gray[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp_features.append(int(binary_string, 2))
            
            # Create histogram of LBP values (ตรงกับระบบเทรน)
            if lbp_features:
                hist, _ = np.histogram(lbp_features, bins=16, range=(0, 256))
                return hist.tolist()
            else:
                return [0] * 16
                
        except Exception as e:
            print(f"Error computing LBP features: {e}")
            return [0] * 16
    
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """เตรียมรูปภาพสำหรับการทำนาย (wrapper สำหรับ extract_features)"""
        return self.extract_features(image)
    
    def predict(self, image: np.ndarray) -> Dict:
        """ทำนายผลจากรูปภาพ"""
        if self.model is None:
            return {
                "success": False,
                "error": "Model not loaded",
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {}
            }
        
        try:
            # สกัดฟีเจอร์จากรูปภาพ
            features = self.extract_features(image)
            if features is None:
                return {
                    "success": False,
                    "error": "Failed to extract features from image",
                    "predicted_class": None,
                    "confidence": 0.0,
                    "probabilities": {}
                }
            
            # ใช้ scaler ถ้ามี
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # ใช้ PCA ถ้ามี
            if self.pca is not None:
                features = self.pca.transform(features)
            
            # ทำนาย
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # แปลงผลลัพธ์
            if self.label_encoder is not None:
                # ใช้ label encoder ถ้ามี
                predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            elif self.class_mapping:
                # ใช้ class mapping
                class_names = list(self.class_mapping.keys())
                predicted_class = class_names[prediction] if prediction < len(class_names) else f"Unknown_{prediction}"
            else:
                predicted_class = f"Class_{prediction}"
            
            confidence = float(np.max(probabilities))
            
            # สร้าง probability dictionary
            if self.label_encoder is not None:
                class_names = self.label_encoder.classes_
            elif self.class_mapping:
                class_names = list(self.class_mapping.keys())
            else:
                class_names = [f"Class_{i}" for i in range(len(probabilities))]
            
            prob_dict = {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
                if i < len(class_names)
            }
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_index": int(prediction),
                "probabilities": prob_dict,
                "feature_count": features.shape[1],
                "model_info": {
                    "version": self.model_info.get('version', '2.0') if self.model_info else '2.0',
                    "accuracy": self.model_info.get('accuracy', 0.0) if self.model_info else 0.0,
                    "training_date": self.model_info.get('training_date', 'Unknown') if self.model_info else 'Unknown'
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {}
            }
    
    def predict_from_file(self, image_path: str) -> Dict:
        """ทำนายจากไฟล์รูปภาพ"""
        try:
            # โหลดรูปภาพ
            image = cv2.imread(image_path)
            if image is None:
                # ลองใช้ PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
            
            return self.predict(image)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load image: {str(e)}",
                "predicted_class": None,
                "confidence": 0.0,
                "probabilities": {}
            }
    
    def get_model_status(self) -> Dict:
        """ได้สถานะของโมเดล"""
        return {
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "label_encoder_loaded": self.label_encoder is not None,
            "pca_loaded": self.pca is not None,
            "model_path": str(self.model_path),
            "num_classes": len(self.class_mapping) if self.class_mapping else 0,
            "class_names": list(self.class_mapping.keys()) if self.class_mapping else [],
            "image_size": self.image_size,
            "model_info": self.model_info
        }

# Global classifier instance
updated_classifier = None

def get_updated_classifier() -> UpdatedAmuletClassifier:
    """ได้ classifier instance (singleton pattern)"""
    global updated_classifier
    if updated_classifier is None:
        updated_classifier = UpdatedAmuletClassifier()
        # ลองโหลดโมเดลอัตโนมัติ
        try:
            updated_classifier.load_model()
            print("🎯 Auto-loaded trained model successfully!")
        except Exception as e:
            print(f"⚠️ Could not auto-load model: {e}")
    return updated_classifier

def predict_image_file(image_path: str) -> Dict:
    """ฟังก์ชันสำหรับทำนายจากไฟล์รูปภาพ (สำหรับ frontend)"""
    try:
        classifier = get_updated_classifier()
        return classifier.predict_from_file(image_path)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to predict image: {str(e)}",
            "predicted_class": None,
            "confidence": 0.0,
            "probabilities": {}
        }

def predict_image_array(image_array: np.ndarray) -> Dict:
    """ฟังก์ชันสำหรับทำนายจาก numpy array (สำหรับ frontend)"""
    try:
        classifier = get_updated_classifier()
        return classifier.predict(image_array)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to predict image: {str(e)}",
            "predicted_class": None,
            "confidence": 0.0,
            "probabilities": {}
        }

def check_model_availability() -> Dict:
    """ตรวจสอบว่าโมเดลพร้อมใช้งานหรือไม่"""
    try:
        classifier = get_updated_classifier()
        status = classifier.get_model_status()
        return {
            "available": status["model_loaded"],
            "status": status,
            "message": "Model ready for predictions" if status["model_loaded"] else "Model not loaded"
        }
    except Exception as e:
        return {
            "available": False,
            "status": None,
            "message": f"Error checking model: {str(e)}"
        }

def test_model():
    """ทดสอบโมเดล"""
    print("=== Testing Updated Amulet Classifier ===")
    
    classifier = UpdatedAmuletClassifier()
    success = classifier.load_model()
    
    if success:
        status = classifier.get_model_status()
        print(f"✅ Model Status:")
        print(f"   📁 Path: {status['model_path']}")
        print(f"   🏷️ Classes: {status['num_classes']}")
        print(f"   📋 Class names: {status['class_names']}")
        print(f"   📐 Image size: {status['image_size']}")
        
        # สร้างรูปภาพทดสอบ
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"\n🧪 Testing with random image shape: {test_image.shape}")
        
        result = classifier.predict(test_image)
        if result['success']:
            print(f"✅ Test Prediction:")
            print(f"   🔮 Predicted: {result['predicted_class']}")
            print(f"   📊 Confidence: {result['confidence']:.4f}")
            print(f"   🎯 Features extracted: {result.get('feature_count', 'Unknown')}")
            print(f"   📈 Top 3 probabilities:")
            
            # แสดง top 3 probabilities
            sorted_probs = sorted(result['probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for class_name, prob in sorted_probs:
                print(f"      {class_name}: {prob:.4f}")
        else:
            print(f"❌ Test Prediction Failed: {result['error']}")
    else:
        print("❌ Failed to load model")
        print("💡 Make sure you have trained a model first using train_ai_model.py")

if __name__ == "__main__":
    test_model()