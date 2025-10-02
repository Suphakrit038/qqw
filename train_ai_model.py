#!/usr/bin/env python3
"""
🤖 Amulet AI Training System
ระบบเทรนโมเดล AI สำหรับจำแนกพระเครื่อง
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AmuletAITrainer:
    """คลาสหลักสำหรับเทรนโมเดล AI"""
    
    def __init__(self, dataset_path="dataset", output_path="trained_model"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.model = None
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = []
        
        print("🚀 Amulet AI Trainer initialized")
        print(f"📁 Dataset path: {self.dataset_path}")
        print(f"💾 Output path: {self.output_path}")
    
    def extract_image_features(self, image_path, target_size=(224, 224)):
        """สกัดฟีเจอร์จากรูปภาพ"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Extract basic features
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
            
            # Edge density
            from scipy import ndimage
            edges = ndimage.sobel(gray)
            edge_density = np.mean(edges > np.mean(edges))
            features.append(edge_density)
            
            # Local binary pattern (simplified)
            lbp_features = self.compute_lbp_features(gray)
            features.extend(lbp_features)
            
            return np.array(features)
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return None
    
    def compute_lbp_features(self, gray_image, radius=1, n_points=8):
        """คำนวณ Local Binary Pattern features"""
        height, width = gray_image.shape
        lbp_features = []
        
        # Simplified LBP computation
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = gray_image[i, j]
                binary_string = ''
                
                # 8-neighborhood
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp_features.append(int(binary_string, 2))
        
        # Create histogram of LBP values
        hist, _ = np.histogram(lbp_features, bins=16, range=(0, 256))
        return hist.tolist()
    
    def load_dataset(self):
        """โหลด dataset จากโฟลเดอร์"""
        print("📂 Loading dataset...")
        
        # Check different possible dataset structures
        possible_paths = [
            self.dataset_path / "splits" / "train",
            self.dataset_path / "processed",
            self.dataset_path / "raw" / "main_dataset",
            self.dataset_path
        ]
        
        train_path = None
        for path in possible_paths:
            if path.exists() and any(path.iterdir()):
                train_path = path
                break
        
        if train_path is None:
            raise FileNotFoundError(f"ไม่พบ dataset ใน {possible_paths}")
        
        print(f"📁 Using dataset from: {train_path}")
        
        features = []
        labels = []
        self.class_names = []
        
        # Scan for class folders
        class_folders = [f for f in train_path.iterdir() if f.is_dir()]
        
        if not class_folders:
            raise ValueError(f"ไม่พบโฟลเดอร์คลาสใน {train_path}")
        
        print(f"🏷️ Found {len(class_folders)} classes:")
        
        for class_folder in sorted(class_folders):
            class_name = class_folder.name
            self.class_names.append(class_name)
            print(f"   - {class_name}")
            
            # Find images in class folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            image_files = [f for f in class_folder.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            print(f"     📸 Processing {len(image_files)} images...")
            
            for img_file in image_files:
                feature_vector = self.extract_image_features(img_file)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(class_name)
        
        if not features:
            raise ValueError("ไม่พบรูปภาพที่สามารถประมวลผลได้")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   📊 Total samples: {len(X)}")
        print(f"   🎯 Features per sample: {X.shape[1]}")
        print(f"   🏷️ Classes: {len(set(y))}")
        
        return X, y
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """เตรียมข้อมูลสำหรับการเทรน"""
        print("🔄 Preparing data...")
        
        # Load dataset
        X, y = self.load_dataset()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Apply PCA
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        print(f"✅ Data prepared!")
        print(f"   🎯 Training samples: {len(self.X_train)}")
        print(f"   🧪 Testing samples: {len(self.X_test)}")
        print(f"   📐 PCA components: {self.pca.n_components_}")
        print(f"   📊 Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
    
    def train_model(self):
        """เทรนโมเดล AI"""
        print("🤖 Training AI model...")
        
        # Define multiple models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"   🔄 Training {name}...")
            
            # Cross validation
            cv_scores = cross_val_score(model, self.X_train_pca, self.y_train, cv=5)
            mean_score = np.mean(cv_scores)
            
            print(f"      📊 CV Score: {mean_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        
        # Train best model on full training data
        print(f"🏆 Best model: {best_name} (CV Score: {best_score:.3f})")
        print("   🔄 Training final model...")
        
        self.model = best_model
        self.model.fit(self.X_train_pca, self.y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(self.X_test_pca)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"   🎯 Test Accuracy: {test_accuracy:.3f}")
        
        return test_accuracy
    
    def evaluate_model(self):
        """ประเมินผลโมเดล"""
        print("📊 Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test_pca)
        y_pred_proba = self.model.predict_proba(self.X_test_pca)
        
        # Classification report
        report = classification_report(
            self.y_test, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("📈 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Save evaluation plots
        self.save_evaluation_plots(cm, report)
        
        return report
    
    def save_evaluation_plots(self, confusion_matrix, report):
        """บันทึกกราฟการประเมินผล"""
        print("📊 Saving evaluation plots...")
        
        try:
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Accuracy by class
            plt.figure(figsize=(12, 6))
            classes = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
            f1_scores = [report[cls]['f1-score'] for cls in classes]
            
            plt.bar(classes, f1_scores, color='skyblue', edgecolor='navy')
            plt.title('F1-Score by Class')
            plt.xlabel('Class')
            plt.ylabel('F1-Score')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(self.output_path / 'f1_scores.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Evaluation plots saved!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save plots: {e}")
    
    def save_model(self):
        """บันทึกโมเดลและ components"""
        print("💾 Saving model...")
        
        # Save model components
        model_files = {
            'classifier.joblib': self.model,
            'scaler.joblib': self.scaler,
            'label_encoder.joblib': self.label_encoder,
            'pca.joblib': self.pca
        }
        
        for filename, component in model_files.items():
            joblib.dump(component, self.output_path / filename)
            print(f"   ✅ Saved {filename}")
        
        # Save class mapping
        class_mapping = {
            'current_classes': {str(i): str(name) for i, name in enumerate(self.class_names)},
            'num_classes': int(len(self.class_names)),
            'feature_dim': int(self.pca.n_components_),
            'training_date': datetime.now().isoformat()
        }
        
        with open(self.output_path / 'labels.json', 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Saved labels.json")
        
        # Save training info
        training_info = {
            'model_type': type(self.model).__name__,
            'n_samples_train': int(len(self.X_train)),
            'n_samples_test': int(len(self.X_test)),
            'n_features_original': int(self.X_train.shape[1]),
            'n_features_pca': int(self.pca.n_components_),
            'explained_variance_ratio': float(self.pca.explained_variance_ratio_.sum()),
            'class_names': [str(name) for name in self.class_names],
            'training_completed': datetime.now().isoformat()
        }
        
        with open(self.output_path / 'training_info.json', 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Saved training_info.json")
        print(f"🎉 Model saved successfully to {self.output_path}")
    
    def train_complete_pipeline(self):
        """เทรนโมเดลแบบครบวงจร"""
        print("🚀 Starting complete AI training pipeline...")
        print("=" * 60)
        
        try:
            # Step 1: Prepare data
            self.prepare_data()
            print("=" * 60)
            
            # Step 2: Train model
            accuracy = self.train_model()
            print("=" * 60)
            
            # Step 3: Evaluate model
            report = self.evaluate_model()
            print("=" * 60)
            
            # Step 4: Save model
            self.save_model()
            print("=" * 60)
            
            print("🎉 Training completed successfully!")
            print(f"📊 Final Test Accuracy: {accuracy:.3f}")
            print(f"💾 Model saved to: {self.output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """ฟังก์ชันหลักสำหรับเทรนโมเดล"""
    print("🤖 Amulet AI Training System")
    print("=" * 60)
    
    # Initialize trainer
    trainer = AmuletAITrainer(
        dataset_path="dataset",
        output_path="trained_model"
    )
    
    # Start training
    success = trainer.train_complete_pipeline()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📋 Next steps:")
        print("   1. Check the 'trained_model' folder for saved files")
        print("   2. Run the frontend to test the model")
        print("   3. Use the API for predictions")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()