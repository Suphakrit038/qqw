#!/usr/bin/env python3
"""
ทดสอบการโหลดโมเดล
"""
import joblib
import os
from pathlib import Path

# กำหนด path
project_root = Path(__file__).parent
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")

# ตรวจสอบไฟล์โมเดล
model_files = [
    "trained_model/classifier.joblib",
    "trained_model/scaler.joblib", 
    "trained_model/pca.joblib",
    "trained_model/label_encoder.joblib"
]

print("\n=== ตรวจสอบไฟล์โมเดล ===")
for file_path in model_files:
    full_path = project_root / file_path
    exists = full_path.exists()
    print(f"{file_path}: {'✓' if exists else '✗'} ({full_path})")

print("\n=== ทดสอบโหลดโมเดล ===")
try:
    # ลองโหลดทีละไฟล์
    classifier_path = project_root / "trained_model/classifier.joblib"
    print(f"Loading classifier from: {classifier_path}")
    classifier = joblib.load(str(classifier_path))
    print("✓ Classifier loaded successfully")
    
    scaler_path = project_root / "trained_model/scaler.joblib"
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(str(scaler_path))
    print("✓ Scaler loaded successfully")
    
    pca_path = project_root / "trained_model/pca.joblib"
    print(f"Loading PCA from: {pca_path}")
    pca = joblib.load(str(pca_path))
    print("✓ PCA loaded successfully")
    
    label_encoder_path = project_root / "trained_model/label_encoder.joblib"
    print(f"Loading label encoder from: {label_encoder_path}")
    label_encoder = joblib.load(str(label_encoder_path))
    print("✓ Label encoder loaded successfully")
    
    print("\n=== ข้อมูลโมเดล ===")
    print(f"Classifier type: {type(classifier)}")
    print(f"Scaler type: {type(scaler)}")
    print(f"PCA components: {pca.n_components_}")
    print(f"Label classes: {label_encoder.classes_}")
    
except Exception as e:
    print(f"✗ Error loading models: {e}")
    import traceback
    traceback.print_exc()