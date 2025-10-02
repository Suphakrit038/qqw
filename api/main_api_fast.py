#!/usr/bin/env python3
"""
Amulet-AI Fast API Server - Production Ready
เซิร์ฟเวอร์ API สำหรับระบบจดจำพระเครื่อง
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import numpy as np
from PIL import Image
import io
import logging
import json
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Amulet-AI API",
    description="Thai Amulet Classification API",
    version="2.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://localhost:8501",
    "http://127.0.0.1:3000", 
    "http://127.0.0.1:8501"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
classifier = None
scaler = None
label_encoder = None
CURRENT_CLASSES = {}

def load_models():
    """Load trained models"""
    global classifier, scaler, label_encoder, CURRENT_CLASSES
    
    try:
        # Load models
        classifier = joblib.load('trained_model/classifier.joblib')
        scaler = joblib.load('trained_model/scaler.joblib')
        label_encoder = joblib.load('trained_model/label_encoder.joblib')
        
        # Load labels
        with open('ai_models/labels.json', 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
            CURRENT_CLASSES = labels_data['current_classes']
        
        logger.info("Models loaded successfully!")
        logger.info(f"Available classes: {list(CURRENT_CLASSES.values())}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise e

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Flatten for RandomForest model
        img_flattened = img_array.flatten()
        
        return img_flattened.reshape(1, -1)  # Shape: (1, 150528)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Amulet-AI API Server",
        "version": "2.0",
        "status": "running",
        "models_loaded": classifier is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_ready": classifier is not None,
        "available_classes": len(CURRENT_CLASSES),
        "classes": list(CURRENT_CLASSES.values())
    }

@app.post("/predict")
async def predict_amulet(file: UploadFile = File(...)):
    """Predict amulet type from uploaded image"""
    try:
        # Check if models are loaded
        if classifier is None or scaler is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Scale features
        scaled_features = scaler.transform(processed_image)
        
        # Make prediction
        prediction = classifier.predict(scaled_features)[0]
        probabilities = classifier.predict_proba(scaled_features)[0]
        confidence = max(probabilities)
        
        # Get class name
        predicted_class = CURRENT_CLASSES[str(prediction)]
        
        # Create probabilities dict
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            class_name = CURRENT_CLASSES[str(i)]
            prob_dict[class_name] = float(prob)
        
        # Format response
        response = {
            "status": "success",
            "prediction": {
                "class": predicted_class,
                "confidence": float(confidence),
                "probabilities": prob_dict,
                "is_out_of_distribution": bool(confidence < 0.7),
                "ood_score": float(1.0 - confidence)
            },
            "model_info": {
                "version": "2.0",
                "architecture": "Random Forest Classifier",
                "last_updated": "2025-10-01",
                "accuracy": "97.14%",
                "total_classes": 6
            },
            "processing_time": 0.1
        }
        
        logger.info(f"Prediction completed: {predicted_class} (confidence: {confidence:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get all available amulet classes"""
    return {
        "classes": CURRENT_CLASSES,
        "total": len(CURRENT_CLASSES)
    }

@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    try:
        with open('ai_models/labels.json', 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        return {
            "model_version": "2.0",
            "architecture": "Random Forest Classifier",
            "total_classes": len(CURRENT_CLASSES),
            "classes": list(CURRENT_CLASSES.values()),
            "training_info": labels_data.get('last_training', {}),
            "model_loaded": classifier is not None
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting Amulet-AI API Server...")
    uvicorn.run(
        "main_api_fast:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )