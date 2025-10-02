#!/usr/bin/env python3
"""
API module (relocated from backend.api.main_api)
"""
# ...existing code from previous file with minimal path adjustments...
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Query, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import logging
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
import asyncio
from pydantic import BaseModel
import json
from pathlib import Path
import sys

# Adjust sys.path to project root
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration and security modules
from core.config import config
from core.security import security as security_manager, validator
from core.rate_limiter import rate_limiter, get_client_id, apply_rate_limit
from core.error_handling import (
    error_logger, retry_on_failure, error_context, safe_execute,
    ModelError, ProcessingError, NetworkError, ValidationError
)
from core.performance import (
    image_cache, connection_pool, performance_monitor, timed_operation
)
from core.memory_management import (
    memory_monitor, streaming_handler, gc_manager, memory_efficient_operation,
    memory_limit_context
)
from core.thread_safety import (
    global_stats, lock_manager, thread_pool_manager, thread_safe_operation,
    ThreadSafeDict, AtomicCounter
)

try:
    from ai_models.updated_classifier import UpdatedAmuletClassifier, get_updated_classifier
except ImportError as e:
    print(f"Failed to import model: {e}")
    sys.exit(1)

try:
    from ai_models.twobranch.inference import TwoBranchInference
    _has_twobranch = True
except Exception:
    TwoBranchInference = None  # type: ignore
    _has_twobranch = False

# Configure logging using centralized configuration
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    images: List[str]
    request_id: Optional[str] = None

class PredictionResponse(BaseModel):
    status: str
    is_supported: bool
    predicted_class: Optional[str] = None
    thai_name: Optional[str] = None
    confidence: Optional[float] = None
    detailed_results: Optional[List[Dict]] = None
    explanations: Optional[Dict] = None
    performance: Dict[str, Any]
    request_id: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    sla_compliance: Dict[str, Any]
    timestamp: str

class MetricsResponse(BaseModel):
    requests_total: int
    requests_per_second: float
    error_rate: float
    latency_percentiles: Dict[str, float]
    memory_usage_mb: float
    cache_performance: Dict[str, Any]
    uptime_minutes: float

# Thread-safe global variables
classifier = None
twobranch_infer = None
app_start_time = time.time()
request_counter = AtomicCounter()
error_counter = AtomicCounter()
request_times = ThreadSafeDict()
security_bearer = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_bearer)):
    """Verify JWT token using security manager"""
    if not credentials:
        if config.REQUIRE_AUTH:
            raise HTTPException(status_code=401, detail="Authorization token required")
        return None
    
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    
    if payload is None and config.REQUIRE_AUTH:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload

def rate_limit_check(request: Request):
    """Apply rate limiting using centralized rate limiter"""
    client_id = get_client_id(request)
    
    # Check rate limit
    is_allowed, rate_info = apply_rate_limit(client_id, 'classify')
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {rate_info['retry_after']} seconds.",
            headers={
                "X-RateLimit-Limit": str(rate_info['limit']),
                "X-RateLimit-Remaining": str(rate_info['remaining']),
                "X-RateLimit-Reset": str(rate_info['reset']),
                "Retry-After": str(rate_info['retry_after'])
            }
        )
    
    return rate_info

@retry_on_failure(max_retries=3, exceptions=(Exception,))
@thread_safe_operation("model_loading")
def load_classifier():
    """Load updated classifier with new trained model"""
    global classifier
    
    with error_context("model_loading"):
        try:
            # ใช้ classifier ใหม่
            classifier = get_updated_classifier()
            
            if classifier.model is None:
                raise ModelError("Failed to load updated classifier model")
                
            error_logger.logger.info(f"✅ Updated classifier loaded successfully")
            error_logger.logger.info(f"📊 Model accuracy: {classifier.model_info.get('accuracy', 0):.4f}")
            error_logger.logger.info(f"🏷️ Classes: {len(classifier.class_mapping)}")
            
        except Exception as e:
            error_logger.log_error(e, context={"model_type": "updated_classifier"})
            raise ModelError(f"Failed to load updated classifier: {str(e)}")

app = FastAPI(
    title="Amulet-AI API",
    version="4.0.0",
    description="Secure Amulet Classification API"
)

# Configure CORS with security settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)
thresholds_router = APIRouter()

@thresholds_router.get("/model/thresholds")
def get_model_thresholds():
    import os, json as _json, numpy as _np
    tpath = os.getenv("THRESHOLDS_PATH", "trained_twobranch/thresholds.json")
    cpath = os.getenv("CENTROIDS_PATH", "trained_twobranch/centroids.npy")
    resp: Dict[str, Any] = {}
    if Path(tpath).exists():
        try:
            with open(tpath, 'r', encoding='utf-8') as f:
                resp['thresholds'] = _json.load(f)
        except Exception as e:
            resp['thresholds'] = {"error": str(e)}
    else:
        resp['thresholds'] = None
    if Path(cpath).exists():
        try:
            cent = _np.load(cpath)
            resp['centroids_shape'] = list(cent.shape)
        except Exception as e:
            resp['centroids_shape'] = f"error:{e}"
    else:
        resp['centroids_shape'] = None
    return resp

app.include_router(thresholds_router)

@app.on_event("startup")
async def startup_event():
    """Initialize application with enhanced error handling and monitoring"""
    try:
        with error_context("application_startup"):
            # Load models
            load_classifier()
            
            # Initialize monitoring
            memory_monitor.log_usage("startup")
            performance_monitor.record_counter("startup")
            
            error_logger.logger.info("API startup completed successfully")
            
    except Exception as e:
        error_logger.log_error(e, context={"operation": "startup"})
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Close connection pool
        await connection_pool.close()
        
        # Cleanup thread resources
        from core.thread_safety import cleanup_thread_resources
        cleanup_thread_resources()
        
        # Perform final garbage collection
        gc_manager.perform_cleanup(force=True)
        
        error_logger.logger.info("API shutdown completed")
        
    except Exception as e:
        error_logger.log_error(e, context={"operation": "shutdown"})

@memory_efficient_operation("image_processing")
@timed_operation("image_processing")
def process_uploaded_image(image_data: bytes, content_type: str = None) -> np.ndarray:
    """Process uploaded image with security validation and memory management"""
    try:
        # Validate file size
        if not validator.validate_file_size(len(image_data)):
            raise ValidationError("Image file too large")
        
        # Validate image type if content type provided
        if content_type and not validator.validate_image_type(content_type, image_data):
            raise ValidationError("Invalid image type")
        
        # Check memory pressure before processing
        pressure = memory_monitor.check_memory_pressure()
        if pressure["level"] == "critical":
            gc_manager.perform_cleanup(force=True)
        
        # Use streaming handler for large images
        if len(image_data) > config.LARGE_IMAGE_THRESHOLD:
            processed_data = streaming_handler.process_large_image(image_data)
            image_data = processed_data
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        
        # Validate image dimensions
        if not validator.validate_image_dimensions(image.width, image.height):
            raise ValidationError("Image dimensions out of allowed range")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
        
    except (ValidationError, ProcessingError):
        raise
    except Exception as e:
        error_logger.log_error(e, context={"image_size": len(image_data)})
        raise ProcessingError(f"Invalid image format: {str(e)}")

def validate_image_pair(front_image: np.ndarray, back_image: np.ndarray) -> bool:
    """Validate image pair with configuration-based limits"""
    if front_image.shape[:2] != back_image.shape[:2]:
        logger.warning("Image dimensions mismatch - will resize")
    
    # Use validator for dimension checks
    if not validator.validate_image_dimensions(front_image.shape[1], front_image.shape[0]):
        raise HTTPException(status_code=400, detail="Front image dimensions out of allowed range")
    
    if not validator.validate_image_dimensions(back_image.shape[1], back_image.shape[0]):
        raise HTTPException(status_code=400, detail="Back image dimensions out of allowed range")
    
    return True

@app.get("/")
async def root():
    return {"service": "Amulet-AI API", "version": "4.0.0"}

@app.post("/predict")
@timed_operation("prediction")
async def predict_amulet(
    request: Request,
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
    model: str = Query("enhanced"),
    preprocess: str = Query("standard"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    rate_info: dict = Depends(rate_limit_check),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Predict amulet classification with enhanced security and performance"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Increment request counter atomically
    request_count = request_counter.increment()
    
    try:
        with error_context("prediction_request", request_id=request_id):
            # Validate model is loaded
            if classifier is None:
                error_counter.increment()
                raise ModelError("Model not loaded")
            
            # Validate API parameters
            api_params = {'model': model, 'preprocess': preprocess}
            param_errors = validator.validate_api_params(api_params)
            if param_errors:
                error_counter.increment()
                raise ValidationError(f"Invalid parameters: {param_errors}")
            
            # Validate image types
            if (front_image.content_type not in config.ALLOWED_IMAGE_TYPES or
                back_image.content_type not in config.ALLOWED_IMAGE_TYPES):
                error_counter.increment()
                raise ValidationError("Unsupported image format")
            
            # Read and validate images with memory monitoring
            with memory_limit_context(200):  # 200MB limit for image processing
                front_data = await front_image.read()
                back_data = await back_image.read()
                
                # Check if images are cached
                front_cache_key = image_cache.cache_image(front_data, front_image.filename)
                back_cache_key = image_cache.cache_image(back_data, back_image.filename)
                
                front_np = process_uploaded_image(front_data, front_image.content_type)
                back_np = process_uploaded_image(back_data, back_image.content_type)
                
                validate_image_pair(front_np, back_np)
        
        # ใช้ classifier ใหม่สำหรับการทำนาย
        result = classifier.predict(front_np)  # ใช้รูปหน้าเป็นหลัก
        
        # เพิ่มข้อมูลเพิ่มเติม
        result.update({
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'model_version': '2.0',
            'processing_time': time.time() - start_time,
            'performance': {
                'memory_usage': memory_monitor.get_memory_usage(),
                'request_count': request_count
            }
        })
        
        # Create response with rate limiting headers
        response = JSONResponse(content=result)
        response.headers["X-RateLimit-Limit"] = str(rate_info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(rate_info['reset'])
        
        return response
        
    except (ValidationError, ModelError, ProcessingError) as e:
        error_counter.increment()
        client_id = get_client_id(request)
        error_logger.log_security_event("prediction_validation_error", {
            "client_id": client_id,
            "error": str(e),
            "request_id": request_id
        })
        
        if isinstance(e, ValidationError):
            raise HTTPException(status_code=400, detail=str(e))
        elif isinstance(e, ModelError):
            raise HTTPException(status_code=503, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail="Processing error")
    
    except Exception as e:
        error_counter.increment()
        error_logger.log_error(e, context={
            "request_id": request_id,
            "operation": "prediction",
            "client_id": get_client_id(request)
        })
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
@timed_operation("health_check")
async def health_check(request: Request, rate_info: dict = Depends(lambda req: apply_rate_limit(get_client_id(req), 'health'))):
    """Health check with comprehensive monitoring"""
    try:
        with error_context("health_check"):
            if classifier is None:
                return {"status": "unhealthy", "model_status": {"loaded": False}, "timestamp": datetime.now().isoformat()}
            
            # Get system health
            health_data = safe_execute(lambda: classifier.get_system_health(), {})
            
            # Add API metrics
            request_count = request_counter.get()
            error_count = error_counter.get()
            
            health_data['api_metrics'] = {
                'total_requests': request_count,
                'error_count': error_count,
                'error_rate': error_count / max(request_count, 1),
                'uptime_minutes': (time.time() - app_start_time) / 60
            }
            
            # Add memory and performance stats
            health_data['memory_stats'] = memory_monitor.get_stats()
            health_data['cache_stats'] = image_cache.get_stats()
            health_data['thread_stats'] = global_stats.get_stats()
            
            response = JSONResponse(content={"status": "healthy", **health_data})
            response.headers["X-RateLimit-Limit"] = str(rate_info[1]['limit'])
            response.headers["X-RateLimit-Remaining"] = str(rate_info[1]['remaining'])
            
            return response
            
    except Exception as e:
        error_logger.log_error(e, context={"operation": "health_check"})
        return {"status": "error", "error": str(e)}

@app.get("/metrics")
async def get_metrics(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token),
    rate_info: dict = Depends(lambda req: apply_rate_limit(get_client_id(req), 'default'))
):
    """Get API metrics - requires authentication"""
    try:
        uptime = (time.time() - app_start_time) / 60
        current_time = time.time()
        recent_requests = [t for t in request_times if current_time - t < 60]
        rps = len(recent_requests) / 60
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        latency_percentiles = {}
        cache_performance = {}
        
        if classifier:
            health = classifier.get_system_health()
            latency_percentiles = health.get('performance_metrics', {})
            cache_performance = health.get('cache_performance', {})
        
        # Get current counter values
        request_count = request_counter.get_value()
        error_count = error_counter.get_value()
        
        metrics_data = {
            'requests_total': request_count,
            'requests_per_second': rps,
            'error_rate': error_count / max(request_count, 1),
            'latency_percentiles': latency_percentiles,
            'memory_usage_mb': memory_mb,
            'cache_performance': cache_performance,
            'uptime_minutes': uptime
        }
        
        response = JSONResponse(content=metrics_data)
        response.headers["X-RateLimit-Limit"] = str(rate_info[1]['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_info[1]['remaining'])
        
        return response
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/model/info")
async def get_model_info():
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        health = classifier.get_system_health()
        return {"model_version": "4.0.0", "model_type": "Enhanced Production Classifier", "model_status": health['model_status']}
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")

async def log_request_metrics(request_id: str, processing_time: float, success: bool):
    try:
        metrics_data = {"request_id": request_id, "processing_time": processing_time, "success": success, "timestamp": datetime.now().isoformat(), "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024}
        logger.info(f"Request metrics: {json.dumps(metrics_data)}")
    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors with security logging"""
    client_id = get_client_id(request)
    error_logger.log_security_event("404_access_attempt", {
        "client_id": client_id,
        "path": str(request.url.path),
        "method": request.method
    })
    return JSONResponse(status_code=404, content={"error": "Endpoint not found"})

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors with comprehensive logging"""
    client_id = get_client_id(request)
    error_logger.log_error(exc, context={
        "client_id": client_id,
        "path": str(request.url.path),
        "method": request.method
    })
    return JSONResponse(
        status_code=500, 
        content={
            "error": "Internal server error", 
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api.main_api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        workers=1,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )
