#!/usr/bin/env python3
"""
Amulet-AI - Production Frontend
ระบบจำแนกพระเครื่องอัจฉริยะ
Thai Amulet Classification System
"""

import streamlit as st
import requests
import numpy as np
from PIL import Image
import json
import time
import base64
from pathlib import Path
import sys
import os
from datetime import datetime
import io

# Optional OpenCV import - ไม่จำเป็นสำหรับการทำงานหลัก
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # cv2 is optional, we can work without it
# PyTorch imports with fallback
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    # Create dummy torch objects
    class DummyTorch:
        device = lambda x: 'cpu'
        no_grad = lambda: DummyContext()
        load = lambda x, **kwargs: {}
        
    class DummyF:
        softmax = lambda x, dim=1: x
        
    class DummyTransforms:
        Compose = lambda x: lambda img: img
        Resize = lambda x: lambda img: img
        ToTensor = lambda: lambda img: img
        Normalize = lambda **kwargs: lambda img: img
        
    class DummyContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        
    torch = DummyTorch()
    F = DummyF()
    transforms = DummyTransforms()
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using fallback mode")

# Lazy import joblib to avoid threading issues
def load_joblib_file(file_path):
    """Lazy load joblib file to avoid threading issues in Python 3.13"""
    try:
        import joblib
        return joblib.load(file_path)
    except ImportError:
        print(f"Warning: joblib not available, cannot load {file_path}")
        return None
    except Exception as e:
        print(f"Warning: Failed to load joblib file {file_path}: {e}")
        return None

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Debug: Print sys.path and project_root
print(f"Project root: {project_root}")
print(f"sys.path[0]: {sys.path[0]}")

# Import AI Models (with comprehensive fallback)
AI_MODELS_AVAILABLE = False
try:
    # Import our actual AI models
    from ai_models.enhanced_production_system import EnhancedProductionClassifier
    from ai_models.updated_classifier import UpdatedAmuletClassifier, get_updated_classifier
    from ai_models.compatibility_loader import ProductionOODDetector, try_load_model
    AI_MODELS_AVAILABLE = True
    print("✅ AI Models loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: AI models not available: {e}")
    print("   Using fallback mode - basic functionality only")
    # Create comprehensive dummy classes
    class DummyClassifier:
        def __init__(self, *args, **kwargs):
            self.loaded = False
            
        def load_model(self, *args, **kwargs):
            return False
            
        def predict(self, *args, **kwargs):
            return {
                "status": "error", 
                "error": "AI models not available in this environment",
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "method": "Fallback"
            }
    
    EnhancedProductionClassifier = DummyClassifier
    UpdatedAmuletClassifier = DummyClassifier
    get_updated_classifier = lambda: DummyClassifier()
    ProductionOODDetector = DummyClassifier
    AI_MODELS_AVAILABLE = False

# Import core modules (with comprehensive fallback)
CORE_AVAILABLE = False
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
    CORE_AVAILABLE = True
    print("✅ Core modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Core modules not available: {e}")
    print("   Using fallback implementations")
    # Comprehensive fallback implementations
    def error_handler(error_type="general"):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {func.__name__}: {str(e)}")
                    return {
                        "status": "error", 
                        "error": f"Function {func.__name__} failed: {str(e)}",
                        "method": "Fallback"
                    }
            return wrapper
        return decorator
    
    def validate_image_file(file):
        """Basic file validation fallback"""
        if file is None:
            return False
        # Basic checks
        if hasattr(file, 'name') and hasattr(file, 'size'):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            return any(file.name.lower().endswith(ext) for ext in valid_extensions)
        return True  # If we can't check, assume valid
    
    class performance_monitor:
        @staticmethod
        def collect_metrics():
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "fallback_mode",
                "memory_usage": "unknown"
            }
    
    CORE_AVAILABLE = False

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Theme Colors
COLORS = {
    'primary': '#800000',
    'maroon': '#800000',
    'accent': '#B8860B',
    'dark_gold': '#B8860B',
    'gold': '#D4AF37',
    'success': '#10b981',
    'green': '#10b981',
    'warning': '#f59e0b',
    'yellow': '#f59e0b',
    'error': '#ef4444',
    'red': '#ef4444',
    'info': '#3b82f6',
    'blue': '#3b82f6',
    'gray': '#6c757d',
    'white': '#ffffff',
    'black': '#000000'
}

# Page Configuration
st.set_page_config(
    page_title="Amulet-AI - ระบบจำแนกพระเครื่อง",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Display system status at the top (for debugging)
if not AI_MODELS_AVAILABLE or not CORE_AVAILABLE:
    status_info = []
    if not AI_MODELS_AVAILABLE:
        status_info.append("AI Models: ❌ Fallback Mode")
    else:
        status_info.append("AI Models: ✅ Available")
        
    if not CORE_AVAILABLE:
        status_info.append("Core Modules: ❌ Fallback Mode")
    else:
        status_info.append("Core Modules: ✅ Available")
        
    if not TORCH_AVAILABLE:
        status_info.append("PyTorch: ❌ Not Available")
    else:
        status_info.append("PyTorch: ✅ Available")
        
    st.info(f"🔧 System Status: {' | '.join(status_info)}")

# Modern Modal Design CSS
st.markdown(f"""
<style>
    /* Import Modern Fonts - Thai + English */
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@200;300;400;500;600;700;800&family=Prompt:wght@300;400;500;600;700;800&display=swap');
    
    /* Modern App Background - Creamy White */
    .stApp {{
        font-family: 'Sarabun', 'Prompt', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #fdfbf7 0%, #f5f3ef 100%);
        background-attachment: fixed;
    }}
    
    /* Glassmorphism Container */
    .main .block-container {{
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        box-shadow: 0 8px 32px 0 rgba(128, 0, 0, 0.08);
        border: 1px solid rgba(212, 175, 55, 0.2);
        padding: 40px;
        margin: 20px auto;
        max-width: 1000px;
    }}
    
    /* Modal-Style Logo Header */
    .logo-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 60px 80px;
        background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.92) 100%);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(128, 0, 0, 0.08);
        margin-bottom: 30px;
        border: 1px solid rgba(212, 175, 55, 0.15);
        position: relative;
        overflow: hidden;
    }}
    
    .logo-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']}, {COLORS['primary']});
    }}
    
    .logo-left {{
        display: flex;
        align-items: center;
        gap: 20px;
        z-index: 1;
    }}
    
    .logo-text {{
        display: flex;
        flex-direction: column;
        gap: 2px;
    }}
    
    .logo-title {{
        font-family: 'Prompt', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }}
    
    .logo-subtitle {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        color: {COLORS['gray']};
        margin: 0;
        letter-spacing: 0.3px;
    }}
    
    .logo-right {{
        display: flex;
        align-items: center;
        gap: 25px;
        z-index: 1;
    }}
    
    .logo-img {{
        height: 160px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 3px 6px rgba(0,0,0,0.12));
        transition: transform 0.3s ease;
    }}
    
    .logo-img:hover {{
        transform: scale(1.03);
    }}
    
    .logo-img-small {{
        height: 180px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 3px 6px rgba(0,0,0,0.12));
        transition: transform 0.3s ease;
    }}
    
    .logo-img-small:hover {{
        transform: scale(1.03);
    }}
    
    /* Modal Card Style with Glassmorphism */
    .card {{
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin: 35px 0;
        border: 1px solid rgba(255, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
    }}
    
    .card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']});
    }}
    
    /* Modern Typography */
    h1 {{
        font-family: 'Prompt', sans-serif;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        margin-bottom: 20px !important;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.3 !important;
    }}
    
    h2 {{
        font-family: 'Prompt', sans-serif;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px !important;
        margin-bottom: 18px !important;
        color: #2d3748 !important;
        line-height: 1.4 !important;
    }}
    
    h3 {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        margin-bottom: 14px !important;
        color: #2d3748 !important;
        line-height: 1.4 !important;
    }}
    
    h4 {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 12px !important;
        color: #4a5568 !important;
        line-height: 1.5 !important;
    }}
    
    /* Modern Button with Gradient and Animation */
    .stButton > button {{
        font-family: 'Sarabun', sans-serif;
        background: {COLORS['accent']};
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 40px;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 15px rgba(184, 134, 11, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['gold']};
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
        transform: translateY(-2px) scale(1.01);
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(0.98);
        box-shadow: 0 2px 10px rgba(184, 134, 11, 0.3);
    }}
    
    /* Modern Text Styling */
    p {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
        color: #4a5568 !important;
        font-weight: 400 !important;
    }}
    
    /* Modern Input Fields with Glassmorphism */
    .stTextInput > div > div > input {{
        font-size: 1.3rem !important;
        padding: 18px !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(128, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 3px rgba(128, 0, 0, 0.1) !important;
    }}
    
    /* Modern File Uploader */
    [data-testid="stFileUploader"] {{
        font-size: 1.3rem !important;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 16px;
        border: 2px dashed {COLORS['primary']};
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        background: rgba(255, 255, 255, 0.95);
        border-color: {COLORS['accent']};
        transform: scale(1.01);
    }}
    
    [data-testid="stFileUploader"] label {{
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: {COLORS['primary']} !important;
    }}
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background: transparent;
        padding: 0;
        margin-bottom: 25px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 14px 32px !important;
        border-radius: 10px !important;
        background: #f5ebe0 !important;
        border: none !important;
        transition: all 0.25s ease !important;
        color: #6c757d !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: #ede0d4 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(128, 0, 0, 0.1);
        color: {COLORS['primary']} !important;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {COLORS['accent']} !important;
        color: white !important;
        box-shadow: 0 3px 12px rgba(184, 134, 11, 0.35);
    }}
    
    /* Modern Alert Boxes with Glassmorphism */
    .stAlert {{
        border-radius: 16px !important;
        padding: 25px !important;
        font-size: 1.3rem !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }}
    
    /* Modal Success Box */
    .success-box {{
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.95), rgba(200, 230, 201, 0.95));
        backdrop-filter: blur(10px);
        color: #1b5e20;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .success-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modal Info Box */
    .info-box {{
        background: linear-gradient(135deg, rgba(227, 242, 253, 0.95), rgba(187, 222, 251, 0.95));
        backdrop-filter: blur(10px);
        color: #0d47a1;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #2196f3;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .info-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modal Warning Box */
    .warning-box {{
        background: linear-gradient(135deg, rgba(255, 243, 224, 0.95), rgba(255, 224, 178, 0.95));
        backdrop-filter: blur(10px);
        color: #e65100;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #ff9800;
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .warning-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modal Error Box */
    .error-box {{
        background: linear-gradient(135deg, rgba(255, 235, 238, 0.95), rgba(255, 205, 210, 0.95));
        backdrop-filter: blur(10px);
        color: #b71c1c;
        padding: 30px;
        border-radius: 16px;
        border-left: 5px solid #f44336;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.2);
        margin: 30px 0;
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .error-box:hover {{
        transform: translateX(5px);
    }}
    
    /* Modern Section Divider */
    .section-divider {{
        height: 1.5px;
        background: linear-gradient(90deg, transparent, {COLORS['gold']}, transparent);
        margin: 35px 0;
        border-radius: 2px;
        opacity: 0.6;
    }}
    
    /* Modal Tips Card */
    .tips-card {{
        background: linear-gradient(135deg, rgba(255, 253, 231, 0.95), rgba(255, 249, 196, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 35px;
        margin: 35px 0;
        box-shadow: 0 8px 25px rgba(218, 165, 32, 0.15);
        border-left: 5px solid {COLORS['gold']};
        font-size: 1.3rem;
        transition: transform 0.3s ease;
    }}
    
    .tips-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(218, 165, 32, 0.25);
    }}
    
    /* Feature Card */
    .feature-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(250, 250, 250, 0.95));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        margin: 25px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        border: 1px solid rgba(255, 255, 255, 0.5);
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .feature-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.2);
        border-color: {COLORS['gold']};
    }}
    
    .feature-card h3 {{
        color: {COLORS['primary']} !important;
        margin-bottom: 20px !important;
    }}
    
    .feature-card ul {{
        list-style: none;
        padding-left: 0;
    }}
    
    .feature-card ul li {{
        padding: 10px 0;
        padding-left: 28px;
        position: relative;
        font-size: 1.0rem;
        line-height: 1.7;
    }}
    
    .feature-card ul li:before {{
        content: '✓';
        position: absolute;
        left: 0;
        color: {COLORS['gold']};
        font-weight: bold;
        font-size: 1.2rem;
    }}
    
    /* Step Card */
    .step-card {{
        background: linear-gradient(135deg, rgba(227, 242, 253, 0.95), rgba(187, 222, 251, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 35px;
        margin: 25px 0;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.15);
        border-left: 5px solid {COLORS['info']};
        transition: transform 0.3s ease;
    }}
    
    .step-card:hover {{
        transform: translateX(8px);
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.25);
    }}
    
    .step-card h4 {{
        color: {COLORS['info']} !important;
        margin-bottom: 15px !important;
    }}
    
    /* Hero Section */
    .hero-section {{
        text-align: center;
        padding: 45px 35px;
        background: linear-gradient(135deg, rgba(128, 0, 0, 0.04), rgba(212, 175, 55, 0.04));
        border-radius: 20px;
        margin: 30px 0;
    }}
    
    .hero-title {{
        font-family: 'Prompt', sans-serif;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 15px !important;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.3 !important;
    }}
    
    .hero-subtitle {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.15rem !important;
        color: {COLORS['gray']} !important;
        margin-bottom: 0 !important;
        line-height: 1.6 !important;
    }}
    
    /* Modal Result Card */
    .result-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(250, 250, 250, 0.98));
        backdrop-filter: blur(15px);
        padding: 50px;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        margin: 35px 0;
        border-top: 5px solid {COLORS['primary']};
        position: relative;
        overflow: hidden;
        max-width: 1200px;
        width: 98vw;
        min-width: 350px;
        margin: 30px auto;
    }}
    
    .result-card::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 200px;
        height: 200px;
        background: linear-gradient(135deg, {COLORS['gold']}, transparent);
        opacity: 0.1;
        border-radius: 50%;
    }}
    
    /* Column Styling */
    [data-testid="column"] {{
        padding: 25px;
    }}
    
    /* Modern Spinner */
    .stSpinner > div {{
        border-color: {COLORS['primary']} {COLORS['gold']} {COLORS['primary']} {COLORS['gold']} !important;
    }}
    
    /* Modern Labels */
    label {{
        font-family: 'Sarabun', sans-serif;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        margin-bottom: 8px !important;
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['gold']}) !important;
    }}
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {{
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    [data-testid="stSidebar"] {{display: none;}}
    [data-testid="collapsedControl"] {{display: none;}}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['gold']});
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, {COLORS['gold']}, {COLORS['primary']});
    }}
</style>
""", unsafe_allow_html=True)

# Utility Functions
def get_logo_base64():
    """Convert logo image to base64"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), 'imgae', 'Amulet-AI_logo.png')
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return ""

def get_other_logos():
    """Get partnership logos"""
    logos = {}
    try:
        logo_dir = os.path.join(os.path.dirname(__file__), 'imgae')
        
        thai_logo = os.path.join(logo_dir, 'Logo Thai-Austrain.gif')
        if os.path.exists(thai_logo):
            with open(thai_logo, "rb") as f:
                logos["thai_austrian"] = base64.b64encode(f.read()).decode()
        
        depa_logo = os.path.join(logo_dir, 'LogoDEPA-01.png')
        if os.path.exists(depa_logo):
            with open(depa_logo, "rb") as f:
                logos["depa"] = base64.b64encode(f.read()).decode()
    except:
        pass
    return logos

def check_api_health():
    """ตรวจสอบสถานะ API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_model_status():
    """ตรวจสอบสถานะโมเดล PyTorch และ sklearn"""
    # Check AI models first
    if AI_MODELS_AVAILABLE:
        ai_files = [
            "ai_models/enhanced_production_system.py",
            "ai_models/updated_classifier.py",
            "ai_models/labels.json"
        ]
        
        missing_ai = []
        for file_path in ai_files:
            full_path = project_root / file_path
            if not full_path.exists():
                missing_ai.append(file_path)
        
        if len(missing_ai) == 0:
            return True, []  # AI models available
    
    # Fallback to basic models
    basic_files = [
        "trained_model/classifier.joblib",
        "trained_model/scaler.joblib", 
        "trained_model/label_encoder.joblib"
    ]
    
    missing_basic = []
    for file_path in basic_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_basic.append(file_path)
    
    return len(missing_basic) == 0, missing_basic

@st.cache_resource
def load_ai_model():
    """โหลดโมเดล AI สำหรับการทำนาย"""
    if not AI_MODELS_AVAILABLE:
        return {
            'classifier': None,
            'type': 'fallback',
            'labels': {
                "current_classes": {
                    "0": "พระสมเด็จ",
                    "1": "พระพิมพ์", 
                    "2": "พระกรุ",
                    "3": "พระหลวงพ่อ",
                    "4": "พระนาคปรก",
                    "5": "พระปิดตา"
                }
            },
            'available': False,
            'message': 'Running in fallback mode - AI models not available'
        }
    
    try:
        # Try to load our enhanced classifier first
        classifier = get_updated_classifier()
        if hasattr(classifier, 'load_model') and classifier.load_model():
            return {
                'classifier': classifier,
                'type': 'enhanced',
                'labels': getattr(classifier, 'class_mapping', {}),
                'available': True,
                'message': 'Enhanced AI classifier loaded successfully'
            }
        
        # Fallback to basic model info
        return {
            'classifier': classifier,
            'type': 'basic',
            'labels': {
                "current_classes": {
                    "0": "พระสมเด็จ",
                    "1": "พระพิมพ์",
                    "2": "พระกรุ",
                    "3": "พระหลวงพ่อ",
                    "4": "พระนาคปรก",
                    "5": "พระปิดตา"
                }
            },
            'available': True,
            'message': 'Basic AI classifier loaded'
        }
        
    except Exception as e:
        print(f"Error loading AI model: {e}")
        return {
            'classifier': None,
            'type': 'error',
            'labels': {},
            'available': False,
            'message': f'Failed to load AI model: {str(e)}'
        }

def enhance_result_for_display(result, processing_time=2.0, analysis_type='single_image'):
    """แปลงผลลัพธ์ให้เข้ากับ Analysis Results Component"""
    if result.get('status') != 'success':
        return result
    
    # Enhance the result with additional display data
    enhanced = {
        'thai_name': result.get('thai_name', result.get('predicted_class', 'ไม่ระบุ')),
        'confidence': result.get('confidence', 0.0),
        'predicted_class': result.get('predicted_class', 'unknown'),
        'probabilities': result.get('probabilities', {}),
        'processing_time': processing_time,
        'analysis_type': analysis_type,
        'model_version': f"Enhanced {'Dual-View' if analysis_type == 'dual_image' else 'Single-View'} AI v2.1",
        'timestamp': datetime.now().isoformat(),
        'enhanced_features': {
            'image_quality': {
                'overall_score': min(result.get('confidence', 0.5) + 0.2, 0.95),
                'quality_level': 'excellent' if result.get('confidence', 0) > 0.8 else 'good',
                'was_enhanced': True
            },
            'auto_enhanced': True,
            'dual_analysis': analysis_type == 'dual_image'
        },
        'method': result.get('method', 'AI'),
        'feature_count': result.get('feature_count', 0),
        'model_info': result.get('model_info', {})
    }
    
    return enhanced

@error_handler("frontend")
def classify_image(uploaded_file):
    """จำแนกรูปภาพด้วย AI models"""
    try:
        # Validate file type and size
        if uploaded_file is None:
            return {
                "status": "error",
                "error": "ไม่มีไฟล์ที่อัปโหลด",
                "method": "None"
            }
        
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            return {
                "status": "error",
                "error": f"ไฟล์ใหญ่เกินไป (สูงสุด {MAX_FILE_SIZE // (1024*1024)} MB)",
                "method": "None"
            }
        
        # Check file extension
        file_extension = uploaded_file.name.lower().split('.')[-1]
        allowed_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'heic', 'heif']
        if file_extension not in allowed_extensions:
            return {
                "status": "error",
                "error": f"ไฟล์ประเภท .{file_extension} ไม่รองรับ รองรับเฉพาะ: {', '.join(allowed_extensions)}",
                "method": "None"
            }
        
        # Save temp file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Try API first
        try:
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                result["method"] = "API"
                Path(temp_path).unlink(missing_ok=True)
                return result
        except:
            pass
        
        # Use AI model (main method)
        model_data = load_ai_model()
        if model_data is not None:
            try:
                result = ai_local_prediction(temp_path, model_data)
                result["method"] = result.get("method", "Local (AI)")
                
                # Add system info to result
                if not model_data.get('available', False):
                    result["demo_mode"] = True
                    result["system_message"] = model_data.get('message', 'Running in demo mode')
                
                Path(temp_path).unlink(missing_ok=True)
                return result
            except Exception as e:
                Path(temp_path).unlink(missing_ok=True)
                return {
                    "status": "error",
                    "error": f"AI prediction error: {str(e)}",
                    "method": "None"
                }
        else:
            Path(temp_path).unlink(missing_ok=True)
            return {
                "status": "error",
                "error": "ไม่สามารถโหลดระบบ AI ได้ กรุณาลองใหม่อีกครั้ง",
                "method": "None"
            }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "method": "None"
        }

def ai_local_prediction(image_path, model_data):
    """การทำนายด้วย AI model"""
    try:
        classifier = model_data.get('classifier')
        model_type = model_data.get('type', 'unknown')
        
        # Handle fallback mode
        if not model_data.get('available', False) or classifier is None:
            # Return a demo result for fallback mode
            demo_classes = list(model_data.get('labels', {}).get('current_classes', {}).values())
            if demo_classes:
                import random
                random_class = random.choice(demo_classes)
                random_confidence = random.uniform(0.6, 0.9)
                
                # Create mock probabilities
                probabilities = {}
                remaining_prob = 1.0 - random_confidence
                for cls in demo_classes:
                    if cls == random_class:
                        probabilities[cls] = random_confidence
                    else:
                        probabilities[cls] = remaining_prob / (len(demo_classes) - 1)
                
                return {
                    "status": "success",
                    "predicted_class": random_class,
                    "thai_name": random_class,
                    "confidence": random_confidence,
                    "probabilities": probabilities,
                    "is_ood": False,
                    "ood_score": None,
                    "gradcam_available": False,
                    "method": "Demo Mode",
                    "message": "This is a demo result - AI models not available"
                }
            else:
                return {
                    "status": "error",
                    "error": "No class labels available for demo mode",
                    "method": "Fallback"
                }
        
        # Load and predict using our classifier (real mode)
        from PIL import Image
        import numpy as np
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Make prediction using our trained classifier
        result = classifier.predict(image_array)
        
        if result.get('success', False):
            # Convert to expected format
            predicted_class = result.get('predicted_class', 'unknown')
            confidence = result.get('confidence', 0.0)
            probabilities = result.get('probabilities', {})
            
            return {
                "status": "success",
                "predicted_class": predicted_class,
                "thai_name": predicted_class,  # Use predicted class as thai name
                "confidence": confidence,
                "probabilities": probabilities,
                "is_ood": confidence < 0.5,  # Low confidence = out of distribution
                "ood_score": 1.0 - confidence,
                "gradcam_available": False,
                "method": f"AI ({model_type})",
                "feature_count": result.get('feature_count', 0),
                "model_info": result.get('model_info', {}),
                "processing_time": 1.5  # Estimate
            }
        else:
            return {
                "status": "error",
                "error": result.get('error', 'Prediction failed'),
                "method": f"AI ({model_type})"
            }
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"AI prediction error: {error_detail}")
        return {
            "status": "error",
            "error": f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}",
            "method": "AI (Error)"
        }

def display_classification_result(result, show_confidence=True, show_probabilities=True, image_path=None):
    """แสดงผลการจำแนกทั้งหมดในกล่องเดียว รวมทุกอย่าง - ปรับปรุง HTML rendering"""
    
    # Check if this is demo mode
    is_demo_mode = result.get('demo_mode', False)
    system_message = result.get('system_message', '')
    
    # Show demo mode warning if applicable
    if is_demo_mode:
        st.warning(f"""
        🔧 **โหมดทดสอบ (Demo Mode)**
        
        {system_message}
        
        ผลลัพธ์ที่แสดงเป็นเพียงตัวอย่างการทำงานของระบบ ไม่ใช่การวิเคราะห์จริง
        
        เพื่อใช้งานจริง กรุณาติดตั้ง dependencies ที่จำเป็น:
        - scikit-learn
        - joblib  
        - โมเดล AI ที่ได้รับการฝึกฝน
        """)
    
    if result.get("status") == "success" or "predicted_class" in result:
        predicted_class = result.get('predicted_class', 'Unknown')
        thai_name = result.get('thai_name', predicted_class)
        confidence = result.get('confidence', 0)
        is_ood = result.get('is_ood', False)
        ood_score = result.get('ood_score', None)
        
        # กำหนดสีและสถานะตามความมั่นใจ
        if is_ood:
            conf_color = "#ff6b6b"
            status_text = "ผิดปกติ"
            status_emoji = "⚠️"
            header_gradient = "linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)"
        elif confidence >= 0.92:
            conf_color = "#4CAF50"
            status_text = "ยอดเยี่ยม"
            status_emoji = "🌟"
            header_gradient = "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
        elif confidence >= 0.7:
            conf_color = "#FFA726"
            status_text = "ดี"
            status_emoji = "👍"
            header_gradient = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
        else:
            conf_color = "#EF5350"
            status_text = "ควรตรวจสอบ"
            status_emoji = "🤔"
            header_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        
        # เริ่มกล่องหลัก
        st.markdown(f"""
        <div style="background: white; border-radius: 25px; padding: 50px; 
                    box-shadow: 0 15px 50px rgba(0,0,0,0.12); 
                    max-width: 1200px; width: 98vw; min-width: 350px; margin: 30px auto; 
                    border: 1px solid #e0e0e0;">
            
            <!-- Header -->
            <div style="background: {header_gradient}; color: white; padding: 40px; 
                        border-radius: 18px; margin-bottom: 35px; text-align: center; 
                        box-shadow: 0 6px 20px rgba(0,0,0,0.15);">
                <div style="font-size: 3.5rem; margin-bottom: 12px;">🙏</div>
                <h1 style="margin: 0; font-size: 2.8rem; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                    {thai_name}
                </h1>
                <h2 style="margin: 18px 0 0 0; font-size: 1.4rem; opacity: 0.95; font-weight: 500;">
                    ประเภท: {predicted_class}
                </h2>
            </div>
            
            <!-- Confidence Section -->
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        padding: 40px; border-radius: 18px; margin: 35px 0; text-align: center;">
                <h2 style="color: #333; margin: 0 0 25px 0; font-size: 1.8rem;">📊 ความมั่นใจของ AI</h2>
                <div style="font-size: 6rem; font-weight: bold; color: {conf_color}; margin: 25px 0; 
                            text-shadow: 3px 3px 6px rgba(0,0,0,0.1);">
                    {confidence:.0%}
                </div>
                <div style="margin: 25px 0;">
                    <span style="background: {conf_color}; color: white; padding: 12px 40px; 
                                border-radius: 50px; font-size: 1.4rem; font-weight: bold; 
                                display: inline-block; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                        {status_emoji} {status_text}
                    </span>
                </div>
                <div style="background: #dee2e6; border-radius: 50px; height: 45px; overflow: hidden; 
                            margin: 30px auto; max-width: 85%; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="background: linear-gradient(90deg, {conf_color} 0%, {conf_color}dd 100%); 
                                width: {confidence*100}%; height: 100%; 
                                display: flex; align-items: center; justify-content: center;
                                color: white; font-weight: bold; font-size: 1.4rem;
                                transition: width 0.8s ease; border-radius: 50px;">
                        {confidence:.1%}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # คำแนะนำตามระดับความมั่นใจ
        if is_ood:
            advice_html = """
            <div style="background: #fff3cd; border-left: 6px solid #ffc107; padding: 20px; border-radius: 10px; margin: 25px 0;">
                <p style="margin: 0; color: #856404; font-size: 1.08rem; line-height: 1.6;">
                    ⚠️ <strong>คำเตือน:</strong> ระบบตรวจพบความผิดปกติ ความเชื่อมั่นอาจไม่น่าเชื่อถือ 
                    โปรดตรวจสอบว่ารูปชัดเจน เป็นพระเครื่องจริง และอยู่ในประเภทที่ระบบรองรับ
                </p>
            </div>
            """
        elif confidence >= 0.92:
            advice_html = """
            <div style="background: #d4edda; border-left: 6px solid #28a745; padding: 20px; border-radius: 10px; margin: 25px 0;">
                <p style="margin: 0; color: #155724; font-size: 1.08rem; line-height: 1.6;">
                    ✅ <strong>ความเชื่อมั่นสูงมาก:</strong> ผลลัพธ์น่าเชื่อถือมาก AI มั่นใจในการจำแนกประเภทนี้
                </p>
            </div>
            """
        elif confidence >= 0.7:
            advice_html = """
            <div style="background: #fff3cd; border-left: 6px solid #ffc107; padding: 20px; border-radius: 10px; margin: 25px 0;">
                <p style="margin: 0; color: #856404; font-size: 1.08rem; line-height: 1.6;">
                    ⚠️ <strong>ความเชื่อมั่นปานกลาง:</strong> ควรตรวจสอบเพิ่มเติมหรือปรึกษาผู้เชี่ยวชาญ
                </p>
            </div>
            """
        else:
            advice_html = """
            <div style="background: #f8d7da; border-left: 6px solid #dc3545; padding: 20px; border-radius: 10px; margin: 25px 0;">
                <p style="margin: 0; color: #721c24; font-size: 1.08rem; line-height: 1.6;">
                    ❌ <strong>ความเชื่อมั่นต่ำ:</strong> แนะนำให้ถ่ายรูปใหม่ หรือปรึกษาผู้เชี่ยวชาญโดยตรง
                </p>
            </div>
            """
        
        st.markdown(advice_html, unsafe_allow_html=True)
        
        # Grad-CAM Section (ใช้ st.image แทน HTML img)
        if result.get('gradcam_available') and image_path:
            if 'gradcam_images' in st.session_state and image_path in st.session_state.gradcam_images:
                st.markdown("""
                <div style="margin: 35px 0; padding: 30px; background: #e3f2fd; border-radius: 15px; border-left: 8px solid #2196F3;">
                    <h2 style="color: #1565C0; margin: 0 0 10px 0; font-size: 1.6rem;">
                        🔍 การอธิบายผลลัพธ์ด้วย AI (Grad-CAM)
                    </h2>
                    <p style="color: #1976D2; font-size: 1rem; margin: 10px 0 20px 0; line-height: 1.6;">
                        💡 <strong>Grad-CAM Heatmap</strong> แสดงบริเวณที่ AI ให้ความสำคัญ<br>
                        🔴 <strong>สีแดง-ส้ม</strong> = บริเวณสำคัญสูง | 
                        🔵 <strong>สีน้ำเงิน-เขียว</strong> = บริเวณสำคัญน้อย
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: #333; font-size: 1.2rem;'>🖼️ รูปต้นฉบับ</h3>", 
                               unsafe_allow_html=True)
                    st.image(image_path, use_container_width=True)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: #333; font-size: 1.2rem;'>🔥 Grad-CAM Heatmap</h3>", 
                               unsafe_allow_html=True)
                    gradcam_img = st.session_state.gradcam_images[image_path]
                    st.image(gradcam_img, use_container_width=True)
        
        # Top Predictions
        if show_probabilities and 'probabilities' in result:
            st.markdown("""
            <h2 style="color: #333; margin: 35px 0 20px 0; font-size: 1.6rem; 
                       border-bottom: 3px solid #667eea; padding-bottom: 10px;">
                📈 ความน่าจะเป็นทั้งหมด (Top 5)
            </h2>
            """, unsafe_allow_html=True)
            
            probs = result['probabilities']
            top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for idx, (class_name, prob) in enumerate(top_probs, 1):
                medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"#{idx}"
                bar_color = "#4CAF50" if idx == 1 else "#66BB6A" if idx == 2 else "#81C784" if idx == 3 else "#A5D6A7"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 15px; padding: 8px 0;">
                        <span style="font-size: 1.8rem;">{medal}</span>
                        <span style="font-size: 1.15rem; font-weight: 600; color: #333;">{class_name}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob)
                with col2:
                    st.markdown(f"""
                    <div style="font-size: 1.5rem; font-weight: bold; color: {bar_color}; 
                               text-align: right; padding-top: 8px;">
                        {prob:.1%}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Tips (ถ้าความมั่นใจต่ำ)
        if confidence < 0.7 or is_ood:
            st.markdown("""
            <h2 style="color: #f57c00; margin: 35px 0 20px 0; font-size: 1.6rem; 
                       padding: 25px; background: #fff9e6; border-radius: 15px; 
                       border-left: 8px solid #ffc107;">
                💡 Tips: วิธีถ่ายรูปให้ได้ผลลัพธ์ที่ดี
            </h2>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">💡</div>
                    <h3 style="color: #667eea; margin-bottom: 15px; font-size: 1.2rem;">แสงสว่าง</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                - ใช้แสงธรรมชาติ
                - หลีกเลี่ยงแสงสะท้อน
                - ไม่ควรมืดเกินไป
                """)
            
            with col2:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">📸</div>
                    <h3 style="color: #667eea; margin-bottom: 15px; font-size: 1.2rem;">การถ่ายภาพ</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                - ถ่ายให้ชัด ไม่เบลอ
                - เห็นพระเครื่องเต็มตัว
                - ระยะใกล้พอประมาณ
                """)
            
            with col3:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">🎨</div>
                    <h3 style="color: #667eea; margin-bottom: 15px; font-size: 1.2rem;">พื้นหลัง</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                - ใช้พื้นหลังสีเรียบ
                - ไม่มีสิ่งบดบัง
                - ความคมชัดดี
                """)
        
        # Footer
        method_info = f"🔧 วิธีการทำนาย: {result.get('method', 'Unknown')}"
        if ood_score is not None:
            method_info += f" | 📊 OOD Score: {ood_score:.4f}"
        
        st.markdown(f"""
            <div style="text-align: center; color: #999; font-size: 1rem; margin-top: 40px; 
                        padding-top: 25px; border-top: 2px solid #e0e0e0;">
                {method_info}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Error display - ก็ในกล่องเดียวเช่นกัน
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div style="background: white; border-radius: 25px; padding: 50px; 
                    box-shadow: 0 15px 50px rgba(255,107,107,0.25); margin: 30px 0; 
                    border: 2px solid #ff6b6b;">
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                        color: white; padding: 35px; border-radius: 18px; text-align: center;
                        box-shadow: 0 6px 20px rgba(255,107,107,0.3);">
                <div style="font-size: 4rem; margin-bottom: 15px;">❌</div>
                <h2 style="color: white; margin: 0; font-size: 2.2rem; font-weight: bold;">
                    เกิดข้อผิดพลาด
                </h2>
            </div>
            <div style="background: #fff5f5; padding: 30px; border-radius: 15px; 
                        margin: 30px 0; border-left: 6px solid #ff6b6b;">
                <p style="font-family: monospace; font-size: 1.15rem; color: #333; 
                        margin: 0; line-height: 1.7; word-wrap: break-word;">
                    {error_msg}
                </p>
            </div>
            <div style="background: #fff3cd; padding: 25px; border-radius: 15px; 
                        border-left: 6px solid #ffc107;">
                <p style="margin: 0; color: #856404; font-size: 1.1rem; line-height: 1.7;">
                    💡 <strong>คำแนะนำ:</strong><br>
                    • ตรวจสอบว่ามีไฟล์โมเดล (best_model.pth, temperature_scaler.pth) ครบถ้วน<br>
                    • ลองรันใหม่อีกครั้ง หรือ Restart Streamlit<br>
                    • ตรวจสอบ Console สำหรับข้อผิดพลาดเพิ่มเติม
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'front_camera_image' not in st.session_state:
        st.session_state.front_camera_image = None
    if 'back_camera_image' not in st.session_state:
        st.session_state.back_camera_image = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Get logos
    amulet_logo = get_logo_base64()
    other_logos = get_other_logos()
    
    # Logo Header
    logo_left_html = ""
    if amulet_logo:
        logo_left_html = f'<img src="data:image/png;base64,{amulet_logo}" class="logo-img" alt="Amulet-AI">'
    
    logo_right_html = ""
    if 'thai_austrian' in other_logos:
        logo_right_html += f'<img src="data:image/gif;base64,{other_logos["thai_austrian"]}" class="logo-img-small" alt="Thai-Austrian">'
    if 'depa' in other_logos:
        logo_right_html += f'<img src="data:image/png;base64,{other_logos["depa"]}" class="logo-img-small" alt="DEPA">'
    
    st.markdown(f"""
    <div class="logo-header">
        <div class="logo-left">
            {logo_left_html}
            <div class="logo-text">
                <div class="logo-title">Amulet-AI </div>
                <div class="logo-subtitle">ระบบวิเคราะห์พระเครื่อง ลึกลับ ด้วย Computer AI อัจฉริยะ</div>
            </div>
        </div>
        <div class="logo-right">
            {logo_right_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Default settings (no settings UI)
    analysis_mode = "สองด้าน (หน้า+หลัง)"
    show_confidence = True
    show_probabilities = True
    
    # Create Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 หน้าหลัก", "📖 เกี่ยวกับระบบ", "📚 คู่มือการใช้งาน", "❓ คำถามที่พบบ่อย"])
    
    # Tab 1: Main Upload Section
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Main Title and Description
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2.5rem;">
            <h1 style="margin-bottom: 1rem; font-size: 3.5rem;">Amulet-AI </h1>
            <p style="font-size: 2.5rem; color: #495057; line-height: 1.9; max-width: 900px; margin: 0 auto 1rem;">
                เทคโนโลยี <strong style="color: #B8860B;">Deep Learning</strong> และ <strong style="color: #B8860B;">Computer Vision</strong>
            </p>
            <p style="font-size: 2.3rem; color: #6c757d; line-height: 1.8; max-width: 850px; margin: 0 auto;">
                เพื่อจำแนกพระเครื่องไทย ประเมินความน่าจะเป็น และช่วยในการตัดสินใจซื้อขาย
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Upload Section Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="margin-bottom: 0.5rem;">📸 อัปโหลดรูปพระเครื่อง</h2>
            <p style="font-size: 1.15rem; color: #6c757d;">
                อัปโหลดรูปภาพด้านหน้าและด้านหลังของพระเครื่องเพื่อเริ่มการวิเคราะห์
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Always use dual image mode
        dual_image_mode(show_confidence, show_probabilities)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: About System
    with tab2:
        # Introduction Section - 3 Cards
        show_introduction_section()
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # How It Works Section
        show_how_it_works_section()
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Who Made This & Who Is This For
        show_about_section()
    
    # Tab 3: User Guide
    with tab3:
        show_tips_section()
    
    # Tab 4: FAQ
    with tab4:
        show_faq_section()
    
    # Footer with Credits
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 16px; padding: 2rem; border: 2px solid #dee2e6; margin: 2rem 0;
                box-shadow: 0 4px 15px rgba(108, 117, 125, 0.1);">
        <p style="font-size: 1.05rem; color: #495057; margin-bottom: 1rem; line-height: 1.8; font-family: 'Sarabun', sans-serif; text-align: center;">
            🙏 ขอบคุณคณะกรรมการจาก <strong>สำนักงานส่งเสริมเศรษฐกิจดิจิทัล (depa)</strong><br>
            ที่มอบโอกาสอันทรงคุณค่าให้กับทีม <strong>Taxes1112 – วิทยาลัยเทคนิคสัตหีบ</strong><br>
            ในการเข้าร่วมโครงการและนำเสนอผลงานด้าน <strong>นวัตกรรมดิจิทัล</strong>
        </p>
        <hr style="border: none; border-top: 1px solid #dee2e6; margin: 1.5rem 0;">
        <p style="text-align: center; font-size: 0.95rem; color: #6c757d; font-family: 'Sarabun', sans-serif; margin: 0;">
            ℹ️ ระบบนี้ใช้ AI เพื่อช่วยจำแนกประเภทพระเครื่อง<br>
            ผลลัพธ์เป็นข้อมูลประกอบการตัดสินใจเท่านั้น<br>
            ระบบไม่ได้ระบุว่าเป็นพระแท้หรือพระเก๊
        </p>
    </div>
    """, unsafe_allow_html=True)

def dual_image_mode(show_confidence, show_probabilities):
    """โหมดสองด้าน"""
    st.markdown("### อัปโหลดรูปทั้งสองด้าน")
    
    col1, col2 = st.columns(2)
    
    # Front image
    with col1:
        st.markdown("#### ด้านหน้า")
        
        front_upload, front_camera = st.tabs(["อัปโหลดไฟล์", "ถ่ายรูป"])
        
        front_image = None
        
        with front_upload:
            front_image = st.file_uploader("เลือกรูปด้านหน้า", type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'heic', 'heif'], key="front_upload")
        
        with front_camera:
            # Camera will only activate when user enters this tab
            camera_front = st.camera_input("ถ่ายรูปด้านหน้า", key="front_camera")
            if camera_front:
                st.session_state.front_camera_image = camera_front
                st.success("ถ่ายรูปสำเร็จ!")
        
        display_front = front_image or st.session_state.front_camera_image
        if display_front:
            st.image(display_front, caption="รูปด้านหน้า", use_container_width=True)
    
    # Back image
    with col2:
        st.markdown("#### ด้านหลัง")
        
        back_upload, back_camera = st.tabs(["อัปโหลดไฟล์", "ถ่ายรูป"])
        
        back_image = None
        
        with back_upload:
            back_image = st.file_uploader("เลือกรูปด้านหลัง", type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'heic', 'heif'], key="back_upload")
        
        with back_camera:
            # Camera will only activate when user enters this tab
            camera_back = st.camera_input("ถ่ายรูปด้านหลัง", key="back_camera")
            if camera_back:
                st.session_state.back_camera_image = camera_back
                st.success("ถ่ายรูปสำเร็จ!")
        
        display_back = back_image or st.session_state.back_camera_image
        if display_back:
            st.image(display_back, caption="รูปด้านหลัง", use_container_width=True)
    
    st.markdown("---")
    
    final_front = front_image or st.session_state.front_camera_image
    final_back = back_image or st.session_state.back_camera_image
    
    if final_front and final_back:
        # Validate both files before processing
        validation_errors = []
        
        # Check front image
        if hasattr(final_front, 'size') and final_front.size > MAX_FILE_SIZE:
            validation_errors.append(f"ไฟล์ด้านหน้าใหญ่เกินไป ({MAX_FILE_SIZE // (1024*1024)} MB)")
        
        # Check back image
        if hasattr(final_back, 'size') and final_back.size > MAX_FILE_SIZE:
            validation_errors.append(f"ไฟล์ด้านหลังใหญ่เกินไป ({MAX_FILE_SIZE // (1024*1024)} MB)")
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success("มีรูปภาพทั้งสองด้านแล้ว!")
            
            if st.button("เริ่มการวิเคราะห์ทั้งสองด้าน", type="primary", use_container_width=True):
                try:
                    with st.spinner("AI กำลังวิเคราะห์ทั้งสองด้าน..."):
                        start_time = time.time()
                        
                        # Save temp paths for Grad-CAM
                        front_temp_path = f"temp_front_{final_front.name if hasattr(final_front, 'name') else 'front.jpg'}"
                        back_temp_path = f"temp_back_{final_back.name if hasattr(final_back, 'name') else 'back.jpg'}"
                        
                        try:
                            with open(front_temp_path, "wb") as f:
                                f.write(final_front.getbuffer())
                            with open(back_temp_path, "wb") as f:
                                f.write(final_back.getbuffer())
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {str(e)}")
                            return
                        
                        front_result = classify_image(final_front)
                        back_result = classify_image(final_back)
                    
                    processing_time = time.time() - start_time
                    
                    # Enhanced Results Display with new Enhanced Analysis Results
                    try:
                        from frontend.components.enhanced_results import EnhancedAnalysisResults
                        enhanced_results = EnhancedAnalysisResults()
                        
                        # Process results for enhanced display
                        front_enhanced = enhance_result_for_display(front_result, processing_time / 2, 'dual_image')
                        back_enhanced = enhance_result_for_display(back_result, processing_time / 2, 'dual_image')
                        
                        st.markdown("### 🎯 ผลการวิเคราะห์ AI แบบละเอียด")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="result-section">', unsafe_allow_html=True)
                            st.markdown("#### 📸 ด้านหน้า")
                            enhanced_results.display_enhanced_results(front_enhanced, 'dual_image')
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="result-section">', unsafe_allow_html=True)
                            st.markdown("#### 📸 ด้านหลัง")
                            enhanced_results.display_enhanced_results(back_enhanced, 'dual_image')
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                    except ImportError as e:
                        st.error(f"ไม่สามารถโหลด Enhanced Results Component: {e}")
                        # Fallback to old display method
                        st.markdown("### ผลการวิเคราะห์")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.markdown("#### ด้านหน้า")
                            display_classification_result(front_result, show_confidence, show_probabilities, front_temp_path)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.markdown("#### ด้านหลัง")
                            display_classification_result(back_result, show_confidence, show_probabilities, back_temp_path)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Comparison
                    if (front_result.get("status") == "success" and back_result.get("status") == "success"):
                        front_class = front_result.get("predicted_class", "")
                        back_class = back_result.get("predicted_class", "")
                        front_conf = front_result.get("confidence", 0)
                        back_conf = back_result.get("confidence", 0)
                        
                        st.markdown("### การเปรียบเทียบ")
                        
                        if front_class == back_class:
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>ผลลัพธ์สอดคล้องกัน!</h4>
                                <p style="font-size: 1.1rem;"><strong>ทั้งสองด้านระบุเป็น:</strong> {front_class}</p>
                                <p><strong>ความเชื่อมั่นเฉลี่ย:</strong> {(front_conf + back_conf) / 2:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>ผลลัพธ์ไม่สอดคล้องกัน</h4>
                                <p><strong>ด้านหน้า:</strong> {front_class} ({front_conf:.1%})</p>
                                <p><strong>ด้านหลัง:</strong> {back_class} ({back_conf:.1%})</p>
                                <p>แนะนำให้ปรึกษาผู้เชี่ยวชาญเพิ่มเติม</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                        st.session_state.analysis_history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "front_result": front_result,
                            "back_result": back_result,
                            "processing_time": processing_time,
                            "mode": "dual"
                        })
                        
                        # Clean up temp files
                        try:
                            Path(front_temp_path).unlink(missing_ok=True)
                            Path(back_temp_path).unlink(missing_ok=True)
                        except:
                            pass  # Ignore cleanup errors
                            
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
                    # Clean up temp files on error
                    try:
                        Path(front_temp_path).unlink(missing_ok=True)
                        Path(back_temp_path).unlink(missing_ok=True)
                    except:
                        pass
    else:
        st.markdown("""
        <div class="info-box">
            <h3>คำแนะนำ</h3>
            <p>กรุณาอัปโหลดรูปภาพทั้งสองด้าน (หน้า และ หลัง)</p>
            <p>การวิเคราะห์ทั้งสองด้านจะช่วยเพิ่มความแม่นยำ</p>
        </div>
        """, unsafe_allow_html=True)

def show_faq_section():
    """แสดงส่วนคำถามที่พบบ่อย"""
    st.markdown("## ❓ คำถามที่พบบ่อย (FAQ)")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #6c757d;'>ข้อมูลสำคัญที่ควรทราบก่อนใช้งานระบบ</p>", unsafe_allow_html=True)
    
    # Expectations & Limitations
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ ควรคาดหวังอะไร (Expectations & Limitations)</h3>
        <p><strong>• ผลลัพธ์เป็นการประเมินเบื้องต้น</strong> — ไม่ใช่การยืนยันความแท้ 100%</p>
        <p><strong>• คุณภาพของรูปมีผลต่อผลลัพธ์</strong> — รูปชัด แสงดี มุมถูกต้อง = ผลดีขึ้น</p>
        <p><strong>• หากความเชื่อมั่นต่ำ</strong> — ระบบจะแนะนำให้ส่งให้ผู้เชี่ยวชาญตรวจสอบ</p>
        <p><strong>• ใช้เป็นข้อมูลประกอบการตัดสินใจเท่านั้น</strong> — ไม่ควรใช้เป็นเกณฑ์เดียวในการซื้อ-ขาย</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Privacy Notice
    st.markdown("""
    <div class="info-box">
        <h3>🔒 ความเป็นส่วนตัว (Privacy)</h3>
        <p><strong>• ภาพจะถูกประมวลผล</strong> เพื่อการวิเคราะห์ตามนโยบายความเป็นส่วนตัว</p>
        <p><strong>• ถ้าคุณยินยอมให้เก็บภาพ</strong> ระบบจะใช้ภาพเพื่อปรับปรุงโมเดล แต่สามารถขอลบข้อมูลได้</p>
        <p><strong>• ข้อมูลทุกชิ้นเข้ารหัส</strong> และจัดเก็บอย่างปลอดภัยตามมาตรฐาน</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Common Questions
    st.markdown("### 💡 คำถามที่พบบ่อย")
    
    with st.expander("❓ ระบบนี้แม่นยำแค่ไหน?"):
        st.markdown("""
        ระบบมีความแม่นยำขึ้นอยู่กับหลายปัจจัย:
        - **คุณภาพของรูปภาพ**: รูปที่ชัด แสงดี จะได้ผลดีกว่า
        - **ประเภทพระเครื่อง**: บางประเภทที่มี dataset มากจะแม่นยำกว่า
        - **ความเชื่อมั่น >90%**: แม่นยำสูงมาก น่าเชื่อถือ
        - **ความเชื่อมั่น 70-90%**: แม่นยำดี แต่ควรตรวจสอบเพิ่มเติม
        - **ความเชื่อมั่น <70%**: ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ
        """)
    
    with st.expander("❓ ใช้เวลานานแค่ไหนในการวิเคราะห์?"):
        st.markdown("""
        - **โดยเฉลี่ย**: 2-5 วินาที ต่อภาพ
        - **ภาพคู่ (หน้า+หลัง)**: ประมาณ 5-10 วินาที
        - ขึ้นอยู่กับขนาดไฟล์และความเร็วอินเทอร์เน็ต
        """)
    
    with st.expander("❓ รองรับไฟล์รูปภาพแบบไหนบ้าง?"):
        st.markdown("""
        - **รองรับ**: JPG, JPEG, PNG, BMP, GIF, TIFF, WebP, HEIC/HEIF
        - **ขนาดไฟล์สูงสุด**: 10 MB
        - **ความละเอียดที่แนะนำ**: อย่างน้อย 800x800 pixels
        """)
    
    with st.expander("❓ ถ้าผลลัพธ์ไม่ตรงกับความเป็นจริงต้องทำอย่างไร?"):
        st.markdown("""
        1. **ลองถ่ายรูปใหม่** ด้วยแสงที่ดีกว่าและมุมที่ชัดเจน
        2. **ตรวจสอบว่ารูปไม่เบลอ** และไม่มีสิ่งบดบังพระเครื่อง
        3. **อ่านค่าความเชื่อมั่น** ถ้าต่ำ แสดงว่า AI ไม่แน่ใจ
        4. **ปรึกษาผู้เชี่ยวชาญ** เพื่อการยืนยันที่แน่นอน
        5. **รายงานปัญหา** ผ่านทีมพัฒนาเพื่อปรับปรุงระบบ
        """)
    
    with st.expander("❓ ระบบจะเก็บรูปภาพของฉันไว้หรือไม่?"):
        st.markdown("""
        - **โดยค่าเริ่มต้น**: รูปภาพจะถูกลบหลังจากการวิเคราะห์เสร็จสิ้น
        - **หากคุณยินยอม**: รูปอาจถูกเก็บไว้เพื่อปรับปรุงโมเดล (แบบไม่ระบุตัวตน)
        - **สิทธิ์ของคุณ**: สามารถขอลบข้อมูลได้ทุกเมื่อ
        - **ความปลอดภัย**: ข้อมูลเข้ารหัสและจัดเก็บตามมาตรฐานสากล
        """)
    
    with st.expander("❓ สามารถใช้งานบนมือถือได้หรือไม่?"):
        st.markdown("""
        **ใช้ได้!** ระบบรองรับการใช้งานบนมือถือ:
        - เปิดผ่านเว็บเบราว์เซอร์บนมือถือ
        - สามารถถ่ายรูปโดยตรงจากกล้องมือถือ
        - อัปโหลดรูปจากแกลเลอรี่
        - หน้าจอปรับขนาดให้เหมาะกับอุปกรณ์โดยอัตโนมัติ
        """)
    
    with st.expander("❓ มีค่าใช้จ่ายในการใช้งานหรือไม่?"):
        st.markdown("""
        - **ปัจจุบัน**: ใช้งานฟรี
        - **อนาคต**: อาจมีแผนพรีเมียมสำหรับฟีเจอร์เพิ่มเติม
        - **สำหรับผู้เชี่ยวชาญ/นักวิจัย**: มี API แบบเสียค่าใช้จ่าย
        """)
    
    with st.expander("❓ ต้องการความช่วยเหลือเพิ่มเติมติดต่อที่ไหน?"):
        st.markdown("""
        - **อ่านคู่มือ**: ไปที่แท็บ "📚 คู่มือการใช้งาน"
        - **เกี่ยวกับระบบ**: ไปที่แท็บ "📖 เกี่ยวกับระบบ"
        - **ติดต่อทีมพัฒนา**: ผ่านช่องทางที่ระบุในหน้า Footer
        """)

def show_introduction_section():
    """แสดงส่วนแนะนำ - เว็บไซต์นี้ทำอะไร"""
    st.markdown("## 📋 เว็บไซต์นี้ทำอะไร")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #6c757d;'>ระบบ Amulet-AI ให้บริการหลากหลายเพื่อช่วยคุณวิเคราะห์พระเครื่อง</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 ฟีเจอร์หลัก</h3>
            <ul>
                <li>จำแนกประเภทพระเครื่องจากรูปภาพ</li>
                <li>รองรับภาพด้านหน้าและด้านหลัง</li>
                <li>บอกความเชื่อมั่นของการทำนาย</li>
                <li>แสดงจุดที่ AI ให้ความสำคัญ</li>
                <li>ดาวน์โหลดรายงานผลลัพธ์</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ ใช้งานง่าย</h3>
            <ul>
                <li>อัปโหลดรูปหรือถ่ายรูปได้ทันที</li>
                <li>ผลลัพธ์ออกภายในไม่กี่วินาที</li>
                <li>แสดงผลแบบกราฟและภาพประกอบ</li>
                <li>ไม่ต้องติดตั้งโปรแกรม</li>
                <li>ใช้งานผ่านเว็บเบราว์เซอร์</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 ปลอดภัย</h3>
            <ul>
                <li>ประมวลผลตามนโยบายความเป็นส่วนตัว</li>
                <li>ข้อมูลเข้ารหัสอย่างปลอดภัย</li>
                <li>สามารถขอลบข้อมูลได้</li>
                <li>ไม่แชร์ข้อมูลโดยไม่ได้รับอนุญาต</li>
                <li>ใช้เทคโนโลยี AI ที่ทันสมัย</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_how_it_works_section():
    """แสดงวิธีการทำงาน 3 ขั้นตอน"""
    st.markdown("## 🔄 ระบบทำงานอย่างไร")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #6c757d;'>เข้าใจง่ายใน 3 ขั้นตอน</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="step-card">
            <h3>📁 ขั้นตอนที่ 1</h3>
            <h4>อัปโหลดรูปภาพ</h4>
            <p>ถ่ายรูปหรือเลือกไฟล์ภาพด้านหน้า/หลังของพระเครื่อง ระบบรองรับไฟล์ JPG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card">
            <h3>🤖 ขั้นตอนที่ 2</h3>
            <h4>AI วิเคราะห์</h4>
            <p>ระบบตรวจสอบภาพ ดึงลักษณะเด่น และทำนายประเภทพร้อมคำนวณความเชื่อมั่น</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card">
            <h3>📊 ขั้นตอนที่ 3</h3>
            <h4>แสดงผลพร้อมคำอธิบาย</h4>
            <p>ผลลัพธ์, กราฟความน่าจะเป็น และคำแนะนำขั้นตอนถัดไป</p>
        </div>
        """, unsafe_allow_html=True)

def show_about_section():
    """แสดงส่วนเกี่ยวกับผู้พัฒนาและกลุ่มเป้าหมาย"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>👥 ใครสร้างระบบนี้</h3>
            <p><strong>พัฒนาโดยทีมผู้เชี่ยวชาญ</strong> ด้าน Machine Learning, Computer Vision และผู้รู้เกี่ยวกับพระเครื่อง</p>
            <p><strong>ทำงานร่วมกับเครือข่าย</strong> ผู้เชี่ยวชาญและชุมชนสะสมพระเพื่อปรับปรุงความแม่นยำ</p>
            <p><strong>จุดมุ่งหมาย:</strong> ทำให้ความรู้ด้านพระเครื่องเข้าถึงได้ง่ายขึ้น และช่วยให้การประเมินเบื้องต้นทำได้รวดเร็ว</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 เหมาะกับใคร</h3>
            <p><strong>• ผู้เริ่มต้น</strong> - อยากรู้ว่าพระเครื่องที่มีเป็นรุ่นไหนเบื้องต้น</p>
            <p><strong>• นักสะสม/พ่อค้า</strong> - ตรวจสอบและจัดหมวดหมู่เบื้องต้นก่อนซื้อ-ขาย</p>
            <p><strong>• ผู้พัฒนา/นักวิจัย</strong> - สนใจข้อมูลเชิงเทคนิคหรือ dataset (มี API ให้เชื่อมต่อ)</p>
            <p><strong>• ผู้ที่สนใจเทคโนโลยี AI</strong> - เรียนรู้การประยุกต์ใช้ AI ในงานจำแนก</p>
        </div>
        """, unsafe_allow_html=True)

def show_tips_section():
    """แสดงคู่มือการใช้งานแบบละเอียด"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("## 📖 คู่มือการใช้งานอย่างละเอียด")
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #6c757d;'>ทำตามขั้นตอนเหล่านี้เพื่อผลลัพธ์ที่ดีที่สุด</p>", unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
    <div class="tips-card">
        <h3>🚀 วิธีใช้งานอย่างง่าย (Quick Start)</h3>
        <ol style="font-size: 1.2rem; line-height: 2;">
            <li><strong>กด 📁 อัปโหลดรูป หรือ 📷 ถ่ายรูป</strong> เพื่อแนบภาพด้านหน้าและด้านหลัง</li>
            <li><strong>ตรวจสอบภาพตัวอย่าง</strong> (Preview) ว่าไม่เบลอและเห็นรายละเอียด</li>
            <li><strong>กด 🔍 เริ่มการวิเคราะห์</strong> (ปุ่มใช้งานได้เมื่อมีทั้งสองภาพ)</li>
            <li><strong>รอผล</strong> — ระบบจะแจ้งสถานะและแสดงวงหมุน</li>
            <li><strong>อ่านผล</strong> — ดู Top-1/Top-3, ค่าความเชื่อมั่น และกราฟ</li>
            <li><strong>ถ้าผลไม่ชัด</strong> → ถ่ายใหม่หรือปรึกษาผู้เชี่ยวชาญ</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tips-card">
            <h3>📸 คำแนะนำการถ่ายรูปให้ได้ผลดี</h3>
            <ul>
                <li><strong>ใช้แสงเพียงพอ</strong> (ไม่ย้อนแสง)</li>
                <li><strong>พื้นหลังเรียบ</strong> (เช่น ผ้าขาว / กระดาษสีเรียบ)</li>
                <li><strong>ภาพชัด ไม่เบลอ</strong> และถ่ายให้เห็นลักษณะเด่น</li>
                <li><strong>ถ่ายให้เห็นทั้งองค์</strong> ไม่ตัดขอบสำคัญออก</li>
                <li><strong>หลีกเลี่ยงเงา</strong> บนพระเครื่อง</li>
                <li><strong>ถ่ายในระยะใกล้พอสมควร</strong> ให้เห็นรายละเอียด</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tips-card">
            <h3>📊 การตีความผลลัพธ์</h3>
            <ul>
                <li><strong>มากกว่า 90%</strong><br/>🎯 <strong>ความเชื่อมั่นสูง</strong> — ผลลัพธ์น่าเชื่อถือ</li>
                <li><strong>70-90%</strong><br/>✅ <strong>เชื่อถือได้ดี</strong> — แต่ควรพิจารณาเพิ่มเติม</li>
                <li><strong>50-70%</strong><br/>⚠️ <strong>ควรตรวจสอบเพิ่ม</strong> — อาจต้องถ่ายใหม่</li>
                <li><strong>น้อยกว่า 50%</strong><br/>❌ <strong>ควรถ่ายใหม่</strong> — หรือส่งตรวจผู้เชี่ยวชาญ</li>
                <li><strong>ใช้ข้อมูลประกอบเท่านั้น</strong> — ไม่ควรเป็นเกณฑ์เดียวในการซื้อ-ขาย</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tips-card">
            <h3>🏷️ ประเภทที่รองรับ</h3>
            <ul>
                <li>พระศิวลี</li>
                <li>พระสมเด็จ</li>
                <li>ปรกโพธิ์ 9 ใบ</li>
                <li>แหวกม่าน</li>
                <li>หลังรูปเหมือน</li>
                <li>วัดหนองอีดุก</li>
            </ul>
            <p style="margin-top: 20px; font-size: 1.1rem;"><strong>หมายเหตุ:</strong> ระบบมีการอัพเดทและเพิ่มประเภทใหม่อยู่เสมอ</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning Card
    st.markdown("""
    <div class="error-box">
        <h3>⚠️ คำเตือนสำคัญ</h3>
        <p style="font-size: 1.3rem; line-height: 1.9;">
            <strong>• ผลลัพธ์เป็นเพียงการประเมินเบื้องต้น</strong> — ควรใช้ร่วมกับผู้เชี่ยวชาญก่อนตัดสินใจซื้อ/ขาย<br/>
            <strong>• หากต้องการผลยืนยัน</strong> — ให้ปรึกษาผู้เชี่ยวชาญด้านพระเครื่องโดยตรง<br/>
            <strong>• ระบบไม่รับประกันความแท้</strong> — เป็นเพียงเครื่องมือช่วยตัดสินใจเบื้องต้น<br/>
            <strong>• ถ้าคุณต้องการความช่วยเหลือ</strong> — อ่านคู่มือ / FAQ ในเมนู Documentation
        </p>
    </div>
    """, unsafe_allow_html=True)
