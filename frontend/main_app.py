#!/usr/bin/env python3
"""
Amulet-AI - Production Frontend
ระบบจำแนกพระเครื่องอัจฉริยะ
Thai Amulet Classification System

ระบบ AI สำหรับการจำแนกประเภทพระเครื่องไทย
ใช้เทคโนโลยี Deep Learning และ Computer Vision
เพื่อช่วยวิเคราะห์และประเมินพระเครื่องอย่างแม่นยำ
"""

import streamlit as st
import requests
import tempfile
import joblib
import sys
from pathlib import Path

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import enhanced modules (with fallback)
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
except:
    def error_handler(error_type="general"):
        def decorator(func):
            return func
        return decorator

    class performance_monitor:
        @staticmethod
        def log_performance(func_name, execution_time):
            pass
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Thai amulet information database
AMULET_INFO = {
    "phra_sivali": {
        "thai_name": "พระสีวลี",
        "full_name": "พระสีวลี มหาลาภ",
        "temple": "วัดต่างๆ",
        "period": "พ.ศ. 2400-2500",
        "description": "พระที่เชื่อว่านำโชคลาภ มีเงินทองใช้ไม่ขาดมือ",
        "price_range": {"min": 500, "max": 15000, "avg": 3500}
    },
    "portrait_back": {
        "thai_name": "พระรูปหล่อหลังภาพ",
        "full_name": "พระรูปหล่อ หลังภาพพระ",
        "temple": "วัดต่างๆ",
        "period": "พ.ศ. 2450-2550",
        "description": "พระรูปหล่อที่มีภาพพระด้านหลัง นิยมในยุคปัจจุบัน",
        "price_range": {"min": 300, "max": 8000, "avg": 2200}
    },
    "prok_bodhi_9_leaves": {
        "thai_name": "ใบโพธิ์ 9 ใบ",
        "full_name": "พระใบโพธิ์ 9 ใบ",
        "temple": "วัดมหาธาตุ และวัดต่างๆ",
        "period": "พ.ศ. 2380-2450",
        "description": "พระโบราณที่มีรูปแบบใบโพธิ์ 9 ใบ สวยงามและหายาก",
        "price_range": {"min": 2000, "max": 25000, "avg": 8500}
    },
    "somdej_pratanporn_buddhagavak": {
        "thai_name": "พระสมเด็จ ประตานพรณ์",
        "full_name": "พระสมเด็จ วัดประตานพรณ์ พิมพ์ใหญ่",
        "temple": "วัดประตานพรณ์ (วัดระฆัง)",
        "period": "พ.ศ. 2397-2415 (รัชกาลที่ 4)",
        "description": "พระสมเด็จสายวัดระฆัง ของสมเด็จพระพุฒาจารย์ (โต พรหมรังสี)",
        "price_range": {"min": 5000, "max": 150000, "avg": 35000}
    },
    "waek_man": {
        "thai_name": "แหวนมาน",
        "full_name": "แหวนมานต์ขลัง",
        "temple": "วัดต่างๆ ในภาคเหนือ",
        "period": "พ.ศ. 2400-2500",
        "description": "แหวนมนต์ขลัง ป้องกันภัยและเสริมดวง",
        "price_range": {"min": 800, "max": 12000, "avg": 4200}
    },
    "wat_nong_e_duk": {
        "thai_name": "พระวัดหนองอีดุก",
        "full_name": "พระวัดหนองอีดุก จ.สุพรรณบุรี",
        "temple": "วัดหนองอีดุก สุพรรณบุรี",
        "period": "พ.ศ. 2480-2530",
        "description": "พระขุนแผนจากวัดหนองอีดุก มีชื่อเสียงด้านเมตตามหานิยม",
        "price_range": {"min": 1200, "max": 18000, "avg": 6800}
    }
}

# Import enhanced modules (with fallback)
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
except:
    def error_handler(error_type="general"):
        def decorator(func):
            return func
        return decorator

    class performance_monitor:
        @staticmethod
        def log_performance(func_name, execution_time):
            pass

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
    page_icon="พ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Modal Design CSS with Mobile Camera Support
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
    
    /* Mobile-First Responsive Design */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 20px 15px !important;
            margin: 10px 5px !important;
            border-radius: 16px !important;
        }}
        
        .logo-header {{
            flex-direction: column !important;
            padding: 30px 20px !important;
            text-align: center !important;
            gap: 20px !important;
        }}
        
        .logo-title {{
            font-size: 2.5rem !important;
        }}
        
        .logo-subtitle {{
            font-size: 1.2rem !important;
        }}
        
        .logo-img, .logo-img-small {{
            height: 120px !important;
        }}
        
        [data-testid="column"] {{
            padding: 10px 5px !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.9rem !important;
            padding: 10px 16px !important;
        }}
        
        .card, .feature-card, .result-card {{
            padding: 25px 20px !important;
            margin: 20px 0 !important;
        }}
        
        h1 {{
            font-size: 2.2rem !important;
        }}
        
        h2 {{
            font-size: 1.8rem !important;
        }}
        
        h3 {{
            font-size: 1.4rem !important;
        }}
        
        .tips-card, .success-box, .error-box, .warning-box, .info-box {{
            padding: 20px 15px !important;
            font-size: 1.1rem !important;
        }}
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
    
    /* Camera Controls - Mobile Optimized */
    .camera-container {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(128, 0, 0, 0.1);
        border: 2px solid {COLORS['primary']};
        text-align: center;
    }}
    
    .camera-button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
        color: white;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(128, 0, 0, 0.3);
        transition: all 0.3s ease;
        min-width: 140px;
    }}
    
    .camera-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(128, 0, 0, 0.4);
    }}
    
    .camera-preview {{
        max-width: 100%;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 15px 0;
    }}
    
    .camera-video {{
        width: 100%;
        max-width: 400px;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 15px 0;
    }}
    
    @media (max-width: 768px) {{
        .camera-container {{
            padding: 20px 15px;
            margin: 15px 0;
        }}
        
        .camera-button {{
            padding: 12px 20px;
            font-size: 1rem;
            min-width: 120px;
            margin: 8px;
        }}
        
        .camera-video {{
            max-width: 100%;
            height: 250px;
            object-fit: cover;
        }}
        
        .camera-preview {{
            max-height: 200px;
            object-fit: cover;
        }}
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

<script>
// Camera functionality with single permission request
let cameraStream = null;
let currentMode = 'front'; // 'front' or 'back'

function requestCameraPermission() {{
    return navigator.mediaDevices.getUserMedia({{
        video: {{
            facingMode: 'user',
            width: {{ ideal: 1280, max: 1920 }},
            height: {{ ideal: 720, max: 1080 }}
        }}
    }});
}}

function switchCamera(mode) {{
    currentMode = mode;
    if (cameraStream) {{
        // หยุดกล้องเก่า
        cameraStream.getTracks().forEach(track => track.stop());
    }}
    
    // เริ่มกล้องใหม่ตามโหมด
    const constraints = {{
        video: {{
            facingMode: mode === 'front' ? 'user' : 'environment',
            width: {{ ideal: 1280, max: 1920 }},
            height: {{ ideal: 720, max: 1080 }}
        }}
    }};
    
    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {{
            cameraStream = stream;
            const video = document.getElementById('camera-video');
            if (video) {{
                video.srcObject = stream;
            }}
        }})
        .catch(err => {{
            console.error('Error switching camera:', err);
            alert('ไม่สามารถเปลี่ยนกล้องได้: ' + err.message);
        }});
}}

function capturePhoto(targetMode) {{
    const video = document.getElementById('camera-video');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    if (video && video.videoWidth > 0) {{
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        // แปลงเป็น blob และส่งไปยัง Streamlit
        canvas.toBlob(blob => {{
            const formData = new FormData();
            formData.append('file', blob, `captured_${{targetMode}}.jpg`);
            
            // ส่งข้อมูลไปยัง Streamlit ผ่าน session state
            window.parent.postMessage({{
                type: 'camera_capture',
                mode: targetMode,
                dataUrl: canvas.toDataURL('image/jpeg', 0.8)
            }}, '*');
        }}, 'image/jpeg', 0.8);
    }}
}}

function stopCamera() {{
    if (cameraStream) {{
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }}
    const video = document.getElementById('camera-video');
    if (video) {{
        video.srcObject = null;
    }}
}}
</script>
""", unsafe_allow_html=True)

# Utility Functions
def get_logo_base64():
    """Convert logo image to base64"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), 'image', 'Amulet-AI_logo.png')
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
        logo_dir = os.path.join(os.path.dirname(__file__), 'image')

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

def create_camera_interface():
    """สร้าง interface สำหรับกล้อง"""
    st.markdown("""
    <div class="camera-container">
        <h4>📷 ถ่ายรูปด้วยกล้อง</h4>
        <p>กดปุ่มเพื่อเริ่มใช้กล้อง (ขออนุญาตครั้งเดียว ใช้ได้ทั้งหน้าและหลัง)</p>
        
        <button class="camera-button" onclick="requestCameraPermission().then(stream => {
            cameraStream = stream;
            document.getElementById('camera-video').srcObject = stream;
            document.getElementById('camera-controls').style.display = 'block';
            document.getElementById('start-camera').style.display = 'none';
        }).catch(err => {
            alert('ไม่สามารถเข้าถึงกล้องได้: ' + err.message);
        });" id="start-camera">เริ่มใช้กล้อง</button>
        
        <div id="camera-controls" style="display: none;">
            <video id="camera-video" class="camera-video" autoplay playsinline muted></video>
            <br>
            <button class="camera-button" onclick="switchCamera('user')">กล้องหน้า</button>
            <button class="camera-button" onclick="switchCamera('environment')">กล้องหลัง</button>
            <br>
            <button class="camera-button" onclick="capturePhoto('front')" style="background: #10b981;">ถ่ายรูปหน้า</button>
            <button class="camera-button" onclick="capturePhoto('back')" style="background: #3b82f6;">ถ่ายรูปหลัง</button>
            <br>
            <button class="camera-button" onclick="stopCamera(); document.getElementById('camera-controls').style.display = 'none'; document.getElementById('start-camera').style.display = 'block';" style="background: #ef4444;">หยุดกล้อง</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

def check_api_health():
    """ตรวจสอบสถานะ API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_model_status():
    """ตรวจสอบสถานะโมเดล"""
    model_files = [
        "trained_model/classifier.joblib",
        "trained_model/scaler.joblib",
        "trained_model/label_encoder.joblib"
    ]

    missing_files = []
    for file_path in model_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    return len(missing_files) == 0, missing_files

@error_handler("frontend")
def classify_image(uploaded_file):
    """จำแนกรูปภาพ"""
    try:
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

        # Fallback to local prediction
        result = local_prediction(temp_path)
        result["method"] = "Local"
        Path(temp_path).unlink(missing_ok=True)
        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "method": "None"
        }

def local_prediction(image_path):
    """การทำนายแบบ local"""
    try:
        import joblib
        
        if cv2 is None:
            return {
                "status": "error",
                "error": "OpenCV not available for image processing"
            }

        classifier = joblib.load(str(project_root / "trained_model/classifier.joblib"))
        scaler = joblib.load(str(project_root / "trained_model/scaler.joblib"))
        label_encoder = joblib.load(str(project_root / "trained_model/label_encoder.joblib"))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()

        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        try:
            labels_path = project_root / "ai_models/labels.json"
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            thai_name = labels.get("current_classes", {}).get(str(prediction), predicted_class)
        except:
            thai_name = predicted_class

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "thai_name": thai_name,
            "confidence": confidence,
            "probabilities": {
                label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def display_classification_result(result, show_confidence=True, show_probabilities=True):
    """แสดงผลการจำแนก"""
    if result.get("status") == "success" or "predicted_class" in result:
        predicted_class = result.get('predicted_class', 'Unknown')
        thai_name = result.get('thai_name', predicted_class)
        confidence = result.get('confidence', 0)

        st.markdown(f"""
        <div class="success-box">
            <h3>ผลการจำแนก</h3>
            <p style="font-size: 1.2rem;"><strong>ประเภทพระเครื่อง:</strong> {predicted_class}</p>
            <p style="font-size: 1.2rem;"><strong>ชื่อภาษาไทย:</strong> {thai_name}</p>
        </div>
        """, unsafe_allow_html=True)

        if show_confidence and confidence > 0:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("ความเชื่อมั่น", f"{confidence:.1%}")
            with col2:
                st.progress(confidence)

            if confidence >= 0.9:
                st.success("ความเชื่อมั่นสูงมาก - ผลลัพธ์น่าเชื่อถือ")
            elif confidence >= 0.7:
                st.warning("ความเชื่อมั่นปานกลาง - ควรตรวจสอบเพิ่มเติม")
            else:
                st.error("ความเชื่อมั่นต่ำ - ควรถ่ายรูปใหม่หรือปรึกษาผู้เชี่ยวชาญ")

        if show_probabilities and 'probabilities' in result:
            with st.expander("ดูความน่าจะเป็นทั้งหมด"):
                probs = result['probabilities']
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{class_name}**")
                    with col2:
                        st.write(f"{prob:.1%}")
                    st.progress(prob)

        st.caption(f"วิธีการทำนาย: {result.get('method', 'Unknown')}")

    else:
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div class="error-box">
            <h3>เกิดข้อผิดพลาด</h3>
            <p>{error_msg}</p>
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
    if 'camera_permission_granted' not in st.session_state:
        st.session_state.camera_permission_granted = False
    
    # Check system dependencies and show warnings if needed
    if not CV2_AVAILABLE:
        st.sidebar.warning("⚠️ OpenCV ไม่พร้อมใช้งาน - ฟีเจอร์การประมวลผลภาพบางส่วนอาจไม่ทำงาน")
    
    # JavaScript listener for camera captures
    st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'camera_capture') {
            // ส่งข้อมูลรูปที่ถ่ายไปยัง Streamlit session state
            const mode = event.data.mode;
            const dataUrl = event.data.dataUrl;
            
            // แปลง data URL เป็น file object
            fetch(dataUrl)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], `captured_${mode}.jpg`, { type: 'image/jpeg' });
                    
                    // อัปเดต session state (ต้องใช้ Streamlit API)
                    if (mode === 'front') {
                        window.streamlit_front_image = dataUrl;
                    } else {
                        window.streamlit_back_image = dataUrl;
                    }
                    
                    // แจ้งให้ Streamlit รู้ว่ามีการเปลี่ยนแปลง
                    window.parent.postMessage({
                        type: 'streamlit_update',
                        mode: mode,
                        dataUrl: dataUrl
                    }, '*');
                });
        }
    });
    </script>
    """, unsafe_allow_html=True)
    if 'camera_permission_granted' not in st.session_state:
        st.session_state.camera_permission_granted = False
    
    # JavaScript listener for camera captures
    st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'camera_capture') {
            // ส่งข้อมูลรูปที่ถ่ายไปยัง Streamlit session state
            const mode = event.data.mode;
            const dataUrl = event.data.dataUrl;
            
            // แปลง data URL เป็น file object
            fetch(dataUrl)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], `captured_${mode}.jpg`, { type: 'image/jpeg' });
                    
                    // อัปเดต session state (ต้องใช้ Streamlit API)
                    if (mode === 'front') {
                        window.streamlit_front_image = dataUrl;
                    } else {
                        window.streamlit_back_image = dataUrl;
                    }
                    
                    // แจ้งให้ Streamlit รู้ว่ามีการเปลี่ยนแปลง
                    window.parent.postMessage({
                        type: 'streamlit_update',
                        mode: mode,
                        dataUrl: dataUrl
                    }, '*');
                });
        }
    });
    </script>
    """, unsafe_allow_html=True)

# AI Model Functions
def check_model_status():
    """ตรวจสอบสถานะโมเดล"""
    model_files = [
        "trained_model/classifier.joblib",
        "trained_model/scaler.joblib", 
        "trained_model/label_encoder.joblib"
    ]
    
    missing_files = []
    for file_path in model_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

@error_handler("frontend")
def classify_image(uploaded_file):
    """จำแนกรูปภาพ"""
    try:
        return local_prediction_from_file(uploaded_file)
    except Exception as e:
        return {"status": "error", "error": str(e)}

def local_prediction_from_file(uploaded_file):
    """การทำนายจากไฟล์ที่อัปโหลด"""
    try:
        if cv2 is None:
            return {
                "status": "error", 
                "error": "OpenCV not installed. Please install opencv-python."
            }
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            if hasattr(uploaded_file, 'getvalue'):
                tmp_file.write(uploaded_file.getvalue())
            else:
                tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        result = local_prediction(tmp_path)
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

def local_prediction(image_path):
    """การทำนายแบบ local พร้อมข้อมูลครบถ้วน"""
    try:
        start_time = time.time()
        
        model_available, missing_files = check_model_status()
        
        if not model_available:
            return {
                "status": "error",
                "error": f"Missing model files: {', '.join(missing_files)}"
            }
        
        if cv2 is None:
            return {
                "status": "error",
                "error": "OpenCV not installed"
            }

        # Load models
        classifier = joblib.load(str(project_root / "trained_model/classifier.joblib"))
        scaler = joblib.load(str(project_root / "trained_model/scaler.joblib"))
        label_encoder = joblib.load(str(project_root / "trained_model/label_encoder.joblib"))

        # Process image
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "error": "Cannot read image file"}
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()

        # Make prediction
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        processing_time = time.time() - start_time

        # Get amulet information
        amulet_info = AMULET_INFO.get(predicted_class, {
            "thai_name": predicted_class,
            "full_name": predicted_class,
            "temple": "ไม่ระบุ",
            "period": "ไม่ระบุ",
            "description": "ไม่มีข้อมูล",
            "price_range": {"min": 0, "max": 0, "avg": 0}
        })
        
        # Create top 3 predictions
        all_classes = label_encoder.classes_
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = []
        
        for i, idx in enumerate(top_3_indices):
            class_name = all_classes[idx]
            prob = float(probabilities[idx])
            info = AMULET_INFO.get(class_name, {"thai_name": class_name})
            
            # Color coding
            if prob > 0.7:
                color = "#4caf50"
                color_name = "สูง"
            elif prob > 0.3:
                color = "#ff9800"
                color_name = "ปานกลาง"
            else:
                color = "#f44336"
                color_name = "ต่ำ"
                
            top_3_predictions.append({
                "rank": i + 1,
                "class": class_name,
                "thai_name": info.get("thai_name", class_name),
                "confidence": prob,
                "color": color,
                "color_name": color_name
            })

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "thai_name": amulet_info["thai_name"],
            "full_name": amulet_info["full_name"],
            "confidence": confidence,
            "processing_time": processing_time,
            "method": "Local AI Model (RandomForest)",
            "amulet_info": amulet_info,
            "top_3_predictions": top_3_predictions,
            "probabilities": {
                all_classes[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def display_classification_result_enhanced(result, image_label="", show_confidence=True):
    """แสดงผลการจำแนกแบบครบถ้วน"""
    if result.get("status") == "success" or "predicted_class" in result:
        predicted_class = result.get('predicted_class', 'Unknown')
        thai_name = result.get('thai_name', predicted_class)
        full_name = result.get('full_name', thai_name)
        confidence = result.get('confidence', 0)
        processing_time = result.get('processing_time', 0)
        amulet_info = result.get('amulet_info', {})
        top_3 = result.get('top_3_predictions', [])

        # Header result
        color_class = '#4caf50' if confidence > 0.8 else '#ff9800' if confidence > 0.6 else '#f44336'
        confidence_level = 'สูงมาก' if confidence > 0.8 else 'ปานกลาง' if confidence > 0.6 else 'ต่ำ'
        
        st.markdown(f"""
        <div class="success-box">
            <h3>ผลการวิเคราะห์{image_label}</h3>
            <p style="font-size: 1.3rem; margin: 15px 0;"><strong>ประเภทพระ:</strong> {full_name}</p>
            <p style="font-size: 1.2rem; margin: 10px 0;"><strong>ความมั่นใจ:</strong> 
                <span style="color: {color_class}; font-weight: bold;">
                    {confidence:.1%} ({confidence_level})
                </span>
            </p>
            <p style="font-size: 1.1rem; margin: 10px 0;"><strong>เวลาประมวลผล:</strong> {processing_time:.1f} วินาที</p>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 Predictions
        if top_3:
            st.markdown(f"""
            <div class="info-box">
                <h4>Top 3 การทำนาย{image_label}</h4>
                <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                    <thead>
                        <tr style="background: rgba(128,0,0,0.1);">
                            <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">อันดับ</th>
                            <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">รุ่น/พิมพ์</th>
                            <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">ความมั่นใจ</th>
                        </tr>
                    </thead>
                    <tbody>
            """, unsafe_allow_html=True)
            
            medals = ["🥇", "🥈", "🥉"]
            for i, pred in enumerate(top_3):
                st.markdown(f"""
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;">{medals[i]} #{pred['rank']}</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{pred['thai_name']}</td>
                            <td style="padding: 8px; text-align: center; border: 1px solid #ddd; font-weight: bold; color: {pred['color']};">{pred['confidence']:.1%}</td>
                        </tr>
                """, unsafe_allow_html=True)
            
            st.markdown("</tbody></table></div>", unsafe_allow_html=True)

        # Historical Information
        st.markdown(f"""
        <div class="tips-card">
            <h4>ข้อมูลประวัติศาสตร์</h4>
            <ul style="font-size: 1.1rem; line-height: 1.6;">
                <li><strong>ปีที่สร้าง:</strong> {amulet_info.get('period', 'ไม่ระบุ')}</li>
                <li><strong>วัด/สถานที่:</strong> {amulet_info.get('temple', 'ไม่ระบุ')}</li>
                <li><strong>คำอธิบาย:</strong> {amulet_info.get('description', 'ไม่มีข้อมูล')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Market Information
        price_info = amulet_info.get('price_range', {})
        if price_info.get('max', 0) > 0:
            st.markdown(f"""
            <div class="warning-box">
                <h4>ช่วงราคาตลาด (ข้อมูลอ้างอิง)</h4>
                <ul style="font-size: 1.1rem; line-height: 1.6;">
                    <li><strong>ต่ำสุด:</strong> {price_info.get('min', 0):,} บาท</li>
                    <li><strong>สูงสุด:</strong> {price_info.get('max', 0):,} บาท</li>
                    <li><strong>ราคาเฉลี่ย:</strong> {price_info.get('avg', 0):,} บาท</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        if show_confidence and confidence > 0:
            st.progress(confidence)

        st.caption(f"🤖 {result.get('method', 'AI Model')}")

    else:
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div class="error-box">
            <h4>เกิดข้อผิดพลาด{image_label}</h4>
            <p style="font-size: 1.1rem;">{error_msg}</p>
        </div>
        """, unsafe_allow_html=True)

def create_camera_interface():
    """สร้าง interface สำหรับกล้อง"""
    st.markdown("""
    <div class="info-box">
        <h4>📷 ฟีเจอร์กล้อง (Mobile Optimized)</h4>
        <p>ฟีเจอร์กล้องทำงานได้ดีที่สุดบนมือถือ</p>
        <p><strong>วิธีใช้งาน:</strong> เปิดเว็บไซต์ผ่านมือถือ แล้วใช้ปุ่ม "ถ่ายรูป" ในแต่ละแท็บ</p>
    </div>
    """, unsafe_allow_html=True)

    # Get logos
    """สร้าง interface สำหรับกล้อง"""
    st.markdown("""
    <div class="camera-container">
        <h4>📷 ถ่ายรูปด้วยกล้อง</h4>
        <p>กดปุ่มเพื่อเริ่มใช้กล้อง (ขออนุญาตครั้งเดียว ใช้ได้ทั้งหน้าและหลัง)</p>
        
        <button class="camera-button" onclick="alert('Camera feature available - check mobile version')">🎥 เริ่มใช้กล้อง</button>
    </div>
    """, unsafe_allow_html=True)

def check_model_status():
    """ตรวจสอบสถานะโมเดล"""
    model_files = [
        "trained_model/classifier.joblib",
        "trained_model/scaler.joblib", 
        "trained_model/label_encoder.joblib"
    ]
    
    missing_files = []
    for file_path in model_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

@error_handler("frontend")
def classify_image(uploaded_file):
    """จำแนกรูปภาพ"""
    try:
        return local_prediction_from_file(uploaded_file)
    except Exception as e:
        return {"status": "error", "error": str(e)}

def local_prediction_from_file(uploaded_file):
    """การทำนายจากไฟล์ที่อัปโหลด"""
    try:
        if cv2 is None:
            return {
                "status": "error", 
                "error": "OpenCV not installed. Please install opencv-python."
            }
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            if hasattr(uploaded_file, 'getvalue'):
                tmp_file.write(uploaded_file.getvalue())
            else:
                tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        result = local_prediction(tmp_path)
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

def local_prediction(image_path):
    """การทำนายแบบ local พร้อมข้อมูลครบถ้วน"""
    try:
        start_time = time.time()
        
        model_available, missing_files = check_model_status()
        
        if not model_available:
            return {
                "status": "error",
                "error": f"Missing model files: {', '.join(missing_files)}"
            }
        
        if cv2 is None:
            return {
                "status": "error",
                "error": "OpenCV not installed"
            }

        # Load models
        classifier = joblib.load(str(project_root / "trained_model/classifier.joblib"))
        scaler = joblib.load(str(project_root / "trained_model/scaler.joblib"))
        label_encoder = joblib.load(str(project_root / "trained_model/label_encoder.joblib"))

        # Process image
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "error": "Cannot read image file"}
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()

        # Make prediction
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        processing_time = time.time() - start_time

        # Get amulet information
        amulet_info = AMULET_INFO.get(predicted_class, {
            "thai_name": predicted_class,
            "full_name": predicted_class,
            "temple": "ไม่ระบุ",
            "period": "ไม่ระบุ",
            "description": "ไม่มีข้อมูล",
            "price_range": {"min": 0, "max": 0, "avg": 0}
        })
        
        # Create top 3 predictions
        all_classes = label_encoder.classes_
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = []
        
        for i, idx in enumerate(top_3_indices):
            class_name = all_classes[idx]
            prob = float(probabilities[idx])
            info = AMULET_INFO.get(class_name, {"thai_name": class_name})
            
            # Color coding
            if prob > 0.7:
                color = "#4caf50"
                color_name = "สูง"
            elif prob > 0.3:
                color = "#ff9800"
                color_name = "ปานกลาง"
            else:
                color = "#f44336"
                color_name = "ต่ำ"
                
            top_3_predictions.append({
                "rank": i + 1,
                "class": class_name,
                "thai_name": info.get("thai_name", class_name),
                "confidence": prob,
                "color": color,
                "color_name": color_name
            })

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "thai_name": amulet_info["thai_name"],
            "full_name": amulet_info["full_name"],
            "confidence": confidence,
            "processing_time": processing_time,
            "method": "Local AI Model (RandomForest)",
            "amulet_info": amulet_info,
            "top_3_predictions": top_3_predictions,
            "probabilities": {
                all_classes[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def display_classification_result(result, show_confidence=True, show_probabilities=True):
    """แสดงผลการจำแนกแบบครบถ้วน"""
    if result.get("status") == "success" or "predicted_class" in result:
        predicted_class = result.get('predicted_class', 'Unknown')
        thai_name = result.get('thai_name', predicted_class)
        full_name = result.get('full_name', thai_name)
        confidence = result.get('confidence', 0)
        processing_time = result.get('processing_time', 0)
        amulet_info = result.get('amulet_info', {})
        top_3 = result.get('top_3_predictions', [])

        # Header result
        st.markdown(f"""
        <div class="success-box">
            <h2>ผลการวิเคราะห์เบื้องต้น</h2>
            <p style="font-size: 1.4rem; margin: 15px 0;"><strong>✅ ประเภทพระ:</strong> {full_name}</p>
            <p style="font-size: 1.3rem; margin: 10px 0;"><strong>ความมั่นใจ:</strong> 
                <span style="color: {'#4caf50' if confidence > 0.8 else '#ff9800' if confidence > 0.6 else '#f44336'}; font-weight: bold;">
                    {confidence:.1%} ({'สูงมาก' if confidence > 0.8 else 'ปานกลาง' if confidence > 0.6 else 'ต่ำ'})
                </span>
            </p>
            <p style="font-size: 1.2rem; margin: 10px 0;"><strong>⏱️ เวลาประมวลผล:</strong> {processing_time:.1f} วินาที</p>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 Predictions
        if top_3:
            st.markdown("""
            <div class="info-box">
                <h3>🏆 Top 3 การทำนาย</h3>
                <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                    <thead>
                        <tr style="background: rgba(128,0,0,0.1);">
                            <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">อันดับ</th>
                            <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">รุ่น/พิมพ์</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">ความมั่นใจ</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">แถบสี</th>
                        </tr>
                    </thead>
                    <tbody>
            """, unsafe_allow_html=True)
            
            medals = ["🥇", "🥈", "🥉"]
            for i, pred in enumerate(top_3):
                st.markdown(f"""
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd;">{medals[i]} #{pred['rank']}</td>
                            <td style="padding: 12px; border: 1px solid #ddd;">{pred['thai_name']}</td>
                            <td style="padding: 12px; text-align: center; border: 1px solid #ddd; font-weight: bold;">{pred['confidence']:.1%}</td>
                            <td style="padding: 12px; text-align: center; border: 1px solid #ddd;"><span style="color: {pred['color']}; font-weight: bold;">●</span> {pred['color_name']}</td>
                        </tr>
                """, unsafe_allow_html=True)
            
            st.markdown("</tbody></table></div>", unsafe_allow_html=True)

        # Market Information (Mocked)
        price_info = amulet_info.get('price_range', {})
        if price_info.get('max', 0) > 0:
            st.markdown(f"""
            <div class="warning-box">
                <h3>📈 ข้อมูลตลาด (Web Scraping Data – Mocked)</h3>
                <p style="font-size: 1.1rem; margin-bottom: 15px;">ดึงจาก: เว็บพระ, ตลาดพระ, eBay, pantipmarket (ข้อมูลจำลอง)</p>
                
                <h4>💰 ช่วงราคาซื้อขาย (ย้อนหลัง 3 ปี):</h4>
                <ul style="font-size: 1.1rem; line-height: 1.8;">
                    <li><strong>ต่ำสุด:</strong> {price_info.get('min', 0):,} บาท</li>
                    <li><strong>สูงสุด:</strong> {price_info.get('max', 0):,} บาท</li>
                    <li><strong>ราคาเฉลี่ย:</strong> {price_info.get('avg', 0):,} บาท</li>
                </ul>
                
                <h4>🏛️ ฐานข้อมูลการขาย:</h4>
                <ul style="font-size: 1.1rem; line-height: 1.8;">
                    <li><strong>เว็บพระ (2023):</strong> ปิดประมูลที่ {int(price_info.get('avg', 0) * 0.8):,} บาท</li>
                    <li><strong>ตลาดพระออนไลน์ (2024):</strong> {int(price_info.get('avg', 0) * 1.1):,} บาท</li>
                    <li><strong>eBay (2024):</strong> {int(price_info.get('avg', 0) / 35):,} USD (~{int(price_info.get('avg', 0) * 0.9):,} บาท)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Historical Information
        st.markdown(f"""
        <div class="tips-card">
            <h3>📅 ข้อมูลประวัติศาสตร์</h3>
            <ul style="font-size: 1.2rem; line-height: 1.8;">
                <li><strong>ปีที่สร้าง (ประมาณ):</strong> {amulet_info.get('period', 'ไม่ระบุ')}</li>
                <li><strong>วัด/สถานที่:</strong> {amulet_info.get('temple', 'ไม่ระบุ')}</li>
                <li><strong>คำอธิบาย:</strong> {amulet_info.get('description', 'ไม่มีข้อมูล')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        if show_confidence and confidence > 0:
            st.progress(confidence)
            st.caption(f"ความเชื่อมั่น: {confidence:.2%}")

        st.caption(f"🤖 วิธีการทำนาย: {result.get('method', 'Unknown')}")

    else:
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ เกิดข้อผิดพลาด</h3>
            <p style="font-size: 1.2rem;">{error_msg}</p>
        </div>
        """, unsafe_allow_html=True)

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

    # Header แบบข้อความธรรมดา
    st.title("Amulet-AI")
    st.subheader("ระบบวิเคราะห์พระเครื่อง ด้วย Computer AI อัจฉริยะ")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Default settings (no settings UI)
    analysis_mode = "สองด้าน (หน้า+หลัง)"
    show_confidence = True
    show_probabilities = True

    # Create Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["หน้าหลัก", "เกี่ยวกับระบบ", "คู่มือการใช้งาน", "คำถามที่พบบ่อย"])

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
            <h2 style="margin-bottom: 0.5rem;">� อัปโหลดรูปพระเครื่อง</h2>
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

    # Footer แบบข้อความธรรมดาอยู่กึ่งกลาง
    st.write("")
    st.markdown("---")
    
    # ใช้ columns เพื่อจัดให้อยู่กึ่งกลาง
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h4>🙏 ขอบคุณ</h4>
            <p><strong>สำนักงานส่งเสริมเศรษฐกิจดิจิทัล (depa)</strong></p>
            <p>ที่มอบโอกาสอันทรงคุณค่าให้กับทีม</p>
            <p><strong>Taxes1112 – วิทยาลัยเทคนิคสัตหีบ</strong></p>
            <p>ในการเข้าร่วมโครงการและนำเสนอผลงานด้าน <strong>นวัตกรรมดิจิทัล</strong></p>
        </div>
        """, unsafe_allow_html=True)

def dual_image_mode(show_confidence, show_probabilities):
    """โหมดสองด้าน - ปรับปรุงสำหรับมือถือ"""
    st.markdown("### อัปโหลดรูปทั้งสองด้าน")
    
    # ตรวจสอบว่าเป็นมือถือหรือไม่
    is_mobile = st.checkbox("ใช้งานบนมือถือ (แสดงแบบแนวตั้ง)", value=False)
    
    if is_mobile:
        # โหมดมือถือ - แสดงแนวตั้ง
        st.markdown("#### โหมดมือถือ")
        
        # Camera interface
        create_camera_interface()
        
        st.markdown("---")
        
        # Front image section
        st.markdown("#### รูปด้านหน้า")
        front_image = st.file_uploader(
            "อัปโหลดรูปด้านหน้า", 
            type=['png', 'jpg', 'jpeg'], 
            key="front_upload_mobile"
        )
        
        if front_image:
            st.image(front_image, caption="รูปด้านหน้า", use_container_width=True)
        elif st.session_state.front_camera_image:
            st.image(st.session_state.front_camera_image, caption="รูปด้านหน้า (จากกล้อง)", use_container_width=True)
        
        st.markdown("---")
        
        # Back image section  
        st.markdown("#### รูปด้านหลัง")
        back_image = st.file_uploader(
            "อัปโหลดรูปด้านหลัง", 
            type=['png', 'jpg', 'jpeg'], 
            key="back_upload_mobile"
        )
        
        if back_image:
            st.image(back_image, caption="รูปด้านหลัง", use_container_width=True)
        elif st.session_state.back_camera_image:
            st.image(st.session_state.back_camera_image, caption="รูปด้านหลัง (จากกล้อง)", use_container_width=True)
            
    else:
        # โหมดเดสก์ทอป - แสดงแนวนอน
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
        
        # Camera interface for desktop
        st.markdown("---")
        create_camera_interface()

    st.markdown("---")

    final_front = front_image or st.session_state.front_camera_image
    final_back = back_image or st.session_state.back_camera_image

    if final_front and final_back:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success("มีรูปภาพทั้งสองด้านแล้ว!")

            if st.button("เริ่มการวิเคราะห์ทั้งสองด้าน", type="primary", use_container_width=True):
                with st.spinner("AI กำลังวิเคราะห์ทั้งสองด้าน..."):
                    start_time = time.time()

                    front_result = classify_image(final_front)
                    back_result = classify_image(final_back)

                    processing_time = time.time() - start_time

                    st.success(f"เสร็จสิ้น! ({processing_time:.2f}s)")

                    st.markdown("### ผลการวิเคราะห์")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### ด้านหน้า")
                        display_classification_result_enhanced(front_result, " (ด้านหน้า)", show_confidence)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown("#### ด้านหลัง")
                        display_classification_result_enhanced(back_result, " (ด้านหลัง)", show_confidence)
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
    st.markdown("## คำถามที่พบบ่อย (FAQ)")
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
        **ใช้ได้อย่างเต็มประสิทธิภาพ!** ระบบได้รับการปรับปรุงพิเศษสำหรับมือถือ:
        
        **🔧 ฟีเจอร์ใหม่สำหรับมือถือ:**
        - **โหมดมือถือ** - แสดงผลแบบแนวตั้งที่เหมาะกับหน้าจอมือถือ  
        - **กล้องอัจฉริยะ** - ขออนุญาตกล้องครั้งเดียว ใช้ได้ทั้งกล้องหน้าและหลัง  
        - **ปุ่มใหญ่พิเศษ** - ออกแบบให้กดง่ายบนหน้าจอสัมผัส  
        - **ปรับขนาดอัตโนมัติ** - รูปภาพและส่วนประกอบปรับขนาดให้เหมาะสม  
        
        **📱 วิธีใช้งาน:**  
        1. เปิดเว็บไซต์ผ่านเบราว์เซอร์มือถือ (Chrome, Safari, Edge)  
        2. เลือก ✅ "ใช้งานบนมือถือ" ในหน้าหลัก  
        3. กด "เริ่มใช้กล้อง" เพื่อขออนุญาตกล้อง  
        4. สลับระหว่างกล้องหน้า/หลัง และถ่ายรูปได้เลย!  
        
        **💡 เทคนิคการถ่ายรูปด้วยมือถือ:**  
        - ใช้แสงธรรมชาติหรือแสงขาวสำหรับความชัดเจน  
        - จับมือถือให้มั่นคงเพื่อป้องกันภาพเบลอ  
        - ถ่ายรูปในระยะใกล้เพื่อรายละเอียดที่ชัดเจน  
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
                <li><strong>ใช้ข้อมูลประกอบเท่านั้น</strong> — ไม่ควรเป็นเกณฑ์เดียว</li>
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

    # Enhanced Warning Card - ใช้ error-box class ที่มีอยู่แล้วใน CSS หลัก
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

if __name__ == "__main__":
    main()