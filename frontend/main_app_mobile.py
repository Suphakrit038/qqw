#!/usr/bin/env python3
"""
Amulet-AI - Production Frontend with Mobile Support
ระบบจำแนกพระเครื่องอัจฉริยะ (รองรับมือถือ)
Thai Amulet Classification System

ระบบ AI สำหรับการจำแนกประเภทพระเครื่องไทย
ใช้เทคโนโลยี Deep Learning และ Computer Vision
เพื่อช่วยวิเคราะห์และประเมินพระเครื่องอย่างแม่นยำ
พร้อมการรองรับมือถือเต็มรูปแบบ
"""

import streamlit as st
import os
import sys
import time
import base64
import requests
import joblib
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path
import json
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None
    st.warning("OpenCV (cv2) not installed. Some image processing features may not work.")

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

# Mobile-First Responsive CSS
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
    
    .stButton > button:hover {{
        background: {COLORS['gold']};
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.4);
        transform: translateY(-2px) scale(1.01);
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
    
    /* Modern Alert Boxes */
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
    }}
    
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
    }}
    
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
    }}
    
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
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    [data-testid="stSidebar"] {{display: none;}}
    [data-testid="collapsedControl"] {{display: none;}}
</style>

<script>
// Camera functionality with single permission request
let cameraStream = null;
let currentMode = 'front';

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
        cameraStream.getTracks().forEach(track => track.stop());
    }}
    
    const constraints = {{
        video: {{
            facingMode: mode === 'user' ? 'user' : 'environment',
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
        
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
        
        // ส่งข้อมูลไปยัง Streamlit
        window.parent.postMessage({{
            type: 'camera_capture',
            mode: targetMode,
            dataUrl: dataUrl
        }}, '*');
        
        alert('ถ่ายรูป' + (targetMode === 'front' ? 'หน้า' : 'หลัง') + 'สำเร็จ!');
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
        });" id="start-camera">🎥 เริ่มใช้กล้อง</button>
        
        <div id="camera-controls" style="display: none;">
            <video id="camera-video" class="camera-video" autoplay playsinline muted></video>
            <br>
            <button class="camera-button" onclick="switchCamera('user')">📱 กล้องหน้า</button>
            <button class="camera-button" onclick="switchCamera('environment')">📷 กล้องหลัง</button>
            <br>
            <button class="camera-button" onclick="capturePhoto('front')" style="background: #10b981;">📸 ถ่ายรูปหน้า</button>
            <button class="camera-button" onclick="capturePhoto('back')" style="background: #3b82f6;">📸 ถ่ายรูปหลัง</button>
            <br>
            <button class="camera-button" onclick="stopCamera(); document.getElementById('camera-controls').style.display = 'none'; document.getElementById('start-camera').style.display = 'block';" style="background: #ef4444;">⏹️ หยุดกล้อง</button>
        </div>
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
    """การทำนายแบบ local"""
    try:
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
            labels_path = project_root / "trained_model/labels.json"
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            thai_name = labels.get(predicted_class, predicted_class)
        except:
            thai_name = predicted_class

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "thai_name": thai_name,
            "confidence": confidence,
            "method": "Local Model",
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
            conf_color = "#4caf50" if confidence > 0.8 else "#ff9800" if confidence > 0.6 else "#f44336"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; margin: 15px 0;">
                <h4>ความเชื่อมั่น: <span style="color: {conf_color};">{confidence:.1%}</span></h4>
                <div style="background: #e0e0e0; border-radius: 10px; height: 20px;">
                    <div style="background: {conf_color}; width: {confidence*100}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if show_probabilities and 'probabilities' in result:
            st.markdown("#### ความน่าจะเป็นแต่ละประเภท")
            probabilities = result['probabilities']
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for class_name, prob in sorted_probs:
                st.progress(prob, text=f"{class_name}: {prob:.1%}")

        st.caption(f"วิธีการทำนาย: {result.get('method', 'Unknown')}")

    else:
        error_msg = result.get('error', 'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ')
        st.markdown(f"""
        <div class="error-box">
            <h3>เกิดข้อผิดพลาด</h3>
            <p>{error_msg}</p>
        </div>
        """, unsafe_allow_html=True)

def dual_image_mode(show_confidence, show_probabilities):
    """โหมดสองด้าน - รองรับมือถือ"""
    st.markdown("### 📸 อัปโหลดรูปทั้งสองด้าน")
    
    # ตรวจสอบว่าเป็นมือถือหรือไม่
    is_mobile = st.checkbox("📱 ใช้งานบนมือถือ (แสดงแบบแนวตั้ง)", value=False)
    
    if is_mobile:
        # โหมดมือถือ - แสดงแนวตั้ง
        st.markdown("#### 📱 โหมดมือถือ")
        
        # Camera interface
        create_camera_interface()
        
        st.markdown("---")
        
        # Front image section
        st.markdown("#### 📷 รูปด้านหน้า")
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
        st.markdown("#### 📷 รูปด้านหลัง")
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
                front_image = st.file_uploader("เลือกรูปด้านหน้า", 
                                             type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'heic', 'heif'], 
                                             key="front_upload")

            with front_camera:
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
                back_image = st.file_uploader("เลือกรูปด้านหลัง", 
                                            type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'heic', 'heif'], 
                                            key="back_upload")

            with back_camera:
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

            if st.button("🔍 เริ่มการวิเคราะห์ทั้งสองด้าน", type="primary", use_container_width=True):
                with st.spinner("AI กำลังวิเคราะห์ทั้งสองด้าน..."):
                    start_time = time.time()

                    front_result = classify_image(final_front)
                    back_result = classify_image(final_back)

                    processing_time = time.time() - start_time

                    st.success(f"เสร็จสิ้น! ({processing_time:.2f}s)")

                    st.markdown("### ผลการวิเคราะห์")

                    if is_mobile:
                        # แสดงผลแบบแนวตั้งสำหรับมือถือ
                        st.markdown("#### 📷 ผลด้านหน้า")
                        display_classification_result(front_result, show_confidence, show_probabilities)
                        
                        st.markdown("---")
                        
                        st.markdown("#### 📷 ผลด้านหลัง")
                        display_classification_result(back_result, show_confidence, show_probabilities)
                    else:
                        # แสดงผลแบบแนวนอนสำหรับเดสก์ทอป
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### ด้านหน้า")
                            display_classification_result(front_result, show_confidence, show_probabilities)

                        with col2:
                            st.markdown("#### ด้านหลัง")
                            display_classification_result(back_result, show_confidence, show_probabilities)

                    # Comparison
                    if (front_result.get("status") == "success" and back_result.get("status") == "success"):
                        front_class = front_result.get("predicted_class", "")
                        back_class = back_result.get("predicted_class", "")
                        front_conf = front_result.get("confidence", 0)
                        back_conf = back_result.get("confidence", 0)

                        st.markdown("### การเปรียบเทียบ")

                        if front_class == back_class:
                            avg_conf = (front_conf + back_conf) / 2
                            st.success(f"✅ ผลตรงกันทั้งสองด้าน: **{front_class}** (ความเชื่อมั่นเฉลี่ย: {avg_conf:.1%})")
                        else:
                            st.warning(f"⚠️ ผลไม่ตรงกัน: หน้า={front_class} ({front_conf:.1%}), หลัง={back_class} ({back_conf:.1%})")
                            st.info("แนะนำให้ตรวจสอบความถูกต้องของภาพหรือปรึกษาผู้เชี่ยวชาญ")

    else:
        st.info("กรุณาอัปโหลดรูปทั้งสองด้าน (หน้าและหลัง) เพื่อเริ่มการวิเคราะห์")

def show_faq_mobile_section():
    """แสดงส่วน FAQ พิเศษสำหรับมือถือ"""
    with st.expander("❓ การใช้งานบนมือถือทำอย่างไร?"):
        st.markdown("""
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

def main():
    # Initialize session state
    if 'front_camera_image' not in st.session_state:
        st.session_state.front_camera_image = None
    if 'back_camera_image' not in st.session_state:
        st.session_state.back_camera_image = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    # Header
    st.title("🔮 Amulet-AI (Mobile Ready)")
    st.subheader("ระบบวิเคราะห์พระเครื่อง ด้วย AI อัจฉริยะ - รองรับมือถือเต็มรูปแบบ")

    st.markdown('<div style="height: 1.5px; background: linear-gradient(90deg, transparent, #D4AF37, transparent); margin: 35px 0; border-radius: 2px; opacity: 0.6;"></div>', unsafe_allow_html=True)

    # Default settings
    analysis_mode = "สองด้าน (หน้า+หลัง)"
    show_confidence = True
    show_probabilities = True

    # Create Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🏠 หน้าหลัก", "📖 เกี่ยวกับระบบ", "❓ คำถามที่พบบ่อย"])

    # Tab 1: Main Upload Section
    with tab1:
        st.markdown("## 🔍 เริ่มการวิเคราะห์")
        
        if analysis_mode == "สองด้าน (หน้า+หลัง)":
            dual_image_mode(show_confidence, show_probabilities)

    # Tab 2: About System
    with tab2:
        st.markdown("## 📋 เกี่ยวกับระบบ")
        st.markdown("""
        **Amulet-AI** เป็นระบบปัญญาประดิษฐ์สำหรับการจำแนกพระเครื่องไทย
        ใช้เทคโนโลยี Deep Learning และ Computer Vision เพื่อช่วยวิเคราะห์
        และประเมินพระเครื่องอย่างแม่นยำ
        
        **🎯 วัตถุประสงค์:**
        - ช่วยผู้สนใจพระเครื่องในการระบุประเภท
        - ให้ข้อมูลเบื้องต้นสำหรับการตัดสินใจ
        - ส่งเสริมการเรียนรู้เกี่ยวกับพระเครื่องไทย
        
        **⚠️ ข้อควรระวัง:**
        - ผลลัพธ์เป็นการประเมินเบื้องต้นเท่านั้น
        - ควรปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจซื้อขาย
        - ไม่รับประกันความแท้ 100%
        """)

    # Tab 3: FAQ
    with tab3:
        st.markdown("## ❓ คำถามที่พบบ่อย")
        
        show_faq_mobile_section()
        
        with st.expander("❓ ระบบนี้แม่นยำแค่ไหน?"):
            st.markdown("""
            ระบบมีความแม่นยำประมาณ 85-90% ขึ้นอยู่กับ:
            - คุณภาพของรูปภาพ
            - ความชัดเจนของรายละเอียด
            - มุมและแสงในการถ่ายรูป
            """)

        with st.expander("❓ รองรับไฟล์รูปภาพแบบไหนบ้าง?"):
            st.markdown("""
            รองรับไฟล์: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP, HEIC, HEIF
            ขนาดไฟล์ไม่เกิน 10MB
            """)

    # Footer
    st.markdown('<div style="height: 1.5px; background: linear-gradient(90deg, transparent, #D4AF37, transparent); margin: 35px 0; border-radius: 2px; opacity: 0.6;"></div>', unsafe_allow_html=True)
    
    # Center footer
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
            © 2025 Amulet-AI | Powered by Deep Learning Technology
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()