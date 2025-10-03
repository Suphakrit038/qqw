#!/usr/bin/env python3
"""
ทดสอบระบบ Amulet AI แบบครบถ้วน
Testing Complete Amulet AI System
"""

import sys
import os
from pathlib import Path
import traceback
import time
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_ai_classifier():
    """ทดสอบ AI Classifier"""
    print("=" * 60)
    print("🤖 ทดสอบ AI Classifier")
    print("=" * 60)
    
    try:
        from ai_models.updated_classifier import get_updated_classifier, check_model_availability
        
        # ตรวจสอบความพร้อมของโมเดล
        print("🔍 ตรวจสอบความพร้อมของโมเดล...")
        availability = check_model_availability()
        print(f"📊 Model Available: {availability['available']}")
        print(f"💬 Message: {availability['message']}")
        
        if availability['available']:
            # ทดสอบการทำนาย
            print("\n🧪 ทดสอบการทำนายด้วยรูปภาพจำลอง...")
            classifier = get_updated_classifier()
            
            # สร้างรูปภาพทดสอบ
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            start_time = time.time()
            result = classifier.predict(test_image)
            processing_time = time.time() - start_time
            
            print(f"⏱️ เวลาประมวลผล: {processing_time:.3f} วินาที")
            
            if result['success']:
                print(f"✅ การทำนายสำเร็จ:")
                print(f"   🎯 คลาสที่ทำนาย: {result['predicted_class']}")
                print(f"   📊 ความมั่นใจ: {result['confidence']:.4f}")
                print(f"   🧮 จำนวนฟีเจอร์: {result['feature_count']}")
                
                print(f"\n📈 Top 3 ความน่าจะเป็น:")
                sorted_probs = sorted(result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                for i, (class_name, prob) in enumerate(sorted_probs):
                    icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    print(f"   {icon} {class_name}: {prob:.4f}")
                
                return True
            else:
                print(f"❌ การทำนายล้มเหลว: {result['error']}")
                return False
        else:
            print("⚠️ โมเดลไม่พร้อมใช้งาน")
            return False
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        traceback.print_exc()
        return False

def test_frontend_components():
    """ทดสอบ Frontend Components"""
    print("\n" + "=" * 60)
    print("🎨 ทดสอบ Frontend Components")
    print("=" * 60)
    
    try:
        # ทดสอบ Analysis Results Component
        print("🧪 ทดสอบ Analysis Results Component...")
        from frontend.components.analysis_results import AnalysisResultsComponent
        
        analysis_component = AnalysisResultsComponent()
        
        # สร้างผลลัพธ์จำลอง
        mock_result = {
            'thai_name': 'พระสมเด็จ',
            'confidence': 0.87,
            'predicted_class': 'somdej',
            'probabilities': {
                'พระสมเด็จ': 0.87,
                'พระนางพญา': 0.08,
                'พระพิมพ์เล็ก': 0.05
            },
            'processing_time': 1.5,
            'analysis_type': 'single_image',
            'enhanced_features': {
                'image_quality': {
                    'overall_score': 0.92,
                    'quality_level': 'excellent',
                    'was_enhanced': True
                },
                'auto_enhanced': True,
                'dual_analysis': False
            }
        }
        
        print("✅ Analysis Results Component โหลดสำเร็จ")
        
        # ทดสอบ components อื่น ๆ
        print("🧪 ทดสอบ Components อื่น ๆ...")
        
        from frontend.components.file_uploader import FileUploaderComponent
        from frontend.components.image_display import ImageDisplayComponent
        from frontend.components.mode_selector import ModeSelectorComponent
        
        print("✅ ทุก Frontend Components โหลดสำเร็จ")
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดใน Frontend Components: {str(e)}")
        traceback.print_exc()
        return False

def test_core_modules():
    """ทดสอบ Core Modules"""
    print("\n" + "=" * 60)
    print("⚙️ ทดสอบ Core Modules")
    print("=" * 60)
    
    try:
        print("🧪 ทดสอบ Error Handling...")
        from core.error_handling_enhanced import error_handler, validate_image_file
        print("✅ Error Handling โหลดสำเร็จ")
        
        print("🧪 ทดสอบ Core Config...")
        from core.config import get_config
        print("✅ Core Config โหลดสำเร็จ")
        
        print("🧪 ทดสอบ Performance Monitoring...")
        from core.performance_monitoring import monitor_performance
        print("✅ Performance Monitoring โหลดสำเร็จ")
        
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดใน Core Modules: {str(e)}")
        traceback.print_exc()
        return False

def test_training_system():
    """ทดสอบระบบเทรน"""
    print("\n" + "=" * 60)
    print("🏋️ ทดสอบระบบเทรน")
    print("=" * 60)
    
    try:
        from trained_model.train_ai_model import AmuletAITrainer
        
        print("✅ Training System โหลดสำเร็จ")
        
        # ตรวจสอบโมเดลที่เทรนแล้ว
        trained_model_path = Path("trained_model")
        if trained_model_path.exists():
            required_files = [
                "classifier.joblib",
                "scaler.joblib", 
                "label_encoder.joblib",
                "pca.joblib",
                "labels.json",
                "training_info.json"
            ]
            
            print("📁 ตรวจสอบไฟล์โมเดลที่เทรนแล้ว:")
            for file_name in required_files:
                file_path = trained_model_path / file_name
                status = "✅" if file_path.exists() else "❌"
                print(f"   {status} {file_name}")
            
            missing_files = [f for f in required_files if not (trained_model_path / f).exists()]
            if not missing_files:
                print("🎉 ไฟล์โมเดลครบถ้วน!")
                return True
            else:
                print(f"⚠️ ไฟล์ที่ขาดหายไป: {missing_files}")
                return False
        else:
            print("⚠️ ไม่พบโฟลเดอร์ trained_model")
            return False
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในระบบเทรน: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """รันการทดสอบทั้งหมด"""
    print("🚀 เริ่มการทดสอบระบบ Amulet AI")
    print("📅 เวลา:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    tests = [
        ("Training System", test_training_system),
        ("AI Classifier", test_ai_classifier),
        ("Core Modules", test_core_modules),
        ("Frontend Components", test_frontend_components),
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n🔄 รันการทดสอบ: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
                print(f"✅ {test_name}: ผ่าน")
            else:
                print(f"❌ {test_name}: ไม่ผ่าน")
        except Exception as e:
            results[test_name] = False
            print(f"💥 {test_name}: เกิดข้อผิดพลาด - {str(e)}")
    
    # สรุปผลการทดสอบ
    print("\n" + "=" * 60)
    print("📊 สรุปผลการทดสอบ")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ ผ่าน" if passed else "❌ ไม่ผ่าน"
        print(f"{status} {test_name}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 อัตราความสำเร็จ: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 ระบบทำงานได้ดี!")
        if success_rate == 100:
            print("💯 ระบบสมบูรณ์แบบ!")
    else:
        print("⚠️ ระบบต้องการการปรับปรุง")
    
    print("\n📝 คำแนะนำ:")
    if not results.get("Training System", False):
        print("- รันการเทรนโมเดลก่อน: python train_ai_model.py")
    if not results.get("AI Classifier", False):
        print("- ตรวจสอบการติดตั้ง dependencies: pip install -r requirements.txt")
    if not results.get("Frontend Components", False):
        print("- ตรวจสอบการติดตั้ง frontend requirements")
    
    print("\n🌐 หากทุกอย่างพร้อม สามารถเริ่มใช้งานได้ที่:")
    print("Frontend: streamlit run frontend/main_app.py")
    print("API: python api/main_api.py")

if __name__ == "__main__":
    main()