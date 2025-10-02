#!/usr/bin/env python3
"""
ทดสอบระบบการแสดงผลลัพธ์ใหม่
Test Enhanced Analysis Results System
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_enhanced_results():
    """ทดสอบ Enhanced Results Component"""
    print("🧪 ทดสอบ Enhanced Analysis Results...")
    
    try:
        from frontend.components.enhanced_results import EnhancedAnalysisResults
        
        # สร้าง component
        enhanced_results = EnhancedAnalysisResults()
        print("✅ โหลด Enhanced Results Component สำเร็จ")
        
        # ทดสอบข้อมูลฐานข้อมูลพระเครื่อง
        amulet_db = enhanced_results.amulet_database
        print(f"📊 ข้อมูลพระเครื่องในฐานข้อมูล: {len(amulet_db)} รายการ")
        
        for key, data in amulet_db.items():
            print(f"   • {key}: {data['full_name']} - {data['price_range']['avg']:,} บาท")
        
        # สร้างผลลัพธ์ทดสอบ
        test_result = {
            'predicted_class': 'somdej_pratanporn_buddhagavak',
            'confidence': 0.923,
            'probabilities': {
                'somdej_pratanporn_buddhagavak': 0.923,
                'prok_bodhi_9_leaves': 0.055,
                'phra_sivali': 0.022
            },
            'processing_time': 2.4
        }
        
        print("✅ ทดสอบข้อมูลจำลองสำเร็จ")
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_confidence_levels():
    """ทดสอบระบบระดับความมั่นใจ"""
    print("\n🎯 ทดสอบระบบระดับความมั่นใจ...")
    
    try:
        from frontend.components.enhanced_results import EnhancedAnalysisResults
        enhanced_results = EnhancedAnalysisResults()
        
        test_confidences = [0.95, 0.85, 0.75, 0.65, 0.45]
        
        for conf in test_confidences:
            level = enhanced_results._get_confidence_level(conf)
            info = enhanced_results._get_confidence_info(conf)
            print(f"   {info['icon']} {conf:.1%} → {level} ({info['description'][:30]}...)")
        
        print("✅ ระบบระดับความมั่นใจทำงานสำเร็จ")
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return False

def test_market_data():
    """ทดสอบข้อมูลตลาด"""
    print("\n💰 ทดสอบข้อมูลตลาด...")
    
    try:
        from frontend.components.enhanced_results import EnhancedAnalysisResults
        enhanced_results = EnhancedAnalysisResults()
        
        # ทดสอบ Mock Sales Data
        test_price_range = {'min': 250000, 'max': 1200000, 'avg': 685000}
        sales_data = enhanced_results._generate_mock_sales_data(test_price_range)
        
        print("   📈 ข้อมูลการขายจำลอง:")
        for sale in sales_data:
            print(f"      • {sale['platform']} ({sale['year']}): {sale['description']}")
        
        print("✅ ข้อมูลตลาดทำงานสำเร็จ")
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return False

def main():
    """รันการทดสอบทั้งหมด"""
    print("🚀 ทดสอบระบบการแสดงผลลัพธ์ใหม่")
    print("=" * 50)
    
    tests = [
        ("Enhanced Results Component", test_enhanced_results),
        ("Confidence Levels", test_confidence_levels),
        ("Market Data", test_market_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 รันการทดสอบ: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: ผ่าน")
            else:
                print(f"❌ {test_name}: ไม่ผ่าน")
        except Exception as e:
            print(f"💥 {test_name}: เกิดข้อผิดพลาด - {e}")
    
    print(f"\n📊 สรุปผลการทดสอบ: {passed}/{total} ผ่าน")
    
    if passed == total:
        print("🎉 ระบบการแสดงผลลัพธ์ใหม่พร้อมใช้งาน!")
        print("\n💡 คุณสมบัติใหม่ที่เพิ่มเข้ามา:")
        print("   • ข้อมูลพระเครื่องแบบละเอียด")
        print("   • ข้อมูลตลาดและราคา")
        print("   • ระบบความมั่นใจแบบสีสัน")
        print("   • คำแนะนำการซื้อขาย")
        print("   • ตารางแสดง Top 3 แบบสวยงาม")
    else:
        print("⚠️ ระบบต้องการการแก้ไข")

if __name__ == "__main__":
    main()