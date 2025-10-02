"""File Validator Utility

ตรวจสอบไฟล์ที่อัพโหลดว่าเป็นไปตามข้อกำหนด
"""

import streamlit as st
from PIL import Image
from typing import List, Dict, Any
import io


class FileValidator:
    """คลาสสำหรับตรวจสอบไฟล์ที่อัพโหลด"""
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):
        self.max_file_size = max_file_size
        self.accepted_types = ['png', 'jpg', 'jpeg']
        self.min_dimensions = (100, 100)  # ขนาดต่ำสุด
        self.max_dimensions = (4000, 4000)  # ขนาดสูงสุด
    
    def validate_file(self, uploaded_file) -> bool:
        """ตรวจสอบไฟล์ที่อัพโหลดครบถ้วน"""
        
        try:
            # ตรวจสอบขนาดไฟล์
            if not self._validate_file_size(uploaded_file):
                return False
            
            # ตรวจสอบรูปแบบไฟล์
            if not self._validate_file_format(uploaded_file):
                return False
            
            # ตรวจสอบความถูกต้องของภาพ
            if not self._validate_image_integrity(uploaded_file):
                return False
            
            # ตรวจสอบขนาดภาพ
            if not self._validate_image_dimensions(uploaded_file):
                return False
            
            return True
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการตรวจสอบไฟล์: {str(e)}")
            return False
    
    def _validate_file_size(self, uploaded_file) -> bool:
        """ตรวจสอบขนาดไฟล์"""
        try:
            # อ่านข้อมูลไฟล์
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # รีเซ็ตตำแหน่ง
            
            file_size = len(file_bytes)
            
            if file_size > self.max_file_size:
                st.error(f"ไฟล์มีขนาดเกิน {self.max_file_size // (1024*1024)}MB กรุณาเลือกไฟล์ใหม่")
                return False
            
            if file_size < 1024:  # น้อยกว่า 1KB
                st.error("ไฟล์มีขนาดเล็กเกินไป กรุณาเลือกไฟล์ใหม่")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"ไม่สามารถตรวจสอบขนาดไฟล์ได้: {str(e)}")
            return False
    
    def _validate_file_format(self, uploaded_file) -> bool:
        """ตรวจสอบรูปแบบไฟล์"""
        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension not in self.accepted_types:
                st.error(f"ไฟล์ต้องเป็นรูปแบบ: {', '.join(self.accepted_types).upper()}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"ไม่สามารถตรวจสอบรูปแบบไฟล์ได้: {str(e)}")
            return False
    
    def _validate_image_integrity(self, uploaded_file) -> bool:
        """ตรวจสอบความถูกต้องของภาพ"""
        try:
            # พยายามเปิดภาพด้วย PIL
            image = Image.open(uploaded_file)
            
            # ตรวจสอบว่าเปิดได้จริง
            image.verify()
            
            # รีเซ็ตไฟล์หลังจาก verify
            uploaded_file.seek(0)
            
            return True
            
        except Exception as e:
            st.error(f"ไฟล์รูปภาพเสียหาย หรือไม่ใช่รูปภาพที่ถูกต้อง: {str(e)}")
            return False
    
    def _validate_image_dimensions(self, uploaded_file) -> bool:
        """ตรวจสอบขนาดของภาพ"""
        try:
            image = Image.open(uploaded_file)
            width, height = image.size
            
            # ตรวจสอบขนาดต่ำสุด
            if width < self.min_dimensions[0] or height < self.min_dimensions[1]:
                st.error(f"ขนาดภาพต้องไม่น้อยกว่า {self.min_dimensions[0]}x{self.min_dimensions[1]} pixels")
                return False
            
            # ตรวจสอบขนาดสูงสุด
            if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
                st.warning(f"ขนาดภาพใหญ่มาก ({width}x{height}) อาจทำให้ประมวลผลช้า")
                # ไม่ return False เพราะยังใช้งานได้
            
            uploaded_file.seek(0)  # รีเซ็ตตำแหน่ง
            return True
            
        except Exception as e:
            st.error(f"ไม่สามารถตรวจสอบขนาดภาพได้: {str(e)}")
            return False
    
    def get_file_info(self, uploaded_file) -> Dict[str, Any]:
        """ดึงข้อมูลไฟล์"""
        try:
            # ข้อมูลพื้นฐาน
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            file_info = {
                'filename': uploaded_file.name,
                'size_bytes': len(file_bytes),
                'size_kb': len(file_bytes) / 1024,
                'size_mb': len(file_bytes) / (1024 * 1024)
            }
            
            # ข้อมูลภาพ
            try:
                image = Image.open(uploaded_file)
                file_info.update({
                    'width': image.width,
                    'height': image.height,
                    'format': image.format,
                    'mode': image.mode,
                    'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
                })
                uploaded_file.seek(0)
            except Exception:
                pass
            
            return file_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def suggest_optimization(self, uploaded_file) -> List[str]:
        """แนะนำการปรับปรุงไฟล์"""
        suggestions = []
        
        try:
            file_info = self.get_file_info(uploaded_file)
            
            # แนะนำเรื่องขนาดไฟล์
            if file_info['size_mb'] > 5:
                suggestions.append("💡 ลดขนาดไฟล์โดยบีบอัดภาพหรือลดความละเอียด")
            
            # แนะนำเรื่องขนาดภาพ
            if 'width' in file_info and 'height' in file_info:
                if file_info['width'] > 2000 or file_info['height'] > 2000:
                    suggestions.append("💡 ลดความละเอียดภาพเป็น 1000x1000 pixels เพื่อความเร็วในการประมวลผล")
                
                if file_info['width'] < 300 or file_info['height'] < 300:
                    suggestions.append("⚠️ ความละเอียดต่ำ อาจส่งผลต่อความแม่นยำของการวิเคราะห์")
            
            # แนะนำเรื่องรูปแบบไฟล์
            if 'format' in file_info:
                if file_info['format'] == 'PNG' and not file_info.get('has_transparency', False):
                    suggestions.append("💡 แปลงเป็น JPEG เพื่อลดขนาดไฟล์ (เนื่องจากไม่มีพื้นหลังโปร่งใส)")
            
            return suggestions
            
        except Exception:
            return ["ไม่สามารถวิเคราะห์ไฟล์เพื่อให้คำแนะนำได้"]