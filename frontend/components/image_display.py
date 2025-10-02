"""Image Display Component

จัดการการแสดงผลรูปภาพที่อัพโหลดและข้อมูลที่เกี่ยวข้อง
"""

import streamlit as st
from PIL import Image
from typing import Dict, Any
from ..utils.image_processor import ImagePreprocessor
from core.error_handling import error_logger


class ImageDisplayComponent:
    """Component สำหรับแสดงรูปภาพและข้อมูลคุณภาพ"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
    
    def display_uploaded_image(self, uploaded_file, image_type: str, max_size: int):
        """แสดงรูปภาพที่อัพโหลดพร้อมข้อมูลคุณภาพ"""
        try:
            # Check file size
            file_size = len(self._read_file_bytes(uploaded_file))
            if file_size > max_size:
                st.error(f"ไฟล์{image_type}มีขนาดเกิน {max_size // (1024*1024)}MB กรุณาเลือกไฟล์ใหม่")
                return False
            
            # Load and display image
            image = Image.open(uploaded_file)
            
            # Enhanced image display container
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 15px; 
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin: 1rem 0;">
            """, unsafe_allow_html=True)
            
            st.image(image, caption=f"📷 {uploaded_file.name}", use_column_width=True)
            
            # Display file info and quality assessment
            self._display_image_info(image, uploaded_file.name, file_size)
            
            st.markdown("</div>", unsafe_allow_html=True)
            return True
            
        except Exception as e:
            error_logger.log_error(e, context={"operation": "display_image", "type": image_type})
            st.error(f"ไม่สามารถแสดงรูป{image_type}ได้: {str(e)}")
            return False
    
    def _read_file_bytes(self, uploaded_file):
        """อ่านไฟล์อย่างปลอดภัย"""
        try:
            uploaded_file.seek(0)
            return uploaded_file.read()
        except Exception:
            return b""
        finally:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
    
    def _display_image_info(self, image: Image.Image, filename: str, file_size: int):
        """แสดงข้อมูลรูปภาพและคุณภาพ"""
        file_size_kb = file_size / 1024
        image_dimensions = f"{image.width} × {image.height}"
        image_format = image.format or "Unknown"
        
        # Quick quality assessment
        quality_metrics = self.preprocessor.assess_image_quality(image)
        quality_score = quality_metrics.get('overall_score', 0.5)
        quality_level = quality_metrics.get('quality_level', 'unknown')
        
        # Quality indicator color
        quality_colors = {
            'excellent': '#10b981',
            'good': '#3b82f6', 
            'fair': '#f59e0b',
            'poor': '#ef4444',
            'unknown': '#6b7280'
        }
        quality_color = quality_colors.get(quality_level, '#6b7280')
        
        # Enhanced file info display
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div>
                    <span style="color: #374151; font-weight: 600;">📁 ขนาด:</span>
                    <span style="color: #3b82f6; font-weight: 700;">{file_size_kb:.1f} KB</span>
                </div>
                <div>
                    <span style="color: #374151; font-weight: 600;">📐 ขนาด:</span>
                    <span style="color: #3b82f6; font-weight: 700;">{image_dimensions}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div>
                    <span style="color: #374151; font-weight: 600;">🎨 รูปแบบ:</span>
                    <span style="color: #3b82f6; font-weight: 700;">{image_format}</span>
                </div>
                <div>
                    <span style="color: #374151; font-weight: 600;">⭐ คุณภาพ:</span>
                    <span style="color: {quality_color}; font-weight: 700;">{quality_level.title()} ({quality_score:.1%})</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quality recommendations
        if quality_level == 'poor':
            st.warning("⚠️ คุณภาพภาพต่ำ - อาจส่งผลต่อความแม่นยำของการวิเคราะห์")
        elif quality_level == 'fair':
            st.info("💡 คุณภาพภาพปานกลาง - AI จะปรับปรุงภาพอัตโนมัติ")
        elif quality_level in ['good', 'excellent']:
            st.success("✅ คุณภาพภาพดี - เหมาะสำหรับการวิเคราะห์")