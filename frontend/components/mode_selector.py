"""Mode Selector Component

Component สำหรับเลือกโหมดการวิเคราะห์ (Single/Dual Image)
"""

import streamlit as st
from typing import Optional


class ModeSelectorComponent:
    """Component สำหรับเลือกโหมดการวิเคราะห์"""
    
    def __init__(self):
        self.mode_info = {
            'dual': {
                'title': 'Dual Image Analysis',
                'description': 'วิเคราะห์จากรูปคู่ (หน้า-หลัง) เพื่อความแม่นยำสูงสุด',
                'features': ['แม่นยำสูง', 'วิเคราะห์รายละเอียด', 'ความน่าเชื่อถือสูง'],
                'color': '#10b981'
            }
        }
    
    def display_mode_selector(self) -> Optional[str]:
        """แสดงตัวเลือกโหมดการวิเคราะห์และคืนค่าโหมดที่เลือก"""
        
        st.markdown('<h2 class="section-title">เลือกโหมดการวิเคราะห์</h2>', unsafe_allow_html=True)
        
        dual_mode = st.button(
            "Dual Image Analysis", 
            help="วิเคราะห์จากรูปคู่ (หน้า-หลัง) - แม่นยำสูง",
            use_container_width=True
        )
        
        # Determine selected mode
        selected_mode = None
        if dual_mode:
            selected_mode = 'dual'
        
        return selected_mode
    
    def display_mode_info(self, mode: str):
        """แสดงข้อมูลโหมดที่เลือก"""
        
        if mode not in self.mode_info:
            return
        
        info = self.mode_info[mode]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {info['color']}15 0%, {info['color']}08 100%);
                    border: 1px solid {info['color']}30; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
            <h4 style="color: {info['color']}; margin-top: 0;">{info['title']}</h4>
            <p style="color: #374151; margin: 0.5rem 0;">{info['description']}</p>
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
        """ + "".join([f"<span style='background: {info['color']}20; color: {info['color']}; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.8rem;'>{feature}</span>" for feature in info['features']]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def get_mode_recommendations(self, mode: str) -> str:
        """ให้คำแนะนำเกี่ยวกับโหมดที่เลือก"""
        
        recommendations = {
            'dual': """
            **คำแนะนำสำหรับ Dual Image Analysis:**
            - ให้ความแม่นยำสูงสุดเนื่องจากวิเคราะห์ทั้งสองด้าน
            - แนะนำสำหรับการตรวจสอบที่สำคัญหรือต้องการความแน่ใจสูง
            - ถ่ายรูปทั้งด้านหน้าและด้านหลังในสภาพแสงเดียวกัน
            - เวลาประมวลผลนานขึ้นเล็กน้อย (< 3 วินาที) แต่แม่นยำกว่า
            """
        }
        
        return recommendations.get(mode, "")
    
    def display_mode_comparison(self):
        """แสดงข้อมูลโหมดการวิเคราะห์"""
        
        st.markdown("### เกี่ยวกับโหมดการวิเคราะห์")
        
        st.markdown("""
        **Dual Image Analysis** เป็นโหมดการวิเคราะห์ที่ให้ความแม่นยำสูงสุด
        
        **คุณสมบัติ:**
        - **ความเร็ว:** รวดเร็ว
        - **ความแม่นยำ:** ยอดเยี่ยม (90-95%)
        - **จำนวนรูป:** 2 รูป (หน้า-หลัง)
        - **ความซับซ้อน:** ปานกลาง
        - **แนะนำสำหรับ:** การตรวจสอบสำคัญ
        """)