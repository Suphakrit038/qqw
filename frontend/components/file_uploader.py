"""File Uploader Component

Component สำหรับอัพโหลดไฟล์รูปภาพ
"""

import streamlit as st
from typing import List, Optional, Tuple
from ..utils.file_validator import FileValidator


class FileUploaderComponent:
    """Component สำหรับจัดการการอัพโหลดไฟล์"""
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):
        self.max_file_size = max_file_size
        self.accepted_types = ['png', 'jpg', 'jpeg']
        self.validator = FileValidator(max_file_size)
    
    def dual_image_uploader(self) -> Tuple[Optional[st.runtime.uploaded_file_manager.UploadedFile], Optional[st.runtime.uploaded_file_manager.UploadedFile]]:
        """File uploader สำหรับรูปคู่ (หน้า-หลัง) พร้อมฟังก์ชั่นถ่ายรูป"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### รูปด้านหน้า")
            
            # Tab สำหรับเลือกวิธีการ
            tab1, tab2 = st.tabs(["อัปโหลดไฟล์", "ถ่ายรูป"])
            
            front_image = None
            
            with tab1:
                front_image = st.file_uploader(
                    "เลือกรูปด้านหน้า",
                    type=self.accepted_types,
                    key="front_image_file",
                    help=f"รองรับไฟล์: {', '.join(self.accepted_types).upper()}"
                )
            
            with tab2:
                st.write("ใช้กล้องอุปกรณ์เพื่อถ่ายรูปโดยตรง")
                
                # ปุ่มเปิดกล้อง
                if st.button("เปิดกล้องถ่ายด้านหน้า", key="open_front_camera"):
                    st.session_state.show_front_camera = True
                
                # แสดงกล้องเมื่อผู้ใช้กดปุ่ม
                if st.session_state.get('show_front_camera', False):
                    front_camera = st.camera_input(
                        "ถ่ายรูปด้านหน้า",
                        key="front_camera",
                        help="ใช้กล้องอุปกรณ์เพื่อถ่ายรูปโดยตรง"
                    )
                    if front_camera:
                        front_image = front_camera
                        st.session_state.show_front_camera = False  # ปิดกล้องหลังถ่ายเสร็จ
            
            front_valid = None
            if front_image:
                if self.validator.validate_file(front_image):
                    front_valid = front_image
                else:
                    st.error("ไฟล์ด้านหน้าไม่ผ่านการตรวจสอบ")
        
        with col2:
            st.markdown("#### รูปด้านหลัง")
            
            # Tab สำหรับเลือกวิธีการ
            tab3, tab4 = st.tabs(["อัปโหลดไฟล์", "ถ่ายรูป"])
            
            back_image = None
            
            with tab3:
                back_image = st.file_uploader(
                    "เลือกรูปด้านหลัง",
                    type=self.accepted_types,
                    key="back_image_file", 
                    help=f"รองรับไฟล์: {', '.join(self.accepted_types).upper()}"
                )
            
            with tab4:
                st.write("ใช้กล้องอุปกรณ์เพื่อถ่ายรูปโดยตรง")
                
                # ปุ่มเปิดกล้อง
                if st.button("เปิดกล้องถ่ายด้านหลัง", key="open_back_camera"):
                    st.session_state.show_back_camera = True
                
                # แสดงกล้องเมื่อผู้ใช้กดปุ่ม
                if st.session_state.get('show_back_camera', False):
                    back_camera = st.camera_input(
                        "ถ่ายรูปด้านหลัง",
                        key="back_camera",
                        help="ใช้กล้องอุปกรณ์เพื่อถ่ายรูปโดยตรง"
                    )
                    if back_camera:
                        back_image = back_camera
                        st.session_state.show_back_camera = False  # ปิดกล้องหลังถ่ายเสร็จ
            
            back_valid = None
            if back_image:
                if self.validator.validate_file(back_image):
                    back_valid = back_image
                else:
                    st.error("ไฟล์ด้านหลังไม่ผ่านการตรวจสอบ")
        
        return front_valid, back_valid
    
    def display_upload_guidelines(self):
        """แสดงคำแนะนำการอัพโหลดไฟล์และการถ่ายรูป"""
        
        with st.expander("คำแนะนำการอัพโหลดรูปภาพ", expanded=False):
            st.markdown("""
            ### เทคนิคการถ่ายรูปและอัปโหลดที่ดี
            
            #### ✅ ควรทำ:
            - **แสงสว่าง**: ถ่ายในที่มีแสงสว่างเพียงพอ หลีกเลี่ยงแสงแรงจ้า
            - **ความชัด**: รูปภาพต้องชัดเจน ไม่เบลอ โฟกัสที่พระเครื่อง
            - **มุมมอง**: ถ่ายให้เห็นพระเครื่องทั้งองค์ ไม่ถูกบดบัง
            - **พื้นหลัง**: ใช้พื้นหลังสีเรียบ ไม่มีลวดลายรบกวน
            - **ระยะห่าง**: ถ่ายให้พระเครื่องเต็มเฟรม แต่ไม่ใกล้เกินไป
            
            #### ❌ ไม่ควรทำ:
            - ถ่ายในที่มืดหรือแสงน้อย
            - ใช้แฟลชที่ทำให้เกิดแสงสะท้อน
            - ถ่ายเอียงหรือไม่ตรง
            - มีนิ้วมือหรือสิ่งกีดขวางในภาพ
            
            ### วิธีการอัปโหลด:
            1. **อัปโหลดไฟล์**: เลือกรูปจากอุปกรณ์ของคุณ
            2. **ถ่ายรูป**: ใช้กล้องอุปกรณ์ถ่ายรูปโดยตรง
            
            ### ข้อกำหนดไฟล์:
            - **รูปแบบ**: JPG, JPEG, PNG
            - **ขนาดไฟล์**: ไม่เกิน 10MB
            - **ความละเอียด**: อย่างน้อย 300x300 พิกเซล
            """)
    
    def display_upload_tips(self, mode: str):
        """แสดงเคล็ดลับเฉพาะสำหรับแต่ละโหมด"""
        
        if mode == 'dual':
            st.info("""
            **เคล็ดลับสำหรับ Dual Image Analysis:**
            - ถ่ายรูปทั้งสองด้านในสภาพแสงเดียวกัน
            - วางพระเครื่องในตำแหน่งเดียวกันทั้งสองด้าน
            - ใช้พื้นหลังสีเดียวกันสำหรับทั้งสองรูป
            - ตรวจสอบให้แน่ใจว่ารูปทั้งสองชัดเจน
            """)
    
    def display_camera_tips(self):
        """แสดงเคล็ดลับการใช้กล้อง"""
        st.markdown("""
        <div style="background: #e3f2fd; padding: 1rem; border-left: 4px solid #2196f3; margin: 1rem 0;">
            <h4 style="color: #1976d2; margin-top: 0;">เคล็ดลับการถ่ายรูปด้วยกล้อง:</h4>
            <ul style="margin-bottom: 0;">
                <li>ถืออุปกรณ์ให้มั่นคงเพื่อความชัด</li>
                <li>จัดให้พระเครื่องอยู่กึ่งกลางเฟรม</li>
                <li>รอให้กล้องโฟกัสก่อนกดถ่าย</li>
                <li>ถ่ายในแสงธรรมชาติจะได้ผลดีที่สุด</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def display_upload_tips(self, mode: str):
        """แสดงเคล็ดลับเฉพาะสำหรับแต่ละโหมด"""
        
        if mode == 'dual':
            st.info("""
            **เคล็ดลับสำหรับ Dual Image Analysis:**
            - ถ่ายในสภาพแสงเดียวกันและพื้นหลังเดียวกัน
            - วางพระเครื่องในตำแหน่งเดียวกันสำหรับทั้งสองด้าน
            - ตรวจสอบให้ทั้งสองรูปมีความชัดเจนเท่าเทียมกัน
            - หากมีข้อความหรือตัวเลข ให้แน่ใจว่าเห็นได้ชัดในทั้งสองด้าน
            """)
    
    def get_upload_status_message(self, front_file, back_file=None, mode='dual'):
        """ส่งคืนข้อความสถานะการอัพโหลด"""
        
        if mode == 'dual':
            if front_file and back_file:
                return "อัพโหลดรูปทั้งสองด้านสำเร็จ - พร้อมวิเคราะห์แบบ Dual-View"
            elif front_file or back_file:
                return "กรุณาอัพโหลดรูปทั้งด้านหน้าและด้านหลัง"
            else:
                return "กรุณาเลือกรูปด้านหน้าและด้านหลัง"
        
        return ""