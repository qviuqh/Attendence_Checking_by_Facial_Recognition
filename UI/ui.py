import os
import cv2
import pandas as pd
import time
import json
import concurrent.futures

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QHBoxLayout, QProgressBar, QMessageBox, QSizePolicy,
    QSplitter, QListWidget, QListWidgetItem, QInputDialog, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon, QMovie
from PyQt5.QtCore import Qt, QTimer, QSize

from api_client import APIClient
# from dialogs import StudentRegistrationDialog
from face_detector import FaceDetector
from attendance_manager import AttendanceManager

from src.feature_engineering import FaceEmbedding
from src.preprocessing import Processing_Img

class FaceAttendanceUI(QWidget):
    def __init__(self):
        super().__init__()
        
        self.log = pd.DataFrame(columns=["student_id", "student_name", "timestamp", "status"])
        
        self.pause_api = False
        self.API = APIClient("http://127.0.0.1:8000/")  # Local host
        
        try:
            self.students = self.API.load_json_data()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi tải dữ liệu", str(e))
            self.students = {}
        
        base_students = {
            stu_id: {"name": stu_name, "id": stu_id, "present": False}
            for stu_id, stu_name in self.students.items()
        }
        
        self.current_student_id = None
        self.attendance = AttendanceManager(base_students)
        # Initialize modules
        self.detector = FaceDetector()
        self.embedder = FaceEmbedding()
        
        # Handle multi threading
        self._last_embed_time = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future = None
        
        # Camera & timers
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.face_detected = False

        # UI state
        self.new_student_info = None
        self.attendance_list_visible = False
        self.base_font_size = 12

        self.show_idle_animation()
        
        self.apply_color_scheme()
        self.init_ui()

    def apply_color_scheme(self):
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#F5F7FA"))
        palette.setColor(QPalette.WindowText, QColor("#212529"))
        self.setPalette(palette)
        self.setStyleSheet("""
            QWidget { background-color: #F5F7FA; color: #212529; }
            QPushButton { background-color: #007BFF; color: white; padding: 10px; border-radius: 5px; border: none; min-height: 40px; }
            QPushButton:hover { background-color: #0069D9; }
            QPushButton:disabled { background-color: #6C757D; color: #E9ECEF; }
            QLabel { color: #212529; }
            QProgressBar { border: 1px solid #6C757D; border-radius: 5px; text-align: center; background-color: #E9ECEF; }
            QProgressBar::chunk { background-color: #007BFF; border-radius: 5px; }
            QListWidget { border: 1px solid #6C757D; border-radius: 5px; background-color: white; padding: 5px; }
            QListWidget::item { border-bottom: 1px solid #E9ECEF; padding: 5px; }
            QListWidget::item:selected { background-color: #E9ECEF; color: #212529; }
            QSplitter::handle { background-color: #6C757D; width: 2px; }
        """)


    def init_ui(self):
        self.setWindowTitle("Face Recognition Attendance System")
        self.setMinimumSize(800, 800) 

        # Main layout with minimal margins to maximize space
        self.main_layout = QHBoxLayout()
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create the overall layout
        self.root_layout = QVBoxLayout()
        self.root_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        self.root_layout.setSpacing(0)  # No spacing
        self.root_layout.addLayout(self.main_layout)

        # Create back button with no background, border, or shape — icon only
        self.back_btn = QPushButton(self)
        self.back_btn.setIcon(QIcon("icon/back_btn.png"))  # Use your PNG icon here
        self.back_btn.setIconSize(QSize(28, 28))  # Set icon size as needed
        self.back_btn.setFixedSize(36, 36)# Slightly larger than icon for click area
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);  /* Optional hover effect */
            }
        """)
        self.back_btn.move(27, 20)

        # Left side (camera) layout
        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignCenter)
        self.left_layout.setSpacing(20)
        
        # Right side (attendance list) layout
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)
        self.right_layout.setSpacing(10)
        
        # Create a splitter widget to allow resizing between camera and list
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left side widget
        self.left_widget = QWidget()
        self.left_widget.setLayout(self.left_layout)
        
        # Right side widget
        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_layout)
        self.right_widget.setVisible(False)  # Initially hidden
        
        # Add widgets to splitter
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([600, 400])  # Initial sizes
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.splitter)

        # Title label
        self.title_label = QLabel("Face Recognition Attendance System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.title_label.setWordWrap(True)  # Allow word wrapping for title
        self.title_label.setStyleSheet("color: #007BFF; font-weight: bold;")
        self.left_layout.addWidget(self.title_label)

        # Create a container for camera preview and footer
        self.camera_container = QWidget()
        self.camera_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_container.setMinimumSize(300, 300)
        self.camera_container.setStyleSheet("border: none; border-radius: 8px; background-color: white;")
        
        # Create a layout for the container
        self.camera_layout = QVBoxLayout(self.camera_container)
        self.camera_layout.setContentsMargins(5, 5, 5, 5)
        self.camera_layout.setSpacing(0)
        
        # Camera preview area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_layout.addWidget(self.image_label, stretch=1)
        
        # Add "Made by ITEA" footer inside the camera container
        self.footer_label = QLabel("Made by ITEA")
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setStyleSheet("""
            color: #6C757D; 
            font-weight: 500;
            padding: 5px 0;
            margin: 0;
            font-size: 14px;
            background-color: transparent;
        """)
        self.camera_layout.addWidget(self.footer_label)
        
        # Add the container to the main layout
        self.left_layout.addWidget(self.camera_container, stretch=1)

        # Status label with styled appearance
        self.status_label = QLabel("Welcome to Face Recognition Attendance System")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status_label.setStyleSheet("color: #6C757D; font-weight: bold; padding: 8px; margin: 5px 0;")
        # Enable text elision (truncation with ellipsis)
        self.status_label.setTextFormat(Qt.PlainText)
        self.left_layout.addWidget(self.status_label)

        # Progress bar with themed colors
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #6C757D;
                border-radius: 5px;
                text-align: center;
                background-color: #E9ECEF;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #007BFF;
                border-radius: 5px;
            }
        """)
        self.left_layout.addWidget(self.progress_bar)

        # Buttons layout
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(15)

        # Create all buttons with themed styles
        self.take_attendance_btn = QPushButton("Take Attendance")
        self.register_face_btn = QPushButton("Register New Face")
        self.confirm_face_btn = QPushButton("Confirm Face")
        self.retry_face_btn = QPushButton("Retry Wrong Face")
        
        # New button to toggle attendance list
        self.show_attendance_list_btn = QPushButton("Show Attendance List")
        # Special styling for show attendance button
        self.show_attendance_list_btn.setStyleSheet("""
            QPushButton {
                background-color: #6C757D;
                color: white;
                padding: 10px;
                border-radius: 5px;
                border: none;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #5A6268;
            }
        """)

        # Special styling for confirmation button
        self.confirm_face_btn.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                padding: 10px;
                border-radius: 5px;
                border: none;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)

        # Configure buttons
        buttons = [
            self.take_attendance_btn,
            self.register_face_btn,
            self.confirm_face_btn,
            self.retry_face_btn
        ]
        
        for btn in buttons:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setMinimumHeight(40)
            self.button_layout.addWidget(btn)

        # Initially hide some buttons
        self.confirm_face_btn.setVisible(False)
        self.retry_face_btn.setVisible(False)
        self.left_layout.addLayout(self.button_layout)
        
        # Add the show attendance list button in a separate layout at the bottom
        self.show_list_layout = QHBoxLayout()
        self.show_list_layout.addWidget(self.show_attendance_list_btn)
        self.left_layout.addLayout(self.show_list_layout)

        # Setup the attendance list on the right side
        self.setup_attendance_list()
        
        # Set the root layout as the widget's layout
        self.setLayout(self.root_layout)
        
        # We need to make sure the back button stays at the correct position
        # even when the window is resized
        self.back_btn.raise_()
        # Update back button position to align with camera frame
        #self.back_btn.move(20, 20)  # Position to align with camera window (inset from edge)
        self.back_btn.setVisible(False)  # Initially hidden
        # Connect signals
        self.take_attendance_btn.clicked.connect(self.take_attendance)
        self.register_face_btn.clicked.connect(self.show_registration_dialog)
        self.confirm_face_btn.clicked.connect(self.confirm_face)
        self.retry_face_btn.clicked.connect(self.retry_wrong_face)
        self.show_attendance_list_btn.clicked.connect(self.toggle_attendance_list)
        self.back_btn.clicked.connect(self.back_to_main)

        self.setup_idle_animation()

        # Set initial font sizes
        self.update_font_sizes()
        
        # Initially update the attendance list
        self.update_attendance_list()

    def setup_idle_animation(self):
        """Set up the GIF animation for when the camera is not active"""
        self.idle_movie = QMovie("icon/9BUtul0Vz2.gif")  # Replace with your GIF path
        self.idle_movie.setScaledSize(QSize(300, 300))  # Set appropriate size
        self.image_label.setMovie(self.idle_movie)
        self.idle_movie.start()
        
    def show_idle_animation(self):
        """Display the GIF animation"""
        if hasattr(self, 'idle_movie'):
            self.image_label.setMovie(self.idle_movie)
            self.idle_movie.start()
            
    def hide_idle_animation(self):
        """Stop the GIF animation when switching to camera"""
        if hasattr(self, 'idle_movie'):
            self.idle_movie.stop()

    def back_to_main(self):
        self.image_label.clear()  # Clear the camera feed display
        self.take_attendance_btn.setVisible(True)
        self.register_face_btn.setVisible(True)
        self.confirm_face_btn.setVisible(False)
        self.retry_face_btn.setVisible(False)
        self.back_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        # Reset status label to initial state
        self.status_label.setText(f"{self.attendance.present_count} / {self.attendance.total_students} students present")
        self.status_label.setStyleSheet("color: #6C757D; font-weight: bold; padding: 8px; margin: 5px 0;") 
        self.stop_camera()

    def setup_attendance_list(self):
        self.attendance_title = QLabel("Student Attendance Status")
        self.attendance_title.setAlignment(Qt.AlignCenter)
        self.attendance_title.setStyleSheet("color: #007BFF; font-weight: bold; font-size: 16px; padding: 10px;")
        self.right_layout.addWidget(self.attendance_title)

        self.attendance_list = QListWidget()
        self.right_layout.addWidget(self.attendance_list)

        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setStyleSheet("color: #6C757D; font-weight: bold;	padding: 10px;")
        self.right_layout.addWidget(self.summary_label)

    def toggle_attendance_list(self):
        self.attendance_list_visible = not self.attendance_list_visible
        self.right_widget.setVisible(self.attendance_list_visible)
        self.show_attendance_list_btn.setText(
            "Hide Attendance List" if self.attendance_list_visible else "Show Attendance List"
        )
        if self.attendance_list_visible:
            self.splitter.setSizes([int(self.width()*0.6), int(self.width()*0.4)])
            self.update_attendance_list()

    # def back_to_main(self):
    #     self.stop_camera()
    #     self.image_label.clear()  # Clear the camera feed display
    #     self.take_attendance_btn.setVisible(True)
    #     self.register_face_btn.setVisible(True)
    #     self.confirm_face_btn.setVisible(False)
    #     self.retry_face_btn.setVisible(False)
    #     # self.back_btn.setVisible(False)
    #     self.progress_bar.setVisible(False)
    #     # Reset status label to initial state
    #     self.status_label.setText(f"{self.attendance.present_count} / {self.attendance.total_students} students present")
    #     self.status_label.setStyleSheet("color: #6C757D; font-weight: bold; padding: 8px; margin: 5px 0;") 

    def update_attendance_list(self):
        if not self.attendance_list_visible:
            return
        self.attendance_list.clear()
        for idx, student in self.attendance.student_data.items():
            status = "✅ Present" if student['present'] else "❌ Absent"
            text = f"{student['name']} (ID: {student['id']}) - {status}"
            item = QListWidgetItem(text)
            item.setForeground(QColor("#28A745") if student['present'] else QColor("#DC3545"))
            self.attendance_list.addItem(item)
        self.summary_label.setText(f"Present: {self.attendance.present_count} / {self.attendance.total_students}")

    def update_font_sizes(self):
        w, h = self.width(), self.height()
        scale = min(w/800, h/800)
        title_size = max(14, int(self.base_font_size*1.4*scale))
        label_size = max(10, int(self.base_font_size*0.9*scale))
        btn_size = max(9, int(self.base_font_size*0.8*scale))
        font = QFont()
        font.setPointSize(title_size); font.setBold(True)
        self.title_label.setFont(font); self.attendance_title.setFont(font)
        font2 = QFont(); font2.setPointSize(label_size)
        self.status_label.setFont(font2); self.summary_label.setFont(font2)
        btn_font = QFont(); btn_font.setPointSize(btn_size)
        for btn in [self.take_attendance_btn, self.register_face_btn,
                    self.confirm_face_btn, self.retry_face_btn,
                    self.show_attendance_list_btn, self.back_btn]:
            btn.setFont(btn_font)

    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Unable to access the camera.")
            return
        self.hide_idle_animation()
        self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.image_label.clear()
            self.show_idle_animation()

    def detect_and_draw_faces(self, frame):
        faces = self.detector.detect_faces(frame)
        self.face_detected = len(faces) > 0
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,123,0), 3)
        return frame, len(faces)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Detect faces and select the largest one
                faces = self.detector.detect_faces(frame)
                self.face_detected = len(faces) > 0

                # Find the largest face by area
                largest_face = None
                max_area = 0
                for (x, y, w, h) in faces:
                    area = w * h
                    if area > max_area:
                        max_area = area
                        largest_face = (x, y, w, h)

                # Draw rectangle only for the largest face
                if largest_face:
                    x, y, w, h = largest_face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 123, 0), 3)
                    face_count = 1
                else:
                    face_count = 0

                # Nếu có face và đã qua 5s, submit API call vào thread pool
                now = time.time()
                if not self.pause_api and largest_face and (now - self._last_embed_time > 5):
                    self._last_embed_time = now
                    if self._future is None or self._future.done():
                        # copy frame để khỏi race condition
                        frame_copy = frame.copy()
                        self._future = self.executor.submit(self.call_api, frame_copy)

                # Kiểm tra nếu có kết quả API đã xong
                if self._future and self._future.done():
                    try:
                        self._response = self._future.result()
                        self.current_student_id = str(self._response["student_id"]) if self._response else None
                    except Exception as e:
                        print("API error:", e)
                        self._response = None
                    finally:
                        self._future = None
                
                if self.confirm_face_btn.isVisible() or self.progress_bar.isVisible():
                    if face_count == 0:
                        current_text = "No face detected. Please position your face in the camera."
                        self.status_label.setText(self.format_status_text(current_text))
                        self.status_label.setStyleSheet("color: #DC3545; font-weight: bold; padding: 8px; margin: 5px 0;")
                        self._response = None
                        self.confirm_face_btn.setEnabled(False)
                    elif face_count == 1 and hasattr(self, "_response") and self._response:
                        # student = self.attendance.student_data.get(self.attendance.current_student)
                        if self._response["confidence"] < 2 / len(self.attendance.student_data):
                            current_text = "Unknown face detected..."
                            self.status_label.setText(self.format_status_text(current_text))
                            self.status_label.setStyleSheet("color: #6C757D; font-weight: bold; padding: 8px; margin: 5px 0;")
                            self.confirm_face_btn.setEnabled(False)
                        else:
                            current_text = "Student ID: {} (confidence: {})".format(self._response["student_id"],self._response["confidence"])
                            self.status_label.setText(self.format_status_text(current_text))
                            self.status_label.setStyleSheet("color: #28A745; font-weight: bold; padding: 8px; margin: 5px 0;")
                            self.confirm_face_btn.setEnabled(True)
                    else:
                        current_text = "Face detected. Processing..."
                        self.status_label.setText(self.format_status_text(current_text))
                        self.status_label.setStyleSheet("color: #007BFF; font-weight: bold; padding: 8px; margin: 5px 0;")
                        self.confirm_face_btn.setEnabled(False)

                label_size = min(self.image_label.width(), self.image_label.height())

                # Crop to square from center
                height, width, _ = frame.shape
                crop_size = min(height, width)
                center_x, center_y = width // 2, height // 2
                x1 = center_x - crop_size // 2
                y1 = center_y - crop_size // 2

                # Ensure crop region is within image bounds
                x1 = max(0, min(x1, width - crop_size))
                y1 = max(0, min(y1, height - crop_size))

                try:
                    square_frame = frame[y1:y1+crop_size, x1:x1+crop_size]
                    resized_frame = cv2.resize(square_frame, (label_size, label_size))
                    rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    qt_image = QImage(
                        rgb_image.data,
                        rgb_image.shape[1],
                        rgb_image.shape[0],
                        rgb_image.strides[0],
                        QImage.Format_RGB888
                    )
                    pixmap = QPixmap.fromImage(qt_image)
                    self.image_label.setPixmap(pixmap)
                except Exception as e:
                    print(f"Error processing frame: {e}")

    def call_api(self, frame):
        emb = self.embedder.embedding_face(frame)
        resp = self.API.predict(emb)
        return resp.json()

    def format_status_text(self, text):
        return text  # Keep full text for simplicity
    
    def show_registration_dialog(self):
        # Mở dialog để người dùng chọn nhiều video
        prep_df = pd.DataFrame(columns=range(513))  # 512 cho embedding + 1 cho student_id
        
        video_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Face Videos",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.flv *.webm *.m4v *.wmv *.3gp)"
        )
        if not video_paths:
            QMessageBox.warning(self, "Missing Videos", "Bạn phải chọn ít nhất một video.")
            return

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Processing Videos")
        msg_box.setText("Video đang được xử lý...\n(Vui lòng KHÔNG tắt ứng dụng)")
        msg_box.setStandardButtons(QMessageBox.NoButton)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStyleSheet("QLabel{min-width:300px; min-height:60px; font-size:14px;}")
        msg_box.show()
        QApplication.processEvents()  # Ensure the message box is rendered immediately
        
        preprocessing = Processing_Img(self.students)
        
        for src_path in video_paths:
            base_name = os.path.basename(src_path)
            name_no_ext = os.path.splitext(base_name)[0]
            try:
                # Đảm bảo định dạng ID_Name
                student_id, student_name = name_no_ext.split('_', 1)
                if not student_name:
                    raise ValueError("Tên không được để trống")
            except ValueError as e:
                QMessageBox.warning(
                    self,
                    "Invalid Filename",
                    f"Bỏ qua '{base_name}': {e}\nĐịnh dạng phải là ID_Name"
                )
                continue

            try:
                df = preprocessing.frame_to_vector(src_path, student_id, student_name)
                prep_df = pd.concat([prep_df, df], ignore_index=True)
            except Exception as ex:
                QMessageBox.warning(
                    self,
                    "Error in Processing",
                    f"Không thể xử lý video '{base_name}': {ex}"
                )
                continue
        
        # Lưu DataFrame vào tệp CSV
        if not prep_df.empty:
            # Gọi API với dữ liệu đã chuyển đổi
            try:
                self.API.push_data(prep_df)
                # Cập nhật dữ liệu JSON
                self.API.push_json_data(self.students)
            except Exception as e:
                QMessageBox.critical(self, "API Error", f"Không thể gửi dữ liệu đến API: {e}")
        
        msg_box.setText(f"Video đã được xử lý xong (Done)")
        QTimer.singleShot(2000, msg_box.accept)

    def take_attendance(self):
        # self.back_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.confirm_face_btn.setVisible(False)
        self.retry_face_btn.setVisible(False)
        self.API.load_model()
        self.start_camera()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(lambda: self.progress_bar.setValue(self.progress_bar.value()+1))
        self.progress_timer.start(30)
        QTimer.singleShot(3000, self.mock_attendance_result)
        self.take_attendance_btn.setVisible(False)
        self.register_face_btn.setVisible(False)
        self.footer_label.setVisible(False)
        self.back_btn.setVisible(True)

    def update_progress(self):
        """Update progress bar animation"""
        self.progress_value += 1
        if self.progress_value <= 100:
            self.progress_bar.setValue(self.progress_value)
        else:
            self.progress_timer.stop()

    def mock_attendance_result(self):
        self.progress_timer.stop()
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

        if self.current_student_id and self.current_student_id in self.attendance.student_data:
            # Nếu có kết quả nhận diện, cập nhật lại danh sách và label
            stu = self.attendance.student_data[self.current_student_id]
            self.status_label.setText(
                f"{stu['name']} (ID: {stu['id']}) - {self.attendance.present_count} / {self.attendance.total_students} students present"
            )
            self.status_label.setStyleSheet("color: #28A745; font-weight: bold; padding: 8px; margin: 5px 0;")
        self.confirm_face_btn.setVisible(True)
        self.retry_face_btn.setVisible(True)

    def confirm_face(self):
        if not self.current_student_id:
            return

        # đánh dấu sinh viên theo mã
        self.attendance.mark_present(self.current_student_id)
        # cập nhật lại danh sách và label
        self.update_attendance_list()
        self.status_label.setText(
            f"{self.attendance.student_data[self.current_student_id]['name']} (ID: {self.current_student_id}) - "
            f"{self.attendance.present_count} / {self.attendance.total_students} present"
        )
        log_entry = {
            "student_id": self.current_student_id,
            "student_name": self.attendance.student_data[self.current_student_id]["name"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": 1
        }
        self.log = pd.concat([self.log, pd.DataFrame([log_entry])], ignore_index=True)

    def retry_wrong_face(self):
        # 1. Tạm dừng việc gọi API
        self.pause_api = True

        # 2. Yêu cầu nhập mã SV thủ công
        student_id, ok = QInputDialog.getText(
            self,
            "Nhập mã SV",
            "Nhận diện sai. Nhập mã sinh viên để điểm danh:"
        )
        if not ok or not student_id.strip():
            self.status_label.setText("❌ Chưa nhập mã. Vui lòng thử lại.")
            # Tháo pause để tiếp tục API
            self.pause_api = False
            return

        student_id = student_id.strip()

        # 3. Đánh dấu điểm danh và cập nhật UI
        if student_id in self.attendance.student_data:
            # Nếu mã sinh viên hợp lệ, đánh dấu điểm danh
            self.attendance.mark_present(student_id)
            self.update_attendance_list()
            self.status_label.setText(
                f"✅ Điểm danh sinh viên {student_id} thành công.\n"
                f"{self.attendance.present_count} / {self.attendance.total_students} present"
            )
            log_entry = {
                "student_id": student_id,
                "student_name": self.attendance.student_data[student_id]["name"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": 0
            }
            self.log = pd.concat([self.log, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            # Nếu mã sinh viên không hợp lệ, thông báo lỗi bằng message box
            QMessageBox.warning(
                self,
                "Lỗi",
                f"Mã sinh viên '{student_id}' không hợp lệ. Vui lòng thử lại."
            )

        # 5. Bật lại việc gọi API cho những lần tiếp theo
        self.pause_api = False

    def finish_retry(self):
        self.progress_timer.stop()
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.confirm_face_btn.setVisible(True)
        self.retry_face_btn.setVisible(True)
        self.take_attendance_btn.setVisible(False)
        self.register_face_btn.setVisible(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_font_sizes()
    
    def closeEvent(self, event):
        if self.cap:
            self.stop_camera()
        
        # Hiển thị thanh loading khi gửi log
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Uploading Log")
        msg_box.setText("Đang gửi dữ liệu điểm danh lên server...\n(Vui lòng đợi trong giây lát)")
        msg_box.setStandardButtons(QMessageBox.NoButton)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStyleSheet("QLabel{min-width:300px; min-height:60px; font-size:14px;}")
        msg_box.show()
        QApplication.processEvents()  # Đảm bảo message box hiển thị ngay

        self.API.push_log(self.log)

        msg_box.setText("Đã gửi dữ liệu thành công! (Done)")
        QTimer.singleShot(2000, msg_box.accept)
        event.accept()