import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QHBoxLayout, QProgressBar, QMessageBox, QSizePolicy,
    QSplitter, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer

from dialogs import StudentRegistrationDialog
from face_detector import FaceDetector
from attendance_manager import AttendanceManager

class FaceAttendanceUI(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize modules
        import copy
        base_students = {
            1: {"name": "John Doe", "id": "12345", "present": False},
            2: {"name": "Jane Smith", "id": "67890", "present": False},
        }
        for i in range(3, 43):
            base_students[i] = {"name": f"Student {i}", "id": f"ID{10000+i}", "present": False}
        self.attendance = AttendanceManager(copy.deepcopy(base_students))
        self.detector = FaceDetector()

        # Camera & timers
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.face_detected = False

        # UI state
        self.new_student_info = None
        self.attendance_list_visible = False
        self.base_font_size = 12

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

        self.main_layout = QHBoxLayout(self)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(20,20,20,20)

        # Splitter
        self.splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        right_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        self.left_layout.setAlignment(Qt.AlignCenter)
        self.left_layout.setSpacing(20)
        self.right_layout = QVBoxLayout(right_widget)
        self.right_layout.setAlignment(Qt.AlignTop)
        self.right_layout.setSpacing(10)
        right_widget.setVisible(False)
        self.right_widget = right_widget

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([600,400])

        self.main_layout.addWidget(self.splitter)

        # Left side UI
        self.title_label = QLabel("Face Recognition Attendance System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("color: #007BFF; font-weight: bold;")
        self.left_layout.addWidget(self.title_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #6C757D; border-radius: 8px; padding: 5px; background-color: white;")
        self.left_layout.addWidget(self.image_label, stretch=1)

        self.status_label = QLabel(f"0 / {self.attendance.total_students} students present")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #6C757D; font-weight: bold;	padding: 8px; margin: 5px 0;")
        self.left_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.left_layout.addWidget(self.progress_bar)

        # Buttons
        self.take_attendance_btn = QPushButton("Take Attendance")
        self.register_face_btn = QPushButton("Register New Face")
        self.confirm_face_btn = QPushButton("Confirm Face")
        self.retry_face_btn = QPushButton("Retry Wrong Face")
        self.show_attendance_list_btn = QPushButton("Show Attendance List")
        self.confirm_face_btn.setVisible(False)
        self.retry_face_btn.setVisible(False)

        btn_layout = QHBoxLayout()
        for btn in [self.take_attendance_btn, self.register_face_btn, self.confirm_face_btn, self.retry_face_btn]:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn_layout.addWidget(btn)
        self.left_layout.addLayout(btn_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.show_attendance_list_btn)
        self.left_layout.addLayout(bottom_layout)

        # Setup attendance list
        self.setup_attendance_list()

        # Connect signals
        self.take_attendance_btn.clicked.connect(self.take_attendance)
        self.register_face_btn.clicked.connect(self.show_registration_dialog)
        self.confirm_face_btn.clicked.connect(self.confirm_face)
        self.retry_face_btn.clicked.connect(self.retry_wrong_face)
        self.show_attendance_list_btn.clicked.connect(self.toggle_attendance_list)

        self.update_font_sizes()
        self.update_attendance_list()

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

    def update_attendance_list(self):
        if not self.attendance_list_visible:
            return
        self.attendance_list.clear()
        for idx, student in sorted(self.attendance.student_data.items()):
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
                    self.show_attendance_list_btn]:
            btn.setFont(btn_font)

    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Unable to access the camera.")
            return
        self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.image_label.clear()

    def detect_and_draw_faces(self, frame):
        faces = self.detector.detect_faces(frame)
        self.face_detected = len(faces) > 0
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,123,0), 3)
        return frame, len(faces)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: return
            frame, cnt = self.detect_and_draw_faces(frame)
            # Status updates omitted for brevity
            size = min(self.image_label.width(), self.image_label.height())
            h, w = frame.shape[:2]
            c = min(h, w)
            cx, cy = w//2, h//2
            x1, y1 = max(0, cx-c//2), max(0, cy-c//2)
            crop = frame[y1:y1+c, x1:x1+c]
            img = cv2.resize(crop, (size, size))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt))

    def format_status_text(self, text):
        return text  # Keep full text for simplicity

    def show_registration_dialog(self):
        dlg = StudentRegistrationDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            info = dlg.get_student_info()
            if not info['name'] or not info['id']:
                QMessageBox.warning(self, "Invalid Input", "Enter both name and ID.")
                return
            self.new_student_info = info
            self.register_face()

    def take_attendance(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.confirm_face_btn.setVisible(False)
        self.retry_face_btn.setVisible(False)
        self.start_camera()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(lambda: self.progress_bar.setValue(self.progress_bar.value()+1))
        self.progress_timer.start(30)
        QTimer.singleShot(3000, self.mock_attendance_result)

    def mock_attendance_result(self):
        self.progress_timer.stop()
        self.progress_bar.setValue(100)
        stu = self.attendance.student_data[self.attendance.current_student]
        self.status_label.setText(f"{stu['name']} (ID: {stu['id']}) - {self.attendance.present_count} / {self.attendance.total_students} students present")
        self.progress_bar.setVisible(False)
        self.confirm_face_btn.setVisible(True)
        self.retry_face_btn.setVisible(True)

    def confirm_face(self):
        self.attendance.mark_present()
        self.update_attendance_list()
        self.status_label.setText(f"{self.attendance.present_count} / {self.attendance.total_students} present")
        self.confirm_face_btn.setVisible(False)
        self.retry_face_btn.setVisible(False)

    def retry_wrong_face(self):
        self.status_label.setText("Retrying recognition...")
        self.progress_bar.setVisible(True)
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(lambda: self.progress_bar.setValue(self.progress_bar.value()+5))
        self.progress_timer.start(30)
        QTimer.singleShot(2000, self.finish_retry)

    def finish_retry(self):
        self.progress_timer.stop()
        self.progress_bar.setVisible(False)
        self.status_label.setText("Retry complete")
        self.confirm_face_btn.setVisible(False)
        self.retry_face_btn.setVisible(False)

    def register_face(self):
        if not self.new_student_info:
            QMessageBox.warning(self, "Error", "Missing student info.")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.take_attendance_btn.setVisible(False)
        self.register_face_btn.setVisible(False)
        self.start_camera()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(lambda: self.progress_bar.setValue(self.progress_bar.value()+1))
        self.progress_timer.start(30)
        self.check_timer = QTimer(self)
        self.check_timer.timeout.connect(lambda: None)
        self.check_timer.start(500)
        QTimer.singleShot(3000, self.finish_register)

    def finish_register(self):
        self.progress_timer.stop()
        self.check_timer.stop()
        self.progress_bar.setVisible(False)
        if self.face_detected:
            self.attendance.add_student(self.new_student_info['name'], self.new_student_info['id'])
            self.status_label.setText(f"Registered {self.new_student_info['name']}. Total: {self.attendance.total_students}")
            QMessageBox.information(self, "Success", f"{self.new_student_info['name']} registered.")
            self.update_attendance_list()
        else:
            QMessageBox.warning(self, "Failed", "No face detected.")
            self.status_label.setText("Registration failed.")
        self.new_student_info = None
        self.take_attendance_btn.setVisible(True)
        self.register_face_btn.setVisible(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_font_sizes()

    def closeEvent(self, event):
        if self.cap: self.stop_camera()
        event.accept()
