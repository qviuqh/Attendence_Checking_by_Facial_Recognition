from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QMessageBox

class StudentRegistrationDialog(QDialog):
    """Dialog for entering student information during registration"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Student Registration")
        self.setMinimumWidth(300)
        self.setStyleSheet("""
            QDialog { background-color: #F5F7FA; }
            QLineEdit { padding: 8px; border: 1px solid #6C757D; border-radius: 4px; background-color: white; }
            QPushButton { background-color: #007BFF; color: white; padding: 8px 16px; border-radius: 4px; border: none; }
            QPushButton:hover { background-color: #0069D9; }
            QLabel { color: #212529; }
        """)
        layout = QFormLayout(self)
        self.name_input = QLineEdit()
        self.id_input = QLineEdit()
        layout.addRow("Student Name:", self.name_input)
        layout.addRow("Student ID:", self.id_input)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)
        self.name_input.setFocus()

    def get_student_info(self):
        return {"name": self.name_input.text(), "id": self.id_input.text()}
