from PyQt5.QtWidgets import QApplication
import sys
from ui import FaceAttendanceUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceAttendanceUI()
    window.show()
    sys.exit(app.exec_())