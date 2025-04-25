import sqlite3

# Tạo (hoặc kết nối) database
conn = sqlite3.connect("attendance_system.db")
cursor = conn.cursor()

# 1. Bảng students
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    student_code TEXT UNIQUE NOT NULL,
    class TEXT,
    email TEXT
)
""")

# 3. Bảng attendance_log
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    checkin_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT,
    FOREIGN KEY(student_id) REFERENCES students(id)
)
""")

conn.commit()
conn.close()

print("Database created thành công!")