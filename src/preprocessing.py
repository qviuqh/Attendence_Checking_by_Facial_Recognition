import cv2
import os
import glob
import numpy as np

class Vid2Img():
    def __init__(self, video_dir, output_dir, fps=30, duration=15):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration  # Tổng số khung hình cần trích xuất
    
    # Hàm xoay khung hình để đảm bảo định dạng dọc
    def rotate_to_portrait(self, frame):
        height, width = frame.shape[:2]
        if width > height:  # Nếu khung hình nằm ngang
            # Xoay 90° theo chiều kim đồng hồ
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def video_to_frames(self):
        # Tạo thư mục đầu ra chính nếu chưa tồn tại
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Lấy danh sách tất cả tệp MP4 trong thư mục video
        video_files = glob.glob(os.path.join(self.video_dir, "*.mov"))

        if not video_files:
            print("Không tìm thấy tệp MP4 nào trong thư mục:", self.video_dir)
            exit()

        # Lặp qua từng tệp video
        for video_file in video_files:
            # Lấy tên tệp (mã sinh viên) từ đường dẫn, bỏ phần mở rộng .mp4
            student_id = os.path.splitext(os.path.basename(video_file))[0]
            
            # Tạo thư mục con cho mã sinh viên
            student_output_dir = os.path.join(self.output_dir, student_id)
            if not os.path.exists(student_output_dir):
                os.makedirs(student_output_dir)
            
            # Tạo đối tượng VideoCapture để đọc video
            cap = cv2.VideoCapture(video_file)
            
            # Kiểm tra xem video có được mở thành công không
            if not cap.isOpened():
                print(f"Lỗi: Không thể mở video {video_file}")
                continue
            
            # In thông tin FPS của video gốc
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Đang xử lý video {video_file} (FPS gốc: {actual_fps})")
            
            # Khởi tạo biến đếm cho khung hình
            counter = 0
            
            # Duyệt qua các khung hình
            while counter < self.total_frames:
                ret, frame = cap.read()
                if not ret:
                    print(f"Hết video hoặc lỗi khi đọc khung hình tại {video_file}")
                    break
                
                # Xoay khung hình để đảm bảo định dạng dọc
                frame = rotate_to_portrait(frame)
                
                # Định nghĩa tên tệp ảnh với số thứ tự có 4 chữ số
                frame_name = f"frame_{counter:04d}.jpg"
                frame_path = os.path.join(student_output_dir, frame_name)
                
                # Lưu khung hình dưới dạng ảnh JPEG
                cv2.imwrite(frame_path, frame)
                counter += 1
            
            # Giải phóng đối tượng VideoCapture
            cap.release()
            
            # In thông báo kết quả cho video hiện tại
            print(f"Đã trích xuất {counter} khung hình từ {video_file}, lưu tại {student_output_dir}")
        print("Hoàn tất xử lý tất cả video.")