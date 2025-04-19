import cv2
import os
import glob

class Processing_Img():
    def __init__(self, video_dir, output_dir, fps=30, duration=15):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration  # Tổng số khung hình cần trích xuất
    
    # Hàm xoay khung hình để đảm bảo định dạng dọc
    def rotate_to_portrait(self, frame):
        """_Thiết lập hàm xoay khung hình để đảm bảo định dạng dọc
        Args:
            frame (Ảnh): Khung hình video cần xử lý.
        1. Nếu chiều rộng lớn hơn chiều cao, xoay khung hình 90° theo chiều kim đồng hồ.
        2. Nếu chiều cao lớn hơn chiều rộng, không cần xoay.
        3. Trả về khung hình đã xử lý.
        Returns:
            frame (Ảnh): Khung hình đã được xoay (nếu cần).
        """
        height, width = frame.shape[:2]
        if width > height:  # Nếu khung hình nằm ngang
            # Xoay 90° theo chiều kim đồng hồ
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def video_to_frames(self):
        """
        _Thiết lập hàm xoay khung hình để đảm bảo định dạng dọc
        Args:
        Đọc tất cả các tệp video trong thư mục video_dir và trích xuất khung hình và xử lý nó từ mỗi video.
        Returns: None (Không trả về giá trị nào)
        """
        # Tạo thư mục đầu ra chính nếu chưa tồn tại
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Lấy danh sách tất cả tệp MP4 trong thư mục video
        video_files = glob.glob(os.path.join(self.video_dir, "*.mov"))

        if not video_files:
            print("Không tìm thấy tệp MOV nào trong thư mục:", self.video_dir)
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
            counter = 1
            
            # Duyệt qua các khung hình
            while counter <= self.total_frames:
                ret, frame = cap.read()
                if not ret:
                    print(f"Hết video hoặc lỗi khi đọc khung hình tại {video_file}")
                    break
                
                # Xoay khung hình để đảm bảo định dạng dọc
                frame = self.rotate_to_portrait(frame)
                frame_name = f"frame_{counter}.jpg"
                frame_path = os.path.join(student_output_dir, frame_name)
                # Lưu khung hình dưới dạng ảnh JPEG
                cv2.imwrite(frame_path, frame)
                counter += 1
            
            # Giải phóng đối tượng VideoCapture
            cap.release()
            
            # In thông báo kết quả cho video hiện tại
            print(f"Đã trích xuất {counter} khung hình từ {video_file}, lưu tại {student_output_dir}")
        print("Hoàn tất xử lý tất cả video.")