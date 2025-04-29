import cv2
import os
import glob
import pandas as pd
from src import feature_engineering as fe

class Processing_Img():
    def __init__(self, video_dir, student):
        self.video_dir = video_dir
        self.student = student
    
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

    def frame_to_vector(self, interval=5):
        """
        Đọc tất cả các tệp video trong thư mục video_dir và trích xuất khung hình và xử lý nó từ mỗi video.
        Embedding các frame đã xử lý đó thành các vector d-512
        Args:
        Returns: DataFrame chứa các vector d-512 và student_id cho mỗi khung hình đã xử lý.
        """
        embedder = fe.FaceEmbedding()

        # Lấy danh sách tất cả tệp MP4 trong thư mục video
        video_files = glob.glob(os.path.join(self.video_dir, "*.mov"))

        if not video_files:
            print("Không tìm thấy tệp MOV nào trong thư mục:", self.video_dir)
            exit()
        
        df = pd.DataFrame(columns=range(513))  # Tạo DataFrame rỗng với 513 cột (512 cho embedding + 1 cho student_id)
        index = 0
        
        # Lặp qua từng tệp video
        for video_file in video_files:
            # Lấy tên tệp (mã sinh viên) từ đường dẫn, bỏ phần mở rộng .mp4
            student_id = os.path.splitext(os.path.basename(video_file))[0]
            
            if student_id in self.student:
                continue  # Bỏ qua video nếu student_id đã có trong danh sách student
            
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
            frame_count = 0
            
            # Duyệt qua các khung hình
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Hết video hoặc lỗi khi đọc khung hình tại {video_file}")
                    break
                
                if frame_count % interval == 0:
                    # Trích xuất khung hình tại khoảng thời gian nhất định
                    # Xoay khung hình để đảm bảo định dạng dọc
                    frame = self.rotate_to_portrait(frame)
                    embedding = embedder.embedding_face(frame)  # Trích xuất embedding cho khuôn mặt
                    if embedding is None:
                        frame_count += 1
                        continue
                    embedding = pd.Series(embedding)
                    embedding = pd.concat([embedding, pd.Series([student_id])], ignore_index=True)
                    df.loc[index] = embedding
                    index += 1
                    counter += 1
                frame_count += 1
            
            # Giải phóng đối tượng VideoCapture
            cap.release()
            # In thông báo kết quả cho video hiện tại
            print(f"Đã trích xuất {counter} khung hình từ video cuar sinh viên {student_id}")
            self.student[student_id] = counter  # Thêm student_id vào danh sách đã xử lý
        print("Hoàn tất xử lý tất cả video.")
        return df