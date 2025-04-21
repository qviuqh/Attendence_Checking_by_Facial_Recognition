import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

# Khởi tạo MTCNN để phát hiện khuôn mặt
mtcnn = MTCNN(image_size=160, margin=10, min_face_size=20, keep_all=False, device='cpu')

# Đường dẫn đến thư mục chứa ảnh đầu vào và thư mục lưu ảnh đã xử lý
input_dir = "dataset/pictures/"  # Thư mục chứa các thư mục con theo mã sinh viên
output_dir = "dataset/processed_faces/"  # Thư mục lưu khuôn mặt đã cắt

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hàm tiền xử lý ảnh
def preprocess_images():
    # Duyệt qua tất cả các thư mục con (mã sinh viên) trong thư mục đầu vào
    for student_id in os.listdir(input_dir):
        student_dir = os.path.join(input_dir, student_id)
        if os.path.isdir(student_dir):  # Kiểm tra xem có phải là thư mục không
            # Tạo thư mục đầu ra cho mã sinh viên
            student_output_dir = os.path.join(output_dir, student_id)
            if not os.path.exists(student_output_dir):
                os.makedirs(student_output_dir)

            # Đếm số thứ tự cho tên file đầu ra
            image_count = 1

            # Duyệt qua tất cả các file ảnh trong thư mục của sinh viên
            for filename in os.listdir(student_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ xử lý file ảnh
                    image_path = os.path.join(student_dir, filename)
                    
                    # Đọc ảnh bằng OpenCV và chuyển sang RGB (MTCNN yêu cầu RGB)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Không thể đọc ảnh: {image_path}")
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Chuyển ảnh thành định dạng PIL để MTCNN xử lý
                    image_pil = Image.fromarray(image_rgb)
                    
                    # Phát hiện khuôn mặt bằng MTCNN
                    boxes, _ = mtcnn.detect(image_pil)
                    
                    # Nếu phát hiện được khuôn mặt
                    if boxes is not None:
                        # Cắt và chuẩn hóa khuôn mặt bằng MTCNN
                        face = mtcnn(image_pil)
                        
                        if face is not None:
                            # Chuyển tensor thành numpy array
                            face = face.permute(1, 2, 0).numpy()
                            face = (face * 255).astype(np.uint8)  # Chuyển về dạng ảnh
                            
                            # Đặt tên file theo dạng <mã_sinh_viên>_<số_thứ_tự>.jpg
                            output_filename = f"{student_id}_{image_count}.jpg"
                            output_path = os.path.join(student_output_dir, output_filename)
                            cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                            print(f"Đã xử lý và lưu khuôn mặt: {output_path}")
                            image_count += 1
                        else:
                            print(f"Không thể cắt khuôn mặt từ ảnh: {filename}")
                    else:
                        print(f"Không tìm thấy khuôn mặt trong ảnh: {filename}")

# Chạy hàm tiền xử lý
if __name__ == "__main__":
    preprocess_images()