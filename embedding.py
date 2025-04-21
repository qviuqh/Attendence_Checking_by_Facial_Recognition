import os
import pickle
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# Khởi tạo mô hình FaceNet (InceptionResnetV1) để tạo embedding
model = InceptionResnetV1(pretrained='vggface2').eval()

# Đường dẫn đến thư mục chứa khuôn mặt đã xử lý và file lưu embedding
input_dir = "dataset/processed_faces/"  # Thư mục chứa các thư mục con theo mã sinh viên
output_file = "output/embeddings.pkl"  # File lưu embedding

# Hàm tạo embedding từ ảnh khuôn mặt
def generate_embeddings():
    embeddings_dict = {}  # Dictionary để lưu embedding: {mã_sinh_viên: [embedding1, embedding2, ...]}

    # Duyệt qua tất cả các thư mục con (mã sinh viên) trong thư mục processed_faces
    for student_id in os.listdir(input_dir):
        student_dir = os.path.join(input_dir, student_id)
        if os.path.isdir(student_dir):  # Kiểm tra xem có phải là thư mục không
            embeddings_dict[student_id] = []  # Khởi tạo danh sách embedding cho mã sinh viên
            
            # Duyệt qua tất cả các file ảnh trong thư mục của sinh viên
            for filename in os.listdir(student_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):  # Chỉ xử lý file ảnh
                    image_path = os.path.join(student_dir, filename)
                    
                    # Đọc ảnh và chuyển thành tensor
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                    image_tensor = image_tensor.unsqueeze(0)  # Thêm batch dimension
                    
                    # Tạo embedding bằng FaceNet
                    with torch.no_grad():
                        embedding = model(image_tensor).numpy().flatten()  # Vector 512 chiều
                    
                    # Thêm embedding vào danh sách của mã sinh viên
                    embeddings_dict[student_id].append(embedding)
                    print(f"Đã tạo embedding cho ảnh: {filename} (Mã sinh viên: {student_id})")

    # Lưu dictionary embedding vào file
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"Đã lưu embedding vào: {output_file}")

# Chạy hàm tạo embedding
if __name__ == "__main__":
    generate_embeddings()