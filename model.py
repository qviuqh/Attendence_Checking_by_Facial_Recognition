import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# Khởi tạo MTCNN để phát hiện và cắt khuôn mặt
mtcnn = MTCNN(image_size=160, margin=10, min_face_size=20, keep_all=False, device='cpu')

# Định nghĩa các biến đổi cho ảnh
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Tải thông tin nhãn từ thư mục processed_faces
data_dir = "dataset/processed_faces/"
label_to_idx = {}
for student_id in os.listdir(data_dir):
    student_dir = os.path.join(data_dir, student_id)
    if os.path.isdir(student_dir):  # Kiểm tra xem có phải là thư mục không
        if student_id not in label_to_idx:
            label_to_idx[student_id] = len(label_to_idx)

# Kiểm tra xem label_to_idx có rỗng không
if not label_to_idx:
    raise ValueError("Không tìm thấy mã sinh viên nào trong processed_faces/.")

# Tải mô hình đã huấn luyện
num_classes = len(label_to_idx)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load("face_classifier.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Hàm nhận dạng khuôn mặt từ ảnh
def recognize_face(image_path):
    # Đọc và tiền xử lý ảnh
    image = Image.open(image_path).convert('RGB')
    
    # Phát hiện và cắt khuôn mặt
    face_tensor = mtcnn(image)
    if face_tensor is None:
        return "Không tìm thấy khuôn mặt trong ảnh."

    # Chuyển tensor về dạng numpy array, sau đó thành PIL Image
    face_np = face_tensor.permute(1, 2, 0).numpy()  # Chuyển từ [C, H, W] về [H, W, C]
    face_np = (face_np * 255).astype(np.uint8)  # Chuyển về dạng uint8
    face_pil = Image.fromarray(face_np)

    # Áp dụng biến đổi cho ảnh
    face = transform(face_pil).unsqueeze(0)  # Thêm batch dimension
    face = face.to(device)

    # Dự đoán với mô hình
    with torch.no_grad():
        output = model(face)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = predicted.item()
        confidence_score = confidence.item()

    # Chuyển label số thành student_id
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    student_id = idx_to_label[predicted_label]

    return f"Sinh viên được nhận dạng: {student_id} (Confidence: {confidence_score:.2f})"

# Kiểm tra với ảnh test_image
if __name__ == "__main__":
    test_image_path = "test_image2.jpg"  # Đường dẫn đến ảnh cần nhận dạng
    result = recognize_face(test_image_path)
    print(result)