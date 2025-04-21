import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Định nghĩa dataset tùy chỉnh cho dữ liệu khuôn mặt
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_idx = {}
        
        # Duyệt qua các thư mục con (mã sinh viên) trong thư mục processed_faces
        for student_id in os.listdir(data_dir):
            student_dir = os.path.join(data_dir, student_id)
            if os.path.isdir(student_dir):  # Kiểm tra xem có phải là thư mục không
                # Gán nhãn cho mã sinh viên
                if student_id not in self.label_to_idx:
                    self.label_to_idx[student_id] = len(self.label_to_idx)
                
                # Duyệt qua tất cả các file ảnh trong thư mục của sinh viên
                for filename in os.listdir(student_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(student_dir, filename)
                        self.images.append(image_path)
                        self.labels.append(self.label_to_idx[student_id])
        
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Định nghĩa các biến đổi cho dữ liệu
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Tải dữ liệu
data_dir = "dataset/processed_faces/"  # Thư mục chứa khuôn mặt đã xử lý
dataset = FaceDataset(data_dir, transform=transform)
# dataset/processed_faces

# Kiểm tra xem dataset có dữ liệu không
if len(dataset) == 0:
    raise ValueError("Dataset rỗng! Vui lòng kiểm tra thư mục processed_faces/.")

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Tải mô hình ResNet18 và điều chỉnh lớp cuối cho số lớp (số sinh viên)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, dataset.num_classes)

# Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Đánh giá trên tập kiểm tra
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Lưu mô hình đã huấn luyện
torch.save(model.state_dict(), "face_classifier.pth")
print("Đã lưu mô hình tại: face_classifier.pth")