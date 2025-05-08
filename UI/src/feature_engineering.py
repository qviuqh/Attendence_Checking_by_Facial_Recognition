from keras_facenet import FaceNet
import cv2
import numpy as np
import os

class FaceEmbedding:
    """
    Class to handle face detection, cropping, and embedding extraction.
    This class uses Haar Cascade classifier for face detection and FaceNet for embedding extraction.
    """
    def __init__(self, model_path=None):
        '''
        Khởi tạo model dectect face và embedding face.
        Args:
            model_path (str): Đường dẫn đến mô hình FaceNet đã được huấn luyện trước.
            Nếu không có, sẽ sử dụng mô hình mặc định.
        '''
        self.model_path = model_path
        self.embedder = FaceNet(model_path) if model_path else FaceNet()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, img):
        """Detects faces in an image using Haar Cascade classifier.
        This function takes an image as input and returns the coordinates of detected faces.
        Args:
            img (_type_): Ảnh chứa khuôn mặt cần phát hiện.
        Returns:
            List: Chứa các tọa độ của khuôn mặt được phát hiện trong định dạng (x, y, w, h).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces  # List of (x, y, w, h)

    def crop_and_preprocess_face(self, img, face):
        """_summary_
        Args:
            img (.jpeg): Ảnh gốc ban đầu.
            face (List): Tọa độ khuôn mặt GẦN NHẤT được phát hiện trong định dạng (x, y, w, h).
        Returns:
            img (.jpeg): Ảnh khuôn mặt đã được cắt và tiền xử lý.
        """
        x, y, w, h = face
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))  # chuẩn cho model như FaceNet
        return face_img

    def extract_embedding(self, face_img):
        # face_img: 160x160x3, RGB
        return self.embedder.embeddings([face_img])[0]  # return vector 512-D
    
    def embedding_face(self, img):
        """
        Trích xuất embedding cho 1 ảnh đầu vào.
        Args:
            img (_type_): Ảnh đầu vào.
        Returns:
            _type_: embedding vector 512-D
        """
        faces = self.detect_face(img)
        if len(faces) == 0:
            return None
        # Lấy khuôn mặt có diện tích lớn nhất
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        face_img = self.crop_and_preprocess_face(img, face)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # face_img = face_img.astype('float32')
        embedding = self.extract_embedding(face_img)
        return embedding