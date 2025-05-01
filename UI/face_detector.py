import cv2

class FaceDetector:
    def __init__(self, cascade_path=None):
        path = cascade_path or cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(path)

    def detect_faces(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
            minSize=minSize, flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces