from keras_facenet import FaceNet

embedder = FaceNet()

def extract_embedding(face_img):
    # face_img: 160x160x3, RGB
    return embedder.embeddings([face_img])[0]  # return vector 512-D