import cv2
import numpy as np
import pickle
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

from cvzone.FaceDetectionModule import FaceDetector
from architecture import *

###### paths and variables #########
face_data = 'Faces/'
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
encodes = []
new_encoding_dict = dict()
l2_normalizer = Normalizer('l2')
detector = FaceDetector()

# Load the encoding dictionary from the pickle file
with open('encodings/encodings.pkl', 'rb') as file:
    encoding_dict = pickle.load(file)


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def new_face_encoding(img_BGR):
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    img_RGB, bbox = detector.findFaces(img_RGB, draw=True)

    for box in bbox:
        x1, y1, x2, y2 = box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        face = None

        if y1 < y2 and x1 < x2:
            face = img_RGB[y1:y2, x1:x2]
        elif y1 < y2 and x2 < x1:
            face = img_RGB[y1:y2, x2:x1]
        elif y2 < y1 and x1 < x2:
            face = img_RGB[y2:y1, x1:x2]
        elif y2 < y1 and x2 < x1:
            face = img_RGB[y2:y1, x2:x1]

        if face is not None and face.shape[0] > 0 and face.shape[1] > 0:
            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        new_encoding_dict['new_face'] = encode
    return new_encoding_dict


# Open the video capture device (e.g., webcam)
cap = cv2.VideoCapture("yogi.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face recognition on the current frame
    new_encoding = new_face_encoding(frame)

    for name, encoding in new_encoding.items():
        max_similarity = -1
        for label, known_embedding in encoding_dict.items():
            similarity = cosine_similarity(encoding.reshape(1, -1), known_embedding.reshape(1, -1))

            if similarity > max_similarity:
                max_similarity = similarity
                matched_person = label

        if max_similarity >= 0.6:
            text = f"Detected face: {matched_person}, Similarity: {max_similarity[0][0]:.2f}"
            cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Unknown', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()