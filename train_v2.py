from architecture import *
import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from cvzone.FaceDetectionModule import FaceDetector


###### paths and variables #########
face_data = 'Faces/'
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
detector = FaceDetector()

###################################

def normalize(img):
    mean, std = img.mean(), img.std()
    # print(mean,std)
    return (img - mean) / std


for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data, face_names)

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        # Perform face detection with YOLO
        # detections = yolo_model.detect_image(img_RGB)
        img_RGB, bbox = detector.findFaces(img_RGB, draw=True)
        # print(bbox)

        for box in bbox:
            # print(box['bbox'][0])

            x1, y1, x2, y2= box['bbox'][0],box['bbox'][1],box['bbox'][2],box['bbox'][3]
            # print(x1, y1, x2, y2)
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            # print(x1,y1,x2,y2)
            face = None  # Initialize the variable to None

            # Attempt to extract the face using different combinations of coordinates
            if y1 < y2 and x1 < x2:
                face = img_RGB[y1:y2, x1:x2]
            elif y1 < y2 and x2 < x1:
                face = img_RGB[y1:y2, x2:x1]
            elif y2 < y1 and x1 < x2:
                face = img_RGB[y2:y1, x1:x2]
            elif y2 < y1 and x2 < x1:
                face = img_RGB[y2:y1, x2:x1]

            if face is not None and face.shape[0] > 0 and face.shape[1] > 0:
                print(face.shape)
                face = normalize(face)
                face = cv2.resize(face, required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = face_encoder.predict(face_d)[0]
                encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode

path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)












# from architecture import *
# import os
# import cv2
# import mtcnn
# import pickle
# import numpy as np
# from sklearn.preprocessing import Normalizer
# from keras.models import load_model
#
# ######pathsandvairables#########
# face_data = 'Faces/'
# required_shape = (160,160)
# face_encoder = InceptionResNetV2()
# path = "facenet_keras_weights.h5"
# face_encoder.load_weights(path)
# face_detector = mtcnn.MTCNN()
# encodes = []
# encoding_dict = dict()
# l2_normalizer = Normalizer('l2')
# ###############################
#
#
# def normalize(img):
#     mean, std = img.mean(), img.std()
#     return (img - mean) / std
#
#
# for face_names in os.listdir(face_data):
#     person_dir = os.path.join(face_data,face_names)
#
#     for image_name in os.listdir(person_dir):
#         image_path = os.path.join(person_dir,image_name)
#
#         img_BGR = cv2.imread(image_path)
#         img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
#
#         x = face_detector.detect_faces(img_RGB)
#         x1, y1, width, height = x[0]['box']
#         x1, y1 = abs(x1) , abs(y1)
#         x2, y2 = x1+width , y1+height
#         face = img_RGB[y1:y2 , x1:x2]
#
#         face = normalize(face)
#         face = cv2.resize(face, required_shape)
#         face_d = np.expand_dims(face, axis=0)
#         encode = face_encoder.predict(face_d)[0]
#         encodes.append(encode)
#
#     if encodes:
#         encode = np.sum(encodes, axis=0 )
#         encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
#         encoding_dict[face_names] = encode
#
# path = 'encodings/encodings.pkl'
# with open(path, 'wb') as file:
#     pickle.dump(encoding_dict, file)
#
#
#
#
#
#
