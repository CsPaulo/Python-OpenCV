from PIL import Image
import cv2
import numpy as np
import os

import zipfile
path = 'Datasets/jones_gabriel.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

def get_image_data():
  paths = [os.path.join('jones_gabriel', f) for f in os.listdir('jones_gabriel')]
  faces = []
  ids = []
  for path in paths:
    image = Image.open(path).convert('L')
    image_np = np.array(image, 'uint8')
    id = int(path.split('.')[1])
    
    ids.append(id)
    faces.append(image_np)
  
  return np.array(ids), faces

ids, faces = get_image_data()

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')
lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('lbph_classifier.yml')