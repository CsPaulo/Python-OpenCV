from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

# Carregamento da base de dados
import zipfile
path = 'Datasets/yalefaces.zip'
zip_object = zipfile.ZipFile(file=path, mode = 'r')
zip_object.extractall('./')
zip_object.close()

# Pré-processamento das imagens
import os
def get_image_data():
  paths = [os.path.join('yalefaces/train', f) for f in os.listdir('yalefaces/train')]
  faces = []
  ids = []
  for path in paths:
    imagem = Image.open(path).convert('L')
    imagem_np = np.array(imagem, 'uint8')
    id = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    ids.append(id)
    faces.append(imagem_np)
    
  return np.array(ids), faces

ids, faces = get_image_data()

# Treinamento do classificador LBPH
#default parameters
#threshold: 1.7976931348623157e+308
#radius: 1
#neighbors: 8
#grid_x: 8
#grid_y: 8
#ideal parameters
#radius: 4
#neighbors: 14
#grid_x: 9
#grid_y: 9

lbph = cv2.face_LBPHFaceRecognizer.create(radius=1, neighbors=8, grid_x=8, grid_y=8)
lbph.train(faces, ids)
lbph.save('lbph_classifier.yml')

# Reconhecimento de faces

lbph_face = cv2.face_LBPHFaceRecognizer.create()
lbph_face.read('lbph_classifier.yml')

image_teste = 'yalefaces/test/subject10.sad.gif'
image = Image.open(image_teste).convert('L')
image_np = np.array(image, 'uint8')

prevision = lbph_face.predict(image_np)

expected_output = int(os.path.split(image_teste)[1].split('.')[0].replace('subject',''))

cv2.putText(image_np, 'Pred:' + str(prevision[0]), (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.putText(image_np, 'Exp:' + str(expected_output), (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv2.imshow('reconhecimento facial', image_np)
cv2.waitKey(0)

# Avaliação do classificador

paths = (os.path.join('yalefaces\test', f) for f in os.listdir('yalefaces\test'))
for path in paths:
  predictions = []
  expected_outputs = []
  prevision, _ = lbph_face.predict(image_np)
  predictions.append(prevision)
  expected_outputs.apprender(expected_output)

predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)

acuracy_score(expected_outputs, predictions)
print(acuracy_score(expected_outputs, predictions))

