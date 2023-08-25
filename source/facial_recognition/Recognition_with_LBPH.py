from PIL import Image
import cv2
import numpy as np

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
lbph = cv2.face_LBPHFaceRecognizer.create()
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


