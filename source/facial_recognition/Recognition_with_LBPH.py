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