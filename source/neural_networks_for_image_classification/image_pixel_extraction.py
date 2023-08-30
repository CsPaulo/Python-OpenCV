import cv2
import numpy as np
import os
import zipfile
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Extração dos pixels das imagens
'''
path = 'Datasets/homer_bart_1.zip'
zip_object = zipfile.ZipFile(file = path, mode = 'r')
zip_object.extractall('./')
zip_object.close()
'''
directory = 'homer_bart_1'
archives = [os.path.join(directory, f)  for f in os.sorted(os.listdir(directory))]

width, height = 128, 128
images = []
classes = []
for image_way in archives:
    try:
        image = cv2.imread(image_way)
        (H, W) = image.shape[:2]
    except:
        continue

    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.racel()
    
    images.append(image)
    name_image = os.path.basename(os.path.normpath(image_way))
    if name_image.startswith('b'):
        classe = 0
    else:
        classe = 1
    
    classes.append(classe)

x = np.asarray(imagens)
y = np.asarray(classes)

# Normalização dos pixels
scaler = MinMaxScaler()
x = scaler.fit_transform(X)


