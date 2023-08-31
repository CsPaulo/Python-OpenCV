import cv2
import numpy as np
import os
import zipfile
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(X)

# Bases de treinanmento e teste
from sklearn.model_selection import train_test_split

x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.2, ramdom_state= 1)

# Construção e treinamento da rede neural
network1 = tf.keras.models.Sequential()
network1.add(tf.keras.layers.Dense(input_shape = (16384,), units = 8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units = 8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

network1.compile(optimzer='Adam', loss='binary_crosentropy', metrics=['accuracy'])

historic = network1.fit(x_training, y_training, epochs = 50)

# Avaliação da rede neural
'''
historic.history.keys()
plt.plot(historic.history['loss'])
plt.plot(historic.history['accuracy])

previsoes = network1.predict(X_teste)

from sklearn.metrics import accuracy_score
accuracy_score(y_teste, previsoes)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_teste, previsoes)

sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(y_teste, previsoes))
'''

# Salvar e carregar a imagem









