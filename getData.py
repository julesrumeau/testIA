import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
import os

# def getMusiqueArray() :

#     current_directory = os.getcwd()

#     file_name = "feather.wav"

#     relative_path = os.path.join(current_directory, file_name)

#     # Charger le fichier audio (assurez-vous qu'il est au format WAV ou convertissez-le)
#     sample_rate, audio_data = wavfile.read(relative_path)

#     # Définir la durée de chaque segment en millisecondes
#     segment_duration_ms = 1  # Par exemple, 1 milliseconde

#     # Convertir la durée du segment en nombre d'échantillons
#     segment_duration_samples = int(segment_duration_ms * sample_rate / 1000)

#     # Calculer combien de segments peuvent être extraits
#     num_segments = len(audio_data) // segment_duration_samples

#     # Découper l'audio en segments de même taille
#     segments = np.array_split(audio_data[:num_segments * segment_duration_samples], num_segments)

#     # Initialiser une liste pour stocker les fréquences pour chaque segment
#     frequencies_per_segment = []

#     # Calculer les fréquences pour chaque segment
#     for segment in segments:
#         frequencies = np.fft.fft(segment)
#         frequencies_per_segment.append(frequencies)

#     # Initialiser une liste pour stocker les fréquences mono
#     frequencies_mono = []

#     # Parcourir tous les segments
#     for segment in frequencies_per_segment:
#         # Moyenne des canaux gauche et droit pour obtenir un canal mono
#         mono_segment = (segment[:, 0] + segment[:, 1]) / 2
#         frequencies_mono.append(mono_segment)

#     print(frequencies_mono)
#     frequencies_mono = np.array(frequencies_mono)
#     print(frequencies_mono.shape)
#     exit()
#     return frequencies_mono


# getMusiqueArray()


def getData(x, y) :
    data = []
    i = random.randint(0,9)
    # i = 2
    j = random.randint(0,8)
    g = random.randint(0,8)
    k = random.randint(0,10)
    m = random.randint(0,22)
    for x_i in range (x) :
        y_i = i % y
        y_j = j + (i % 2)
        y_g = g + (i % 2)
        buffer = y*[0]
        buffer[y_i] = 0.5
        # buffer[y_j] = 3
        # buffer[y_g] = m
        data.append(buffer)
        i += 1
    
    return data[:-1], data[-1]


data_size = 1000
pourcentage_data_train = 0.7
pourcentage_data_test = 0.2
pourcentage_data_evaluate = 0.2
batch_size = 128
epochs = 100

# scaler = StandardScaler()
data_train = [] 
data_test = [] 
data_evaluate = [] 

target_train = [] 
target_test = [] 
target_evaluate = [] 

for i in range (int(data_size * pourcentage_data_train)) :
    res = getData(29,28)
    data_train.append(res[0])
    target_train.append(res[1])

for i in range (int(data_size * pourcentage_data_test)) :
    res = getData(29,28)
    data_test.append(res[0])
    target_test.append(res[1])

for i in range (int(data_size * pourcentage_data_evaluate)) :
    res = getData(29,28)
    data_evaluate.append(res[0])
    target_evaluate.append(res[1])

data_train = np.array(data_train)
data_test = np.array(data_test)
data_evaluate = np.array(data_evaluate)
target_train = np.array(target_train)
target_test = np.array(target_test)
target_evaluate = np.array(target_evaluate)


latent_dim = 28

# Encodeur
inputs = keras.Input(shape=(28, 28))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation="sigmoid")(x)
x = layers.Dense(128, activation="sigmoid")(x)
x = layers.Dense(64, activation="sigmoid")(x)
x = layers.Dense(latent_dim, activation="sigmoid")(x)  
encoder = keras.Model(inputs, x, name="encoder")

# decoder.summary()

inputs = keras.Input(shape=(28,28))

latents = encoder(inputs)

ae = keras.Model(inputs,latents,name="ae")

ae.compile(optimizer="adam", loss ='mse',metrics =["accuracy"])
# ae.compile(optimizer="adam", loss ='binary_crossentropy',metrics =["accuracy"])

callback_bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath='/models/best_model.h5',
                                                        verbose = 0, save_best_only=True)



history = ae.fit(data_train, target_train,
                 batch_size = batch_size,
                 epochs = epochs,
                 verbose = 1,
                 validation_data = (data_test, target_test),
                 callbacks = callback_bestmodel
                 )


ae.save('test.h5')
loss_curve = history.history["loss"]
val_loss_curve = history.history["val_loss"]
acc_curve = history.history["accuracy"]
val_acc_curve = history.history["val_accuracy"]

plt.plot(loss_curve, label = 'loss')
plt.plot(val_loss_curve, label = 'val_loss')
plt.legend()
plt.title("Loss")
plt.show()

plt.plot(acc_curve, label = 'acc')
plt.plot(val_acc_curve, label = 'val_acc')
plt.legend()
plt.title("Acc")
plt.show()



# Ensuite, vous pouvez passer l'image redimensionnée à votre modèle
print(data_test[0].reshape(1,28,28))
res = ae.predict(data_test[0].reshape(1,28,28))
res = res[0]

plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, première sous-fenêtre
plt.imshow(data_test[0:1][0])
# plt.show()

plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, première sous-fenêtre
plt.imshow(res.reshape(1,28))


res = ae.predict(data_evaluate)

res = np.array(res)
target_evaluate = target_evaluate.astype(np.float32)
res = res.astype(np.float32)

print(res.shape)
print(target_evaluate.shape)

print("------------")
print(res[0])
print(target_evaluate[0])

plt.tight_layout()  # Pour éviter que les titres se chevauchent
plt.show()


'''
marche pas mais normal, normalisé les resultat de prediction, pas de dfloat a la con, mais regarder vidéo playlist pour checker
comment faire pour que neuronne ressorte int genre porucen,tage
'''
# confusion_matirx = tf.math.confusion_matrix(
#     labels = target_evaluate[0:2],
#     predictions = res[0:2],
#     num_classes=2
#     )
# print(confusion_matirx)


# # Exemple de données de prédiction et de vérité terrain (à adapter à vos données)
# predictions = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0.7])  # Exemple de prédictions binaires (0 ou 1)
# ground_truth = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0.5, 1])  # Exemple d'étiquettes de vérité terrain (0 ou 1)

# # Assurez-vous que les données sont de type int32
# predictions = predictions.astype(np.int32)
# ground_truth = ground_truth.astype(np.int32)

# # Utilisez la fonction tf.math.confusion_matrix
# confusion_matrix = tf.math.confusion_matrix(ground_truth, predictions, num_classes=2)

# print(confusion_matrix)

