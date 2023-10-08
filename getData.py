import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import yfinance as yf

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

size_past = 60
data_size = 1000
pourcentage_data_train =  0.7
pourcentage_data_test =  0.2
pourcentage_data_evaluate =  0.1


batch_size = 128
epochs = 10000

# Définissez la date de début et la date de fin pour la période de 30 ans
debut = "1993-01-01"
fin = "2023-01-01"

# Utilisez la fonction Ticker pour obtenir les données du CAC 40 depuis Yahoo Finance
cac40 = yf.Ticker("^FCHI")
# cac40 = yf.Ticker("AAPL")

# Obtenez les données historiques
donnees_cac40 = cac40.history(period="30y", start=debut, end=fin)


donnees_cac40 = donnees_cac40["Open"]
donnees_cac40 = np.array(donnees_cac40)

mean_de_base = donnees_cac40.mean()
std_de_base = donnees_cac40.std()
donnees_de_base = donnees_cac40

donnees_cac40 = [(x - mean_de_base) / std_de_base for x in donnees_cac40]

donnes_retrouver = [(x * std_de_base) + mean_de_base for x in donnees_cac40]



# Affichez les premières lignes des données
nbr_data_total = len(donnees_cac40)
print(nbr_data_total)

data = [donnees_cac40[i:i+size_past + 1] for i in range(0, len(donnees_cac40), size_past)]
data = data[:-1]

nbr_data_paquet = len(data)
print(nbr_data_paquet)


target = []
data2 = []
for x in data :
    target.append(x[-1])
    data2.append(x[:-1])

data = data2
data = np.array(data)
target = np.array(target)

print(data.shape)
print(target.shape)

print("------------")

# def getData(taille) :
#     data = []

#     # exit()
#     train = data[:-1]
#     nMoins1 = data[-1]
   
#     return train, nMoins1

# def getData(x) :
#     data = []
#     for x_i in range (x) :
#         i = 2 * x_i**2 + 8 * random.random()
#         data.append(i)


#     data = np.array(data)
#     data = [(x - data.mean()) / data.std() for x in data]

#     # print(data)
#     train = data[:-1]
#     nMoins1 = data[-1]
#     nMoins2 = train[-1]

#     # print(nMoins1)

#     # print(train)
#     # diff = (nMoins2 - nMoins1) / nMoins2 
#     #train / target
#     return train, nMoins1


# scaler = StandardScaler()
taille_data_train = int(nbr_data_paquet * pourcentage_data_train)
taille_data_test = int(nbr_data_paquet * pourcentage_data_test)
taille_data_evaluate = int(nbr_data_paquet * pourcentage_data_evaluate)

# print("taille_data_train", taille_data_train)
# print("taille_data_test", taille_data_test)
# print("taille_data_evaluate", taille_data_evaluate)

data_train = data[:taille_data_train] 
data_test = data[taille_data_train:taille_data_train+taille_data_test] 
data_evaluate = data[taille_data_evaluate:] 


target_train = target[:taille_data_train] 
target_test = target[taille_data_train:taille_data_train+taille_data_test] 
target_evaluate = target[taille_data_evaluate:] 
# print(target_train.shape)
# print(target_test.shape)
# print(target_evaluate.shape)

# for i in range (int(data_size * pourcentage_data_train)) :
#     res = getData(size_past + 1)
#     data_train.append(res[0])
#     target_train.append(res[1])

# for i in range (int(data_size * pourcentage_data_test)) :
#     res = getData(size_past + 1)
#     data_test.append(res[0])
#     target_test.append(res[1])

# for i in range (int(data_size * pourcentage_data_evaluate)) :
#     res = getData(size_past + 1)
#     data_evaluate.append(res[0])
#     target_evaluate.append(res[1])

data_train = np.array(data_train)
data_test = np.array(data_test)
data_evaluate = np.array(data_evaluate)
target_train = np.array(target_train)
target_test = np.array(target_test)
target_evaluate = np.array(target_evaluate)


latent_dim = 1

# Encodeur
inputs = keras.Input(shape=(size_past))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation="sigmoid")(x)
x = layers.Dense(128, activation="sigmoid")(x)
x = layers.Dense(64, activation="sigmoid")(x)
x = layers.Dense(latent_dim, activation="linear")(x)  
encoder = keras.Model(inputs, x, name="encoder")

# decoder.summary()

inputs = keras.Input(shape=(size_past))

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
rand = random.randint(0,taille_data_evaluate-1)
newDataValue = data_evaluate[rand]
newDataValue = np.array(newDataValue)

newTargetValue = target_evaluate[rand]
newTargetValue = np.array(newTargetValue)

res = ae.predict(newDataValue.reshape(1,size_past))
res = res[0]
print(res)
res = np.array(res)


# Créez une figure et un axe
fig, ax = plt.subplots()

# Tracez les données passées en bleu sous forme de courbe
ax.plot(range(0, size_past), newDataValue, label='Passé', color='blue', marker='o', markersize=6)

# Tracez la prédiction future en rouge
ax.plot(size_past, res, label='Futur_predict', color='red', marker='o', markersize=6)
ax.plot(size_past, newTargetValue, label='Futur_real', color='green', marker='o', markersize=6)

# Ajoutez une légende pour distinguer les courbes
ax.legend()
# Affichez le graphique




# data_evaluate_retrouver = [(x * std_de_base) + mean_de_base for x in data_evaluate]


predict = ae.predict(data_evaluate)
predict = np.array(predict)
predict = predict.ravel()

target_evaluate = [(x * std_de_base) + mean_de_base for x in target_evaluate]
data_evaluate = [(x * std_de_base) + mean_de_base for x in data_evaluate]
predict = [(x * std_de_base) + mean_de_base for x in predict]

target_evaluate = np.array(target_evaluate)
predict = np.array(predict)


diff = (predict - target_evaluate) / predict *100



moyenne = np.mean(diff)
ecart_type = np.std(diff)

print(f"Moyenne : {moyenne}")
print(f"Écart-type : {ecart_type}")



target_real_moins_1 = []

for data in data_evaluate :
    target_real_moins_1.append(data[-1])

target_real_moins_1 = np.array(target_real_moins_1)

# print(diff[rand])
# print((target_real_moins_1[rand] - mean_de_base) / std_de_base)
# print((target_evaluate[rand] - mean_de_base) / std_de_base)
# print(target_evaluate[rand])
# print(data_evaluate[rand][-1] == target_evaluate[rand])
sell = []
buy = []
for i in range(0,len(target_real_moins_1)) :
    if target_real_moins_1[i] >= target_evaluate[i] :
        sell.append(diff[i])
    else :
        buy.append(diff[i])


# print("sell", sell[rand])
# print("buy", buy[rand])


plt.show()


plt.hist(diff)
plt.show()

plt.hist(sell)
plt.title("sell")
plt.show()
plt.hist(buy)
plt.title("buy")
plt.show()

# plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, première sous-fenêtre
# plt.imshow(data_test[0:1][0])
# # plt.show()

# plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, première sous-fenêtre
# plt.imshow(res.reshape(1))


# res = ae.predict(data_evaluate)

# res = np.array(res)
# target_evaluate = target_evaluate.astype(np.float32)
# res = res.astype(np.float32)

# print(res.shape)
# print(target_evaluate.shape)

# print("------------")
# print(res[0])
# print(target_evaluate[0])



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

