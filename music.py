import librosa
import numpy as np
import os
import soundfile as sf
import resampy
import sounddevice as sd  # Assurez-vous d'installer la bibliothèque sounddevice si elle n'est pas déjà installée
import keyboard
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


longueur = 100000

np.set_printoptions(threshold=np.inf)  # Cela permet d'afficher l'intégralité du tableau sans aucune limitation



def play(data):
    sd.play(data)
    while sd.get_stream().active and not keyboard.is_pressed('space'):
        continue
    sd.stop()

def encode(file_name) :
    # Charger le fichier audio
    current_directory = os.getcwd()

    relative_path = os.path.join(current_directory, file_name)
    data, _ = sf.read(relative_path)
    # data = data[:1000000]
    # Rééchantillonner les données audio au taux d'échantillonnage cible
    print(data.shape)


    # play(data)

    return data

    # Vous pouvez ensuite utiliser la méthode np.save pour enregistrer le tableau
    # np.save('fichier_audio_spectrogram.npy', spectrogram_normalized)

def splitData(data) :
    
    split_data = [data[i:i + longueur, :] for i in range(0, data.shape[0], longueur)]


    # Vérifier si la taille du dernier morceau est différente de 1000000 et le retirer si c'est le cas
    if len(split_data[-1]) != longueur:
        split_data = split_data[:-1]
    play(data)
    # Vérifier les dimensions des morceaux découpés
    # for i, chunk in enumerate(split_data):
    #     print(f"Chunk {i + 1} : {chunk.shape}")


    return split_data



def getData(listeSon) :
    split_data = []
    for son in listeSon :
        file_name = son
        data = encode(file_name)
        data = splitData(data)
        split_data = split_data + data

    return split_data



listeSon = ["feather.mp3","kaaris.mp3"]
listeSon = ["feather.mp3"]
data = getData(listeSon)
data = np.array(data)
print(data.shape)
exit()

data_size = 1000
pourcentage_data_train =  0.7
pourcentage_data_test =  0.2
pourcentage_data_evaluate =  0.1


batch_size = 1
epochs = 10000


taille_data_train = int(data_size * pourcentage_data_train)
taille_data_test = int(data_size * pourcentage_data_test)
taille_data_evaluate = int(data_size * pourcentage_data_evaluate)


data_train = data[:taille_data_train] 
data_test = data[taille_data_train:taille_data_train+taille_data_test] 
data_evaluate = data[taille_data_evaluate:] 


target_train = data[:taille_data_train] 
target_test = data[taille_data_train:taille_data_train+taille_data_test] 
target_evaluate = data[taille_data_evaluate:] 

data_train = np.array(data_train)
data_test = np.array(data_test)
data_evaluate = np.array(data_evaluate)
target_train = np.array(target_train)
target_test = np.array(target_test)
target_evaluate = np.array(target_evaluate)





latent_dim = 1

# Encodeur
inputs = keras.Input(shape=(longueur, 2))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation="sigmoid")(x)
x = layers.Dense(128, activation="sigmoid")(x)
x = layers.Dense(64, activation="sigmoid")(x)
x = layers.Dense(latent_dim, activation="linear")(x)  
encoder = keras.Model(inputs, x, name="encoder")


inputs = keras.Input(shape=(latent_dim))
x = layers.Dense(latent_dim, activation="linear")(inputs)  
x = layers.Dense(64, activation="sigmoid")(x)
x = layers.Dense(128, activation="sigmoid")(x)
x = layers.Dense(256, activation="sigmoid")(x)
x = layers.Dense(longueur * 2, activation="sigmoid")(x)
x = layers.Reshape((longueur,2))(x)
decoder = keras.Model(inputs, x, name="decoder")


inputs = keras.Input(shape=(longueur, 2))

latents = encoder(inputs)
outputs = decoder(latents)

ae = keras.Model(inputs,outputs, name="ae")

ae.compile(optimizer="adam", loss ='mse',metrics =["accuracy"])


history = ae.fit(data_train, target_train,
                 batch_size = batch_size,
                 epochs = epochs,
                 verbose = 1,
                 validation_data = (data_test, target_test),
                #  callbacks = callback_bestmodel
                 )


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


























