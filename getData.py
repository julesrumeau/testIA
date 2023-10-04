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
from sklearn.preprocessing import StandardScaler

print(tf.__version__)

def getData(x = 20, y = 10) :
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
        # y_g = g + (i % 2)
        buffer = y*[0]
        # buffer[y_i] = 1
        buffer[y_j] += 3
        # buffer[y_g] += m:
        data.append(buffer)
        i += 1
    # print(data)
    return data


data_size = 10000
pourcentage_train_set = 0.8

scaler = StandardScaler()
train_set = [] 
data_set = [] 
for i in range (int(data_size * pourcentage_train_set)) :
    train_set.append(getData(28,28))
for i in range (int(data_size * (1 - pourcentage_train_set))) :
    data_set.append(getData(28,28))

train_set = np.array(train_set)
data_set = np.array(data_set)


latent_dim = 10

# Encodeur
inputs = keras.Input(shape=(28, 28))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation="sigmoid")(x)
x = layers.Dense(128, activation="sigmoid")(x)
x = layers.Dense(64, activation="sigmoid")(x)
x = layers.Dense(latent_dim, activation="sigmoid")(x)  
encoder = keras.Model(inputs, x, name="encoder")

# Décodeur
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(64, activation="sigmoid")(latent_inputs)
x = layers.Dense(128, activation="sigmoid")(x)
x = layers.Dense(256, activation="sigmoid")(x)
x = layers.Dense(28 * 28, activation="sigmoid")(x) 
outputs = layers.Reshape((28, 28))(x)


decoder = keras.Model(latent_inputs, outputs, name = "decoder")

# decoder.summary()

inputs = keras.Input(shape=(28,28))

latents = encoder(inputs)
outputs = decoder(latents)

ae = keras.Model(inputs,outputs,name="ae")

ae.compile(optimizer="adam", loss ='mse',metrics =["accuracy"])
# ae.compile(optimizer="adam", loss ='binary_crossentropy',metrics =["accuracy"])

callback_bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath='/models/best_model.h5',
                                                        verbose = 0, save_best_only=True)


batch_size = 128
epochs = 500
history = ae.fit(train_set, train_set,
                 batch_size = batch_size,
                 epochs = epochs,
                 verbose = 1,
                 validation_data = (data_set, data_set),
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
print(data_set[0].reshape(1,28,28))
res = ae.predict(data_set[0].reshape(1,28,28))
res = res[0]
plt.imshow(data_set[0:1][0])
plt.show()

plt.imshow(res)
plt.show()
