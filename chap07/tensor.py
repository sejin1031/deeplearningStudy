import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split


model = Sequential()
model.add(Dense(100,activation='sigmoid',input_shape=(784,)))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
(x_train_all, y_train_all) , (x_text,y_test) = tf.keras.datasets.fashion_mnist.load_data()


x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42)
x_train = x_train / 255
x_val = x_val / 255

x_train = x_train.reshape(-1,784)
x_val = x_val.reshape(-1,784)

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)

history = model.fit(x_train,y_train_encoded,epochs=40,validation_data=(x_val,y_val_encoded))
