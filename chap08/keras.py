from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

conv1 = tf.keras.Sequential()
conv1.add(Conv2D(10, (3,3),activation = 'relu',padding = 'same', input_shape = (28,28,1)))

conv1.add(MaxPooling2D((2,2)))
conv1.add(Flatten())

conv1.add(Dense(100,activation = 'relu'))
conv1.add(Dense(10, activation = 'softmax'))
conv1.summary()

from sklearn.model_selection import train_test_split

(x_train_all, y_train_all) , (x_text,y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print(x_train_all.shape,y_train_all.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42)

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)

x_train = x_train.reshape(-1,28,28,1)
x_val = x_val.reshape(-1,28,28,1)

# print(x_train.shape)
x_train =  x_train / 255
x_val = x_val / 255

conv1.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
history = conv1.fit(x_train,y_train_encoded,epochs=20,validation_data=(x_val,y_val_encoded))