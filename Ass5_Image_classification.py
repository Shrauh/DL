# Step 1 : import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Step 2 : load the data 
with np.load("mnist.npz",allow_pickle=True) as data:
    x_train,y_train = data['x_train'],data['y_train']
    x_test,y_test = data['x_test'],data['y_test']

# Step 3 : Preprocessing

#Normalize 
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

#reshape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# one hot encoding 
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Step 4 : Data Augmentation 
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Step 5 : Model conf 
model = tf.keras.Sequential([
    data_augmentation,
    #first conv layer
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    # second conv layer 
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

# Step 6 : Model compile
model.compile(
    optimizer = 'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 7 : Train the model 
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data = (x_test,y_test)
)

# Step 8 : Evaluate the model 
loss,accuracy = model.evaluate(x_test,y_test)
print(loss,accuracy)

# Step 9 : Plotting 
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val accuracy')
plt.legend()
plt.grid()
plt.show()

