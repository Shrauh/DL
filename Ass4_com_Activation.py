# Step 1 : import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

# Step 2 : Load the data
# (x_train,y_train),(x_test,y_test) = mnist.load_data()
with np.load("mnist.npz",allow_pickle=True) as data:
    x_train,y_train = data['x_train'],data['y_train']
    x_test,y_test = data['x_test'],data['y_test']

# Step 3 : Preprocessing
# flattening + Normalization 
x_train = x_train.reshape(-1,784).astype('float32')/255.0
x_test = x_test.reshape(-1,784).astype('float32')/255.0

#one hot encoding 
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Step 4 : configurations
activations = ['sigmoid','tanh','relu']
accuracies = []
times = []

# Step 5 : Model Training 
for act in activations:
    #conf 
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128,activation=act,input_shape=(784,)),
        tf.keras.layers.Dense(64,activation=act),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

    # compile
    model.compile(
        optimizer = 'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # train 
    start = time.time()
    model.fit(x_train,y_train,epochs=5,batch_size=64)
    times.append(time.time()-start)

    #evaluate
    loss,accuracy = model.evaluate(x_test,y_test)
    accuracies.append(accuracy)

plt.bar(activations,accuracies,label='accuracies per activation')
plt.legend()
plt.show()

plt.bar(activations,times,label='time per activation')
plt.legend()
plt.show()

x = tf.constant([-2.0,-1.0,0.0,1.0,2.0])
s = tf.sigmoid(x)
t = tf.tanh(x)
plt.plot(x,s*(1-s),label='derivative of sigmoid')
plt.plot(x,1-t**2,label='derivative of tanh')
plt.plot(x,tf.cast(x>0,tf.float32),label='derivative of relu')
plt.legend()
plt.grid()
plt.plot()