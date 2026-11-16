# Step 1 : import libraries 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

# Step 2 : Load the Dataset
with np.load("mnist.npz",allow_pickle=True) as data:
    x_train,y_train = data["x_train"],data["y_train"]
    x_test,y_test = data["x_test"],data["y_test"]

# Step 3 : PreProcessing
# flattening + Normalization 
x_train = x_train.reshape(-1,784).astype('float32') / 255.0
x_test = x_test.reshape(-1,784).astype('float32') / 255.0

# One Hot Encoding
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Step 4 : Experimental Values
learning_rates = [0.001,0.01,0.1]
batch_sizes = [32,64,128]
accuracies=[]

# Step 5 : Model Training , compile , evaluation
for lr,bs in zip(learning_rates,batch_sizes):
    # model conf 
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128,activation='relu',input_shape=(784,)),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

    # model compile
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    # create callback
    tensorboard_callback = TensorBoard(log_dir=f"logs/experiment_{lr}_{bs}")
    
    # Training 
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=bs,
        callbacks=[tensorboard_callback],
        validation_data=(x_test,y_test)
    )

    # evaluate
    loss,accuracy = model.evaluate(x_test,y_test)
    print(accuracy)

# Step 6 : Tensorboard Visualization
%load_ext tensorboard
%tensorboard --logdir logs