# Step 1 : import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2 : Load the dataset
with np.load("imdb.npz",allow_pickle=True) as data:
    x_train,y_train = data['x_train'],data['y_train']
    x_test,y_test = data['x_test'],data['y_test']

# Step 3 : Preprocessing 
x_train = pad_sequences(x_train,maxlen=200)
x_test = pad_sequences(x_test,maxlen=200)# Step 4 : Vocab Size 
vocab_size = max(np.max(x_train),np.max(x_test)) + 1
print(vocab_size)

# Step 4 : Vocab Size 
vocab_size = max(np.max(x_train),np.max(x_test)) + 1
print(vocab_size)

# Step 5 : Model conf
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# Step 6 : Model compile
model.compile(
    optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 7 : Train the model 
history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test,y_test)
)

# Step 8 : Evaluate Model
loss,accuracy = model.evaluate(x_test,y_test)
print(loss,accuracy)

# Step 9 : Plotting
plt.plot(history.history['accuracy'],label='accuracy')
plt.legend()
plt.grid()
plt.show()

# Step 10 : Examples
predictions = model.predict(x_test[:5])

for i in range(5):
    if predictions[i][0]>=0.5:
        print("Predicted : Positive")
    else:
        print("Predicted : Negative")
    print("Actual : ","Positive" if y_test[i]==1 else "Negative")