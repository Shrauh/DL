import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

with np.load("mnist.npz",allow_pickle=True) as data:
    x_train,y_train = data['x_train'],data['y_train']
    x_test,y_test = data['x_test'],data['y_test']

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# Flatten images
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.1,
    random_state=42
)

print("Training:", x_train.shape)
print("Validation:", x_val.shape)


model1 = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model1.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_val, y_val)
)

print("Normal model training completed!")

model2 = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(10, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model2.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_val, y_val)
)

print("L2 model training completed!")

model3 = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model3.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history3 = model3.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_val, y_val)
)

print("Dropout model training completed!")

# Evaluate models on test data
loss1, acc1 = model1.evaluate(x_test, y_test, verbose=0)
loss2, acc2 = model2.evaluate(x_test, y_test, verbose=0)
loss3, acc3 = model3.evaluate(x_test, y_test, verbose=0)

print("Normal Model Accuracy:", acc1 * 100, "%")
print("L2 Model Accuracy:", acc2 * 100, "%")
print("Dropout Model Accuracy:", acc3 * 100, "%")

models = ['Normal', 'L2', 'Dropout']
accuracy = [acc1 * 100, acc2 * 100, acc3 * 100]

plt.bar(models, accuracy)

plt.title("Comparison of Model Accuracy")
plt.xlabel("Regularization Techniques")
plt.ylabel("Accuracy (%)")

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(history1.history['val_loss'], label='Normal')
plt.plot(history2.history['val_loss'], label='L2')
plt.plot(history3.history['val_loss'], label='Dropout')

plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(history1.history['val_accuracy'], label='Normal')
plt.plot(history2.history['val_accuracy'], label='L2')
plt.plot(history3.history['val_accuracy'], label='Dropout')

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.show()