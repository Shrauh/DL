
# Step I: Import the libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical


# Step II: Load the Local MNIST Dataset
data = np.load('mnist.npz')

x_train = data['x_train']
y_train = data['y_train']

x_test = data['x_test']
y_test = data['y_test']

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# Step III: Preprocessing

# Flattening + Normalization
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("After preprocessing:")
print("X Train:", x_train.shape)
print("X Test:", x_test.shape)
print("Y Train:", y_train.shape)
print("Y Test:", y_test.shape)


# Step IV: Model Configuration
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(264, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# Step V: Model Compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Step VI: Train the Model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)


# Step VII: Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")


# Step VIII: Testing
predicted = model.predict(x_test[:1])

predicted_class = tf.argmax(predicted[0]).numpy()
actual_class = tf.argmax(y_test[0]).numpy()

print("Predicted Class:", predicted_class)
print("Actual Class:", actual_class)


# Step IX: Visualize Loss Curve
plt.figure(figsize=(6, 4))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()


# Step X: Visualize Accuracy Curve
plt.figure(figsize=(6, 4))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Curve")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()