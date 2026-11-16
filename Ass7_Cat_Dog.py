import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory


# Step 2 : Load the Dataset
train_dir = "cats_vs_dogs"
test_dir = "cats_vs_dogs"
train_data = image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=64,
    label_mode='binary'
)

test_data = image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=64,
    label_mode='binary'
)

# Step 3 : Preprocessing
train_data = train_data.map(
    lambda x, y: (preprocess_input(x), y)
)

test_data = test_data.map(
    lambda x, y: (preprocess_input(x), y)
)

# Step 4 : Load VGG16 Model and Freeze Layers
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Step 5 : Model Configuration
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 6 : Feature Extraction Model Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training
history1 = model.fit(
    train_data,
    epochs=3,
    validation_data=test_data
)

# Evaluation
loss, accuracy = model.evaluate(test_data)
print("Feature Extraction Loss:", loss)
print("Feature Extraction Accuracy:", accuracy)

# Step 7 : Fine Tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Compile Again
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training
history2 = model.fit(
    train_data,
    epochs=3,
    validation_data=test_data
)

# Evaluation
loss, accuracy = model.evaluate(test_data)
print("Fine Tuning Loss:", loss)
print("Fine Tuning Accuracy:", accuracy)

# Step 8 : Plotting
plt.plot(
    history1.history['accuracy'],
    label='Feature Extraction Accuracy'
)
plt.plot(
    history2.history['accuracy'],
    label='Fine Tuning Accuracy'
)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('VGG16 Accuracy Comparison')
plt.legend()
plt.grid()
plt.show()