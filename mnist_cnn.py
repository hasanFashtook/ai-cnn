import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load the dataset from the local file
base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_path, "mnist.npz")

with np.load(dataset_path, allow_pickle=True) as f:
    train_images, train_labels = f["x_train"], f["y_train"]
    test_images, test_labels = f["x_test"], f["y_test"]

train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
def create_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# Path to save/load the model
model_path = os.path.join(base_path, 'mnist_cnn_model.keras')
checkpoint_path = os.path.join(base_path, 'best_model.keras')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')

# Check if the model already exists
if os.path.exists(model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    print("Loaded trained model.")
else:
    # Create a new model and train it
    model = create_model()
    model.fit(
        train_images, train_labels, epochs=5, validation_data=(test_images, test_labels),
        callbacks=[checkpoint]
    )
    # Save the trained model
    model.save(model_path)
    print("Model trained and saved.")

# Load the best model saved by ModelCheckpoint
model = tf.keras.models.load_model(checkpoint_path)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Plot some predictions
predictions = model.predict(test_images)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Label: {test_labels[i]}, Pred: {predictions[i].argmax()}")
plt.show()
