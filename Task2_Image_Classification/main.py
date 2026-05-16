import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("Training Images Shape:", train_images.shape)
print("Testing Images Shape:", test_images.shape)

plt.imshow(train_images[0])
plt.title("Sample Training Image")
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("Test Accuracy:", test_accuracy)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

predictions = model.predict(test_images)

image_index = 25

predicted_label = np.argmax(predictions[image_index])

print("Predicted Clothing:", class_names[predicted_label])

plt.imshow(test_images[image_index])
plt.title(f"Predicted: {class_names[predicted_label]}")
plt.show()

model.save("fashion_model.h5")

print("Model saved successfully")