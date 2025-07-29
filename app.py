import tensorflow as tf
import os
import imghdr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define image extensions
img_exts = ["jpg", "jpeg", "png", "bmp"]

# Define data directory
data_dir = "data"

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    os.makedirs(os.path.join(data_dir, "happy"))
    os.makedirs(os.path.join(data_dir, "sad"))

# Image validation
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in img_exts:
                print("Image not in ext list {}".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Issue with image {}: {}".format(image_path, e))

# Load data
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# Preprocess data
data = data.map(lambda x, y: (x / 255, y))

# Split data
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), 1, activation="relu", input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), 1, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, (3, 3), 1, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

# Train the model
logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

callbacks = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
model_history = model.fit(train, epochs=20, validation_data=val, callbacks=[callbacks])

# Save the model
model.save("happy_or_sad_model.h5")

# Plot accuracy and loss
plt.plot(model_history.history["accuracy"])
plt.plot(model_history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["train", "val"])
plt.show()

plt.plot(model_history.history["loss"])
plt.plot(model_history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["train", "val"])
plt.show()

# Webcam prediction
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        resized_face = cv2.resize(face, (256, 256))
        reshaped_face = np.expand_dims(resized_face, axis=0) / 255.0
        prediction = model.predict(reshaped_face)

        if prediction > 0.5:
            cv2.putText(frame, "Happy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Sad", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
