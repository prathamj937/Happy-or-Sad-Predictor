import tensorflow as tf
import os
import imghdr
import cv2
import matplotlib.pyplot as plt

img_exts = ["jpg","jpeg","png","bmp"]
data = "data"

os.listdir(os.path.join(data,"happy"))

imgg = cv2.imread(os.path.join(data,"happy","1-2.jpg"))
print(imgg)

imgg.shape

plt.imshow(cv2.cvtColor(imgg,cv2.COLOR_BGR2RGB))

for image_class in os.listdir(data):
    for image in os.listdir(os.path.join(data, image_class)):
        image_path = os.path.join(data, image_class, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in img_exts:
                print(f"Image not in ext list {image_path}")
                os.remove(image_path)
        except Exception as e:
            print(f"Issue with image {image_path}: {e}")

import numpy as np
import matplotlib.pyplot as plt

data_dir = "/content/drive/MyDrive/data"  # Define a new variable for the data directory path
data = tf.keras.utils.image_dataset_from_directory(data_dir)

tf.keras.utils.image_dataset_from_directory??

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

batch[0].shape

data = data.map(lambda x,y: (x/255,y))

data_itr = data.as_numpy_iterator()

batch = data_itr.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

batch[0].max()

fig, ax = plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):
  ax[idx].imshow(img)
  ax[idx].title.set_text(batch[1][idx])

len(data)

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

val_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

!pip install tensorflow keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten

model = Sequential()

model.add(Conv2D(16,(3,3),1,activation="relu",input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1,activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

model.summary()

logs = "/content/logs"

callbacks = tf.keras.callbacks.TensorBoard(log_dir=logs)

model_history = model.fit(train,epochs=20,validation_data=val, callbacks=[callbacks])

import matplotlib.pyplot as plt

plt.plot(model_history.history["accuracy"])
plt.plot(model_history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["train","val"])

plt.plot(model_history.history["loss"])
plt.plot(model_history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["train","val"])

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized = tf.image.resize(rgb, (256,256))

    yhat = model.predict(np.expand_dims(resized/255, 0))

    if yhat > 0.5:
        cv2.putText(frame,'Happy', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame,'Sad', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
