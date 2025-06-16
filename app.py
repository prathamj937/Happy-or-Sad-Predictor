import cv2
from keras.models import load_model
import numpy as np

model = load_model('imageClassification.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face = cv2.resize(frame, (256, 256))
    normalized = face/255.0
    expanded = np.expand_dims(normalized, axis=0)

    prediction = model.predict(expanded)
    label = "Happy 😊" if prediction > 0.5 else "Sad 😢"

    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow("Mood Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()