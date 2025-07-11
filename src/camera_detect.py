import cv2
import numpy as np
from keras.models import load_model
from collections import Counter

def detect_emotion():
    model = load_model('model_file.h5')

    video = cv2.VideoCapture(0)
    faceDetect = cv2.CascadeClassifier(
        r'C:\Users\PARTHA SARATHI\Python\movierecommendersystem\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

    labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                   3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    detected_emotions = []

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)
        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            current_emotion = labels_dict[label]
            detected_emotions.append(current_emotion)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, current_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(detected_emotions) >= 10:  # Stop after 10 faces
            break

    video.release()
    cv2.destroyAllWindows()

    if detected_emotions:
        final_detected_emotion = Counter(detected_emotions).most_common(1)[0][0]
    else:
        final_detected_emotion = None

    return final_detected_emotion
