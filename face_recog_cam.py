import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = os.listdir('Faces/val')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        
        if confidence < 10:
        
            cv.putText(frame, f'Unknown ({confidence:.2f})', (x, y - 10), 
                    cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
        
            cv.putText(frame, f'{people[label]} ({confidence:.2f})', (x, y - 10), 
                    cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

    cv.imshow('Live Camera - Face Recognition', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
