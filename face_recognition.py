import numpy as np
import os
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = os.listdir('Faces/val')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('Faces/val/Leonardo DiCaprio/5.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
clahe = cv.createCLAHE(clipLimit=1.5)
gray = clahe.apply(gray)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=[250,250])

for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Person = {people[label]}: Confidence = {round(confidence, 2)}')

        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
