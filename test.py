import numpy as np
import os
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

val_dir = 'Faces/val'
people = os.listdir(val_dir)

total_images = 0
total_correct = 0
person_correct = {person: 0 for person in people}
person_total = {person: 0 for person in people}

for person in people:
    person_dir = os.path.join(val_dir, person)
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv.imread(img_path)

        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # clahe = cv.createCLAHE(clipLimit=1.5)
        # gray = clahe.apply(gray)

        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=[50,50])

        person_total[person] += 1
        total_images += 1

        if len(faces_rect) == 0:
            print(f"No face detected in {img_path}")
            continue

        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]

            label, confidence = face_recognizer.predict(faces_roi)
            predicted_person = people[label]

            if predicted_person == person:
                total_correct += 1
                person_correct[person] += 1

            print(f"Image: {img_name} | True: {person} | Predicted: {predicted_person} | Confidence: {round(confidence, 2)}")

            cv.putText(img, predicted_person, (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
            cv.imshow('Detected Face', img)
            cv.waitKey(10)

cv.destroyAllWindows()

overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

for person in people:
    if person_total[person] > 0:
        person_accuracy = (person_correct[person] / person_total[person]) * 100
        print(f"Accuracy for {person}: {person_accuracy:.2f}% ({person_correct[person]}/{person_total[person]})")
    else:
        print(f"No images processed for {person}.")
