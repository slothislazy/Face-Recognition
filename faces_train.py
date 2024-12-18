import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

DIR = 'Faces/train'
people = os.listdir(DIR)
print(people)

def augment_image(image):
    augmented_images = []
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Flipping
    augmented_images.append(cv.flip(image, 1))

    # Rotation with padding
    padded_image = cv.copyMakeBorder(image, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=0)
    for angle in [-30, 30]:
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(padded_image, M, (w, h))
        augmented_images.append(rotated)

    # Brightness
    for alpha in [0.9, 1.2]:
        brightened = cv.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented_images.append(brightened)

    # Scaling
    for scale in [0.9, 1.1]:
        scaled = cv.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        scaled = cv.resize(scaled, (w, h))
        augmented_images.append(scaled)

    # Blurring
    augmented_images.append(cv.GaussianBlur(image, (5, 5), 0))
    augmented_images.append(cv.medianBlur(image, 5))

    return augmented_images


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
            
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=1.3)
            gray = clahe.apply(gray)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(50,50))

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]

                features.append(faces_roi)
                labels.append(label)

                augmented_faces = augment_image(faces_roi)
                for aug_face in augmented_faces:
                    features.append(aug_face)
                    labels.append(label)


create_train()
print('Training Done')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)

print('Saving Model')
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
print('Done')