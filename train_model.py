import cv2
import os
import numpy as np

# Initialize face recognizer and face cascade classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the dataset
dataset_path = 'dataset'

# Collect faces and labels
faces = []
labels = []
label_map = {}
current_label = 0

# Loop through the dataset and assign labels
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if os.path.isdir(person_folder):
        label_map[current_label] = person_name
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(current_label)
        current_label += 1

# Train the recognizer with the collected faces
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save('face_recognizer.yml')
print("Training completed and model saved!")