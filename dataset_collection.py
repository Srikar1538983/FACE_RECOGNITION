import cv2
import os

# Create a folder to store datasets
dataset_path = 'dataset'

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the recognizer (will be used for training later)
video_cap = cv2.VideoCapture(0)

# Ask user for the person's name to save images under that name folder
person_name = input("Enter your name: ")

# Create directory for the person if it doesn't exist
person_dir = os.path.join(dataset_path, person_name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

count = 0
while True:
    # Read frame from webcam
    ret, frame = video_cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If faces are detected, save them
    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y + h, x:x + w]
        cv2.imwrite(f"{person_dir}/image_{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow("Face Capture", frame)
    
    # Break loop if 50 images are collected or 'q' is pressed
    if count >= 50:
        print(f"{person_name}'s images collected successfully!")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()