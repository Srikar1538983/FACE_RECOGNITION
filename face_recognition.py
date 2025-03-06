import cv2
import numpy as np

# Load the trained face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognizer.yml')

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the label map
label_map = {
    0: "jaya lakshmi",    # Replace with actual names from your dataset
    1: "yerroju srikar",  # Replace with actual names
}

# Open webcam
video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 50:
            name = label_map.get(label, "Unknown")
        else:
            name = "Unknown"
        
        # Draw a rectangle around the face and put the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{name} ({round(100 - confidence, 2)}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame with detected faces
    cv2.imshow("Face Recognition", frame)

    # Break the loop when 'a' is pressed
    if cv2.waitKey(10) == ord('a'):
        break

# Release the webcam and close the window
video_cap.release()
cv2.destroyAllWindows()