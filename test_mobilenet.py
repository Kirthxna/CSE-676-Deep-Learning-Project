import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from mobilenet_model import *


emotion_categories = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
category_count = len(emotion_categories)



model = MobileNetV2(input_channels=3, num_classes=category_count, compute_device='cpu')

# Load your trained model (ensure the model file path is correct)
model_path = 'mobilenetv2.pth'
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    # Draw rectangle around the faces and predict the emotion
    for (x, y, w, h) in faces:
        roi_gray = frame[y:y+h, x:x+w]
        roi = transform(roi_gray).unsqueeze(0)
        
        with torch.no_grad():
            preds = model(roi)
            emotion = emotion_categories[torch.argmax(preds).item()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
