import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from math import ceil
from effiecientnet_model import *


emotion_categories = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

category_count = len(emotion_categories)


efficient_net_config = {
    "b0" : (1.0, 1.0, 224, 0.2),
    "b1" : (1.0, 1.1, 240, 0.2),
    "b2" : (1.1, 1.2, 260, 0.3),
    "b3" : (1.2, 1.4, 300, 0.3),
    "b4" : (1.4, 1.8, 380, 0.4),
    "b5" : (1.6, 2.2, 456, 0.4),
    "b6" : (1.8, 2.6, 528, 0.5),
    "b7" : (2.0, 3.1, 600, 0.5)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define the device

version = 'b0'
width_mult, depth_mult, res, dropout_rate = efficient_net_config[version]

# Create the network and move it to the correct device
net = CustomEfficientNet(width_mult, depth_mult, dropout_rate,class_count=category_count, computation_device=device).to(device)

# Load your trained model (ensure the model file path is correct)
model_path = './efficientnetb0.pth'
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and predict the emotion
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
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
