import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN model for emotion recognition (7 classes)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):  # 7 classes (angry, disgust, fear, happy, neutral, sad, surprise)
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 7 output classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load("fine_tuned_emotion_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define class labels (assuming these are the 7 classes in the order of the dataset)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((48, 48)),              # Resize to 48x48
    transforms.ToTensor(),                    # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
])

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply the transformations to the frame
    image = Image.fromarray(rgb_frame)  # Convert numpy array to PIL Image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Predict emotion
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Get the predicted emotion
    predicted_emotion = class_labels[predicted.item()]
    
    # Display the predicted emotion on the frame
    cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
