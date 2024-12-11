import torch
import cv2
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Define the class labels (same as in your training code)
class_labels = {0: "Cardboard", 1: "Glass", 2: "Metal", 3: "Paper", 4: "Plastic", 5: "Others"}

# Define the ResNet class for inference
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(class_labels))  # Replace last layer
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# Transformation to preprocess the input image (same as training)
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to make predictions
def predict_image(image):
    image = transformations(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)  # Run the model
        _, pred = torch.max(output, 1)  # Get the predicted class
    return pred.item()

model = ResNet()
model = torch.load("ModelAll.pt", weights_only=False)
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Loop to process each frame from the webcam
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame (BGR) to PIL image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Get prediction
    pred_class = predict_image(pil_image)

    # Get predicted label from the class_labels dictionary
    predicted_label = class_labels[pred_class]

    # Draw the label on the frame
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()