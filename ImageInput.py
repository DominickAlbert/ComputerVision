#btw kalo mau quit camera pencet Q aja, kalo x di pojok kanan atas gak bakal keluar nanti malah error

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.cluster.vq import vq
import pickle  # To save and load the BoW model (KMeans)

# Load the trained neural network model
model = load_model('garbage_classification_model.h5')

# Load the KMeans model for BoW
with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

# Create the SIFT detector
sift = cv2.SIFT_create()

# Function to compute histogram for camera frames
def compute_histogram_from_frame(frame, kmeans):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None or descriptors.size == 0:
        return np.zeros(len(kmeans.cluster_centers_))  # Empty histogram for no features
    
    # Convert descriptors into BoW histogram
    words, _ = vq(descriptors, kmeans.cluster_centers_)
    histogram = np.zeros(len(kmeans.cluster_centers_))
    for word in words:
        histogram[word] += 1
    return histogram

# Label mapping for display
label_mapping = {0: "Cardboard", 1: "Glass", 2: "Metal", 3: "Paper", 4: "Plastic", 5: "Trash"}

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute the BoW histogram from the frame
    histogram = compute_histogram_from_frame(frame, kmeans)
    histogram = histogram.reshape(1, -1)  # Reshape for the model

    # Predict the class using the trained model
    predictions = model.predict(histogram)
    predicted_label = np.argmax(predictions)  # Get the class index
    label = label_mapping[predicted_label]

    # Display the prediction on the video feed
    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Garbage Classification', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
