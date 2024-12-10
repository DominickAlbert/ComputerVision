#btw kalo mau quit camera pencet Q aja, kalo x di pojok kanan atas gak bakal keluar nanti malah error

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.cluster.vq import vq
import pickle  # To save and load the BoW model (KMeans)
import time  # For managing the update interval


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
cap = cv2.VideoCapture(1)  # Use 0 for the default camera
# Variables for controlling label update frequency
last_update_time = time.time()
update_interval = 1.0  # Update label every 1 second
current_label = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute the BoW histogram from the frame
    histogram = compute_histogram_from_frame(frame, kmeans)
    histogram = histogram.reshape(1, -1)  # Reshape for the model

    # Update the label only if enough time has passed
    if time.time() - last_update_time > update_interval:
        predictions = model.predict(histogram)
        predicted_label = np.argmax(predictions)  # Get the class index
        current_label = label_mapping[predicted_label]
        last_update_time = time.time()

    # Display the prediction on the video feed
    cv2.putText(frame, f"Prediction: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Garbage Classification', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

