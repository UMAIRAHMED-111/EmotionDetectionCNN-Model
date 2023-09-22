# Import necessary libraries
import cv2
import numpy as np
from keras.models import model_from_json
import os
import time

# Load model architecture from JSON file
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the pre-trained model from JSON
loaded_model = model_from_json(loaded_model_json)

# Load model weights from H5 file
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Define emotion labels for mapping predictions
emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to predict emotion from facial image
def predict_emotion(face_image):
    """
    Predicts emotion from the facial image.

    Parameters:
        face_image (numpy.ndarray): Grayscale facial image of shape (48, 48).

    Returns:
        str: Predicted emotion label.
    """
    # Resize the face image to match the model input size
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image.astype("float") / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    
    # Make prediction using the loaded model
    prediction = loaded_model.predict(face_image)
    max_index = np.argmax(prediction)
    emotion = emotion_labels[max_index]
    return emotion

# Initialize variables for emotion detection intervals
interval = 10  # Time interval between emotion detections (in seconds)
duration = 45  # Total duration for capturing emotions (in seconds)
emotion_counts = {label: 0 for label in emotion_labels}

# Create a new folder for this stream
folder_name = f"stream_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(folder_name)

# Capture video from default camera
cap = cv2.VideoCapture(0)

# Variables to keep track of the last captured frame and interval time
last_captured_frame = 0
start_time = time.time()

# Lists to store the interval emotions and corresponding frames
interval_emotions = []
interval_frames = []

# Loop over frames and perform emotion detection
while (time.time() - start_time) <= duration:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Crop the face region for emotion prediction
        face_roi = gray[y:y + h, x:x + w]

        # Predict emotion from the cropped face region
        emotion = predict_emotion(face_roi)

        # Update emotion counts
        emotion_counts[emotion] += 1

        # Display the predicted emotion on the frame
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Check if it's time to save the frame for interval report
        if (time.time() - start_time) >= last_captured_frame + interval:
            interval_emotions.append(emotion)
            interval_frames.append(frame.copy())  # Save the frame corresponding to the interval
            last_captured_frame = time.time() - start_time

    # Display the frame with emotions and rectangles around faces
    cv2.imshow("Emotion Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Record the last emotion at the end of the video
end_emotion = predict_emotion(gray)

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Write the interval emotions to the report file
with open(os.path.join(folder_name, 'emotion_report.txt'), 'w') as f:
    f.write(f"Camera was used for {time.time() - start_time:.2f} seconds\n")
    for i, emotion in enumerate(interval_emotions, start=1):
        f.write(f"At {i * interval} seconds, the person's emotion is {emotion}\n")
    f.write(f"At the end, the emotion is {end_emotion} ({emotion_counts[end_emotion]})\n\n")
    f.write("Emotion Counts:\n")
    for emotion, count in emotion_counts.items():
        f.write(f"{emotion}: {count}\n")

# Save the interval frames in the corresponding folder
for i, frame in enumerate(interval_frames, start=1):
    frame_path = os.path.join(folder_name, f"frame_{i}.jpg")
    cv2.imwrite(frame_path, frame)

# Print the overall emotion counts
print("Emotion Counts:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")