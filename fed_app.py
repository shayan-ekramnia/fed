

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os


# Load the pre-trained emotion detection model
model = load_model('final_emotion_model44.h5')

# Define emotion classes
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to detect face and predict emotion
def detect_face_and_emotion(image):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    # Convert the image to grayscale (for face detection)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)

        # Prepare the face for emotion prediction
        face = face.astype('float') / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # Predict the emotion
        preds = model.predict(face)[0]
        emotion = emotion_classes[np.argmax(preds)]

        # Draw rectangle around the face and add label on the NumPy image
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image_np, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return image_np



# Streamlit app interface
st.title("Facial Emotion Detection Application")

# Option to upload an image or use the webcam
option = st.selectbox("Choose input method", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Perform face detection and emotion prediction
        result_img = detect_face_and_emotion(img)
        st.image(result_img, caption="Processed Image", use_column_width=True)

elif option == "Use Webcam":
    # Real-time camera feed
    run = st.checkbox('Run')

    # Start the webcam
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform face detection and emotion prediction on live feed
        result_frame = detect_face_and_emotion(frame)
        
        # Display the processed frame
        FRAME_WINDOW.image(result_frame)

    camera.release()

