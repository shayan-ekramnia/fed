# Facial Emotion Detection Streamlit Application

This application detects facial emotions from both uploaded images and real-time camera input. It was built using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset, which includes seven emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The application leverages OpenCV for face detection and uses the trained model to predict the emotions displayed.

## Features

- **Image Upload:** Users can upload an image, and the model will detect and predict the emotions of the faces in the image.
- **Real-Time Detection:** Users can enable their webcam, and the application will predict emotions in real-time.
- **Interactive Visualization:** The application provides interactive visualizations of the results, including confidence levels for each emotion.
  
## How to Use

1. **Upload an Image**: Click on the "Browse files" button to upload an image from your device.
2. **Real-Time Emotion Detection**: Turn on your webcam by clicking the "Start Camera" button to get live emotion predictions.
3. **Visualizations**: The detected emotions will be displayed alongside the confidence scores in an easy-to-read format.

## Technologies Used

- **Streamlit**: For building the user interface and deploying the web application.
- **TensorFlow/Keras**: For building and training the convolutional neural network (CNN).
- **OpenCV**: For real-time face detection from camera input.
- **Plotly**: For interactive data visualizations.

## How to Run the Application Locally

1. Clone this repository:

```bash
git clone https://github.com/your-username/facial-emotion-detection.git
cd facial-emotion-detection
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Visit the URL provided by Streamlit to interact with the application.

## Live Demo

You can access the live demo of this application on [Streamlit Cloud]([https://your-streamlit-app-url](https://shayan-ekramnia-gvitmdvbucdj2qg9j8gdxw.streamlit.app)).



