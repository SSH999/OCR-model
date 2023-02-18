from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Load the OCR model
try:
    model = load_model("OCR_model.h5")
except Exception as e:
    print(f"Error loading OCR model: {e}")
    model = None

# Define a function to preprocess the image for inference
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.array(img).reshape((28, 28, 1))
    img = img / 255.0
    return img

# Define a function to extract the digit from the model's prediction
def extract_digit(prediction):
    digit = np.argmax(prediction)
    confidence = prediction[digit]
    return digit, confidence

# Define a function to handle errors
def handle_error(error_message):
    return render_template('index.html', error=error_message)

# Define the route for the main page
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.files:
            return handle_error('No image uploaded')
        file = request.files['image']
        # Check if the file is an image
        if not file.filename.endswith(('png', 'jpg', 'jpeg', 'bmp')):
            return handle_error('File is not an image')
        # Read the image and preprocess it for inference
        try:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            preprocessed_image = preprocess_image(image)
        except Exception as e:
            return handle_error(f"Error processing image: {e}")
        # Run the model on the image and extract the predicted digit
        if model is None:
            return handle_error("OCR model could not be loaded")
        try:
            prediction = model.predict(np.array([preprocessed_image]))
            digit, confidence = extract_digit(prediction[0])
        except Exception as e:
            return handle_error(f"Error predicting digit: {e}")
        # Display the predicted digit
        return render_template('index.html', digit=digit, confidence=confidence)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()

