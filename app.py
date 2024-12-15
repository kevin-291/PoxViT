import streamlit as st
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline, AutoImageProcessor, ViTForImageClassification

save_dir = r'C:\Users\kevin\Desktop\VSCode\Final_Sem_Project\Config_Files\best_model'


def getPrediction(image):
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=True)
    vit = ViTForImageClassification.from_pretrained(save_dir)
    model = pipeline('image-classification', model=vit, feature_extractor=processor, device='cuda')

    result = model(image)
    return result

st.title("Disease Prediction")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the image on the Streamlit app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Convert the image to a format suitable for your prediction function
    # Convert the PIL image to a NumPy array (OpenCV format)
    image_np = np.array(image)

    # Convert the image from RGB (PIL format) to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert the OpenCV image (BGR format) to RGB format
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PIL Image for the prediction
    pil_image = Image.fromarray(image_rgb)

    # Get prediction
    prediction = getPrediction(pil_image)

    print(prediction)

    # Extract the label with the highest score
    highest_prediction = max(prediction, key=lambda x: x['score'])

    # Display the label and score as a percentage
    score_percentage = highest_prediction['score'] * 100
    st.write(f"**Prediction:** {highest_prediction['label']}, **Confidence:** {score_percentage:.2f}%")