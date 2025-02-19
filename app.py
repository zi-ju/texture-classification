from model_training.extract_feature import extract_glcm_features, extract_lbp_features
from model_training.preprocess_dataset import preprocess_image
import joblib
import gradio as gr
import numpy as np


# Load the trained models
glcm_decision_tree_model = joblib.load("models/glcm_decision_tree_model.pkl")
lbp_decision_tree_model = joblib.load("models/lbp_decision_tree_model.pkl")


def classify_texture(image, method="glcm"):
    # Preprocess image
    image = preprocess_image(image)

    # Extract features
    if method == "GLCM":
        features = extract_glcm_features(image)
    else:
        features = extract_lbp_features(image)

    features = np.array(features).reshape(1, -1)

    # Select model
    if method == "GLCM":
        model = glcm_decision_tree_model
    else:
        model = lbp_decision_tree_model

    # Make prediction
    pred = model.predict(features)[0]

    # Map prediction back to label
    label_map = {0: "Stone", 1: "Brick", 2: "Wood"}
    return label_map[pred]


# Gradio UI
demo = gr.Interface(
    fn=classify_texture,
    inputs=[gr.Image(type="pil"), gr.Radio(["GLCM", "LBP"], label="Feature Extraction Method")],
    outputs=gr.Label(label="Predicted Texture"),
    title="Texture Classification",
    description="Upload an image to classify it as stone, brick, or wood using GLCM or LBP."
)

demo.launch()
