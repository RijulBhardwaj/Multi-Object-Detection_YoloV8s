"""
streamlit run Hosted_app.py to run app
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
from io import BytesIO
from PIL import Image  # <-- Add this import for Image class
from tabulate import tabulate  # <-- Import tabulate for better formatting of the model parameters

MODEL_NAME = "yolov8s.pt"  # Using the small model for multi-object detection
model = YOLO(MODEL_NAME)  # Load YOLOv8 model from ultralytics

def generate_random_color():
    """Generate a random color for each bounding box"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def detect_objects(img, conf_threshold=0.5):
    """To localize objects after cnn output going one step further"""
    # Convert BGR (OpenCV default) to RGB (for proper visualization in matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform object detection with YOLO
    results = model(img_rgb, conf=conf_threshold)  # Run the model on the image with adjusted confidence

    # results.xywh is a list, so we access the first item for the bounding boxes
    boxes = results[0].boxes.xywh.numpy()  # [x_center, y_center, width, height]
    labels = results[0].names  # Class labels
    confidences = results[0].boxes.conf.numpy()  # Confidence scores for each detected object
    class_ids = results[0].boxes.cls.numpy().astype(int)  # Class IDs of detected objects

    # Draw bounding boxes and labels on the image
    for i in range(len(boxes)):
        x_center, y_center, width, height = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        
        # Only consider detections above the confidence threshold
        if confidence < conf_threshold:
            continue

        # Calculate top-left and bottom-right coordinates
        x1 = int((x_center - width / 2))  # Top-left x coordinate
        y1 = int((y_center - height / 2))  # Top-left y coordinate
        x2 = int((x_center + width / 2))  # Bottom-right x coordinate
        y2 = int((y_center + height / 2))  # Bottom-right y coordinate

        # Generate a random color for the box
        color = generate_random_color()

        # Draw rectangle on the image with the random color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Create label text with the object name and confidence score
        label = f"{labels[class_id]} {confidence:.2f}"

        # Put label with confidence score near the top-left corner of the box
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img

def display_model_parameters():
    # Prepare table for model layers
    model_layers = []
    
    # Collect the parameters and their shapes
    for name, param in model.model.named_parameters():
        model_layers.append([name, param.shape])
    
    # Collect hyperparameters and input details
    hyperparameters = [
        ["Input Size (stride)", model.model.stride],
        ["Number of Classes", model.model.nc],
    ]
    
    # Prepare table for model layers (structure)
    model_structure = []
    for i, module in enumerate(model.model.modules()):
        module_type = type(module).__name__
        model_structure.append([i, module_type])

    # Display the tables in Streamlit
    st.write("### Model Parameters (Weights & Biases):")
    st.table(model_layers)
    
    st.write("### Model Hyperparameters:")
    st.table(hyperparameters)
    
    st.write("### Model Structure (Layers):")
    st.table(model_structure)

# Streamlit UI
st.title('YOLO Multi-Object Detection')

st.write("Upload an image to detect objects")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Convert the uploaded image to an OpenCV format
    image_bytes = uploaded_file.read()
    image_np = np.array(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Run object detection
    detected_img = detect_objects(img)

    # Convert the image back to RGB for display
    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

    # Display the image with bounding boxes using use_container_width
    st.image(detected_img_rgb, channels="RGB", use_container_width=True)

    # Optionally, save the processed image if needed
    buffer = BytesIO()
    detected_img_rgb_pil = Image.fromarray(detected_img_rgb)  # <-- Convert to PIL Image
    detected_img_rgb_pil.save(buffer, format="PNG")
    st.download_button("Download Processed Image", buffer, file_name="detected_image.png", mime="image/png")

    # Display the model parameters and structure
    display_model_parameters()
