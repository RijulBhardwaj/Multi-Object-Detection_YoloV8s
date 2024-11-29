import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
from tabulate import tabulate


# Load the YOLO model (using the YOLOv8s model for balanced performance)

MODEL_NAME = "yolov8s.pt"  # s means the small model

model = YOLO(MODEL_NAME)  # using ultralytics library. Must use git clone to import the model

def generate_random_color():
    """
    This generates a random color for each new obheect
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def detect_objects(image_path, conf_threshold=0.5):
    
    img = cv2.imread(image_path)

    # Convert BGR (OpenCV default) to RGB (for proper visualization in matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform object detection with YOLO
    results = model(img_rgb, conf=conf_threshold)  # Run the model on the image with adjusted confidence less confidence means wil ldetect more images 

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

    # Convert the BGR image back to RGB for visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide the axes
    plt.show()


# Set the path to your input image
image_path = "C:/Users/Rijul Bhardwaj/OneDrive/Pictures/Study/woman.jpg"


detect_objects(image_path, conf_threshold=0.3)  # Lower the confidence threshold to detect more objects


def display_model_parameters():
    # Prepare table for model layers
    model_layers = []
    
    # Collect the parameters and their shapes
    for name, param in model.model.named_parameters():
        model_layers.append([name, param.shape])
    
    # Print the model parameters (weights & biases) in a table
    print("Model Parameters (Weights & Biases):")
    print(tabulate(model_layers, headers=["Parameter Name", "Shape"], tablefmt="grid"))
    
    # Collect hyperparameters and input details
    hyperparameters = [
        ["Input Size (stride)", model.model.stride],
        ["Number of Classes", model.model.nc],
        # Avoiding out_channels as it doesn't exist directly, will try to retrieve other useful details
    ]
    
    # Print the hyperparameters in a table format
    print("\nModel Hyperparameters:")
    print(tabulate(hyperparameters, headers=["Hyperparameter", "Value"], tablefmt="grid"))
    
    # Prepare table for model layers (structure)
    model_structure = []
    for i, module in enumerate(model.model.modules()):
        module_type = type(module).__name__
        model_structure.append([i, module_type])
    
    # Print the model's layers and structure in a table
    print("\nModel Structure (Layers):")
    print(tabulate(model_structure, headers=["Layer Index", "Layer Type"], tablefmt="grid"))

# Call the function to display model parameters in table format
display_model_parameters()

