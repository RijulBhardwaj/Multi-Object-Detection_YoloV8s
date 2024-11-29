import torch
from ultralytics import YOLO
from tabulate import tabulate

# Load the YOLO model (using a pre-trained model from ultralytics)
MODEL_NAME = "yolov8n.pt"  # You can use other models like yolov8m.pt for better accuracy
model = YOLO(MODEL_NAME)  # Load YOLOv8 model from ultralytics

# Function to display model parameters and information in a nice table
def display_model_parameters():
    # Prepare table for model parameters (weights and biases)
    model_layers = []
    
    # Collect the parameters and their shapes
    for name, param in model.model.named_parameters():
        model_layers.append([name, param.shape])
    
    # Print the model parameters (weights & biases) in a table format
    print("Model Parameters (Weights & Biases):")
    print(tabulate(model_layers, headers=["Parameter Name", "Shape"], tablefmt="grid"))
    
    # Collect hyperparameters and input details
    hyperparameters = [
        ["Input Size (stride)", model.model.stride],
        ["Number of Classes", model.model.nc],
        ["Number of Parameters", sum(p.numel() for p in model.model.parameters())],
        # Add more hyperparameters if needed
    ]
    
    # Print the hyperparameters in a table format
    print("\nModel Hyperparameters:")
    print(tabulate(hyperparameters, headers=["Hyperparameter", "Value"], tablefmt="grid"))
    
    # Prepare table for model layers (structure)
    model_structure = []
    for i, module in enumerate(model.model.modules()):
        module_type = type(module).__name__
        model_structure.append([i, module_type])
    
    # Print the model's layers and structure in a table format
    print("\nModel Structure (Layers):")
    print(tabulate(model_structure, headers=["Layer Index", "Layer Type"], tablefmt="grid"))

# Call the function to display model parameters in table format
display_model_parameters()
