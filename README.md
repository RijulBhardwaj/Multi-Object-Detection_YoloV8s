

---

# Multi-Object Detection with YOLOv8

## Overview

This repository contains code for a **Multi-Object Detection** application built using the **YOLOv8** model for real-time object detection. The application is hosted on **Streamlit**, providing an interactive and easy-to-use interface for detecting multiple objects in images and videos.

The model is trained to detect various objects like vehicles, pedestrians, and animals, and can handle various input formats, including single images and video streams. This project leverages the **Ultralytics YOLOv8** implementation, which provides state-of-the-art performance for object detection tasks.

---

## Features

- **Real-time Object Detection**: Detects multiple objects in images and video streams.
- **Streamlit Web App**: Easy-to-use interface for uploading images/videos and visualizing detection results.
- **YOLOv8 Model**: Utilizes the YOLOv8 architecture for fast and accurate object detection.
- **Customizable Detection**: Can be extended to detect various objects by using different YOLO models or datasets.
- **Model Parameter and Hyperparameter Display**: Get detailed insights into the model structure and parameters.

---

## Demo

You can try out the demo application hosted on **Streamlit**. Access it [here](your-streamlit-link).

---

## Installation

To run the project locally, follow these steps:

### Prerequisites

1. Python 3.8 or higher
2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

- **`torch`**: PyTorch for model training and inference.
- **`ultralytics`**: YOLOv8 model implementation by Ultralytics.
- **`streamlit`**: For building the web interface.
- **`tabulate`**: For displaying model parameters and structure in tabular format.

### Running the App Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/multi-object-detection.git
cd multi-object-detection
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

This will start a local server. Open your browser and visit `http://localhost:8501` to interact with the app.

---

## Usage

Once the Streamlit app is running, you can upload an image or video file for object detection. The app will display the results, including:

- **Bounding boxes** around detected objects.
- **Labels** for each detected object.
- **Confidence scores** for each detection.

You can also check out the model parameters and structure by clicking on the **Model Info** button in the interface.

---

## Model Information

The project uses **YOLOv8**, a state-of-the-art object detection model. The key features of YOLOv8 include:

- **Accuracy**: Provides highly accurate results with real-time performance.
- **Efficiency**: YOLOv8 is optimized for fast inference on both GPUs and CPUs.
- **Flexibility**: YOLOv8 can be fine-tuned for specific datasets, enabling customization for different detection tasks.

### Hyperparameters:
- Input size (stride)
- Number of detected classes
- Model architecture details (layers and modules)

---

## Contributing

Feel free to fork the repository and contribute improvements or fixes. If you encounter any issues, please open an issue in the GitHub repository. Contributions are welcome, and please adhere to the following guidelines:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Ultralytics**: For providing the YOLOv8 implementation, which powers the object detection model.
- **Streamlit**: For making it easy to build and deploy web apps.

---

