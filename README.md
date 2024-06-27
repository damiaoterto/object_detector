# Object Detection

This repository contains a Python script for object detection in images using the YOLOv4 neural network and the OpenCV library. The script loads pre-trained weights and configuration, performs object detection on a given image, and displays the resulting image with the detections drawn.

## Requirements

- Python 3.6+
- pipenv for virtual environment management
- OpenCV
- NumPy
- Matplotlib

## Environment Setup

To install the dependencies and set up the virtual environment, follow the steps below:

1. Clone this repository:

    ```sh
    git clone https://github.com/your-username/yolov4-object-detection.git
    cd yolov4-object-detection
    ```

2. Download yolov4 weights on project path:
    ```sh
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    ```

3. Use `pipenv` to create and activate the virtual environment:

    ```sh
    pipenv install
    pipenv shell
    ```

4. Install the dependencies listed in the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

## How to Use

1. Place your input images in the `data` folder.

2. Ensure you have the following files in the `cfg` folder:
    - `coco.names`: File containing the class names.
    - `yolov4.cfg`: YOLOv4 network configuration file.
    - `yolov4.weights`: Pre-trained YOLOv4 weights file.

3. Run the main script `main.py`:

    ```sh
    python main.py
    ```

## Project Structure

```plaintext
├── cfg
│   ├── coco.names
│   ├── yolov4.cfg
│   └── yolov4.weights
├── data
│   ├── dog.jpg
│   ├── horses.jpg
│   └── humans.jpg
├── main.py
├── Pipfile
├── Pipfile.lock
├── requirements.txt
└── README.md
```

## Features

- Loading YOLOv4 network weights and configuration.
- Reading input image.
- Object detection using the YOLOv4 network.
- Displaying the resulting image with bounding boxes and labels of detections.

## Contribution
Contributions are welcome! Feel free to open issues or pull requests for improvements and fixes.

## Licence
This project is licensed under the MIT License. See the LICENSE file for more information.