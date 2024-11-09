# Helmet_numberplate_detection

This project utilizes YOLO v10 to detect if a motorcycle rider is wearing a helmet and to recognize license plate numbers. With the help of powerful YOLO v10 algorithms, this project aims to enhance road safety by detecting violations and recording number plate information in real-time.

##dataset
https://universe.roboflow.com/huynhs-space/helmet-detection-and-license-plate-recognition

## Features

- Helmet Detection: Identifies if the rider is wearing a helmet.
- Number Plate Detection: Recognizes and captures license plate information for further processing.
- Real-Time Detection: Processes video feed from a webcam or video source for immediate detection.
- Image Detection: It can also perform detection on image
- GPU-Accelerated Processing: Runs on Google Colab with Tesla T4 GPU, ensuring fast and efficient detection.


## Project Structure
- Helmet_Number_Plate_Detection.ipynb: Jupyter notebook with main trainning  code only for detecting helmets and number plates using YOLO v10.
- model folder: contain best.pt and last.pt of training which can directly use for detection
- All the graphs and result are also uploaded
- result.csv: It is the trainning results of epochs in csv format

## Tech Stack
**IDE**
- Google Colab: Used as the development environment, with GPU acceleration for efficient model training and inference.
- GPU (Tesla T4): Enabled on Google Colab to handle real-time processing with high performance.
  
**Backend**
- Python: Main programming language for building and implementing the model.
- OpenCV: For handling video input and frame processing.
- YOLO v10: Model for detecting helmets and license plates.
- NumPy: For array manipulation and numerical operations.

## Setup
**Prerequisites**
- Python 3.6 or higher
- Google Collab or any Python based IDE
- Google Colab or Jupyter Notebook


**YOLO v10 - Key Algorithms and Architecture**
- Cross Stage Partial Networks (CSPNet): CSPNet helps YOLO v10 achieve efficient layer aggregation and enhanced learning capacity. This reduces computational cost while maintaining high accuracy, which is essential for recognizing small details like helmets and license plates.
- Anchor-Free Detection: YOLO v10 reduces reliance on predefined anchor boxes, simplifying the model structure and improving detection precision for small or irregularly shaped objects. This is particularly useful for the helmet and license plate detection tasks, where objects can vary greatly in size and shape.
- Multi-Scale Training: This technique allows the model to generalize better across various resolutions. It trains the model to detect objects at different scales, improving robustness in real-world conditions where objects may appear in diverse sizes, distances, and perspectives.
- Non-Maximum Suppression (NMS): YOLO v10 incorporates optimized NMS to filter overlapping bounding boxes, retaining only the most confident detections. This is crucial in scenarios where multiple objects are close together, such as helmets on multiple riders or overlapping license plates.
- Data Augmentation: YOLO v10 employs advanced data augmentation techniques. These augmentations enable the model to perform better in varied environments, improving real-world accuracy.
     - Mosaic Augmentation: Combines four images into one during training, improving the model's ability to detect small objects.
     - Random Cropping and Rotation: Helps the model learn to detect objects from various angles, increasing robustness.
     - Random Cropping and Rotation: Helps the model learn to detect objects from various angles, increasing robustness.
- Dynamic Model Configurations:YOLO v10 offers various configurations (like yolov10n, yolov10s, yolov10m, etc.) to provide a balance between speed and accuracy, allowing for adaptable performance across different hardware and detection tasks. This flexibility ensures the model is optimized for specific detection scenarios, whether high-speed detection or high-accuracy detection is prioritized.

**Fine Tunning and parameters**
1) Learning Rate Schedule:
    - Initial Learning Rate (lr0=0.01): The starting learning rate is set relatively high to accelerate the initial learning phase.
    - Final Learning Rate (lrf=0.0001): The learning rate is gradually decreased to this final value, ensuring stability in the final stages of training and helping the model settle into an optimal minimum.

   
**Installation**
Clone the yolo v10 repository and set up the environment:
```bash
  git clone https://github.com/THU-MIG/yolov10.git

```
Install the required  python dependencies for Volume_Brighness.ipynb:  

```bash
  pip install cv
  pip install numpy
  
```
Move into the directory and install dependencies:

```bash
  cd yolov10
  pip install -r requirements.txt
```
Create weight directory and download model Weights

```bash
  import os
import urllib.request

# Directory for the weights
weights_dir = os.path.join(os.getcwd(), "weights")
os.makedirs(weights_dir, exist_ok=True)

# URLs of weight files
urls = [
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt",
    "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt",
]

# Download each file
for url in urls:
    file_name = os.path.join(weights_dir, os.path.basename(url))
    urllib.request.urlretrieve(url, file_name)
    print(f"Downloaded {file_name}")

```

## Authors

- [@SHOVIK CHAKRABORTY](https://github.com/cshovik)


## License

[MIT](https://github.com/cshovik/Gesture-Based-Computer-Control?tab=MIT-1-ov-file#readme)

## Screenshot:
![screenshot_619](https://github.com/user-attachments/assets/8d0b15f4-3b46-437e-a705-56d1ba11cde9)

