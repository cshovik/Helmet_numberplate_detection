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

## Model Performance Comparison

| Metric       | Previous Performance | Current Performance | Improvement |
|--------------|----------------------|---------------------|-------------|
| mAP          | 83.2%                | 95.6%              | +12.4%      |
| Precision    | 79.6%                | 95.6%              | +16.0%      |
| Recall       | 78.8%                | 99.0%              | +20.2%      |


## Project Structure
- Helmet_Number_Plate_Detection.ipynb: Jupyter notebook with main trainning  code only for detecting helmets and number plates using YOLO v10.
- model folder: contain best.pt and last.pt of training which can directly use for detection
- All the graphs and result are also uploaded
- result.csv: It is the trainning results of epochs in csv format

## YOLO v10 - Key Algorithms and Architecture
- Cross Stage Partial Networks (CSPNet): CSPNet helps YOLO v10 achieve efficient layer aggregation and enhanced learning capacity. This reduces computational cost while maintaining high accuracy, which is essential for recognizing small details like helmets and license plates.
- Anchor-Free Detection: YOLO v10 reduces reliance on predefined anchor boxes, simplifying the model structure and improving detection precision for small or irregularly shaped objects. This is particularly useful for the helmet and license plate detection tasks, where objects can vary greatly in size and shape.
- Multi-Scale Training: This technique allows the model to generalize better across various resolutions. It trains the model to detect objects at different scales, improving robustness in real-world conditions where objects may appear in diverse sizes, distances, and perspectives.
- Non-Maximum Suppression (NMS): YOLO v10 incorporates optimized NMS to filter overlapping bounding boxes, retaining only the most confident detections. This is crucial in scenarios where multiple objects are close together, such as helmets on multiple riders or overlapping license plates.
- Data Augmentation: YOLO v10 employs advanced data augmentation techniques. These augmentations enable the model to perform better in varied environments, improving real-world accuracy.
     - Mosaic Augmentation: Combines four images into one during training, improving the model's ability to detect small objects.
     - Random Cropping and Rotation: Helps the model learn to detect objects from various angles, increasing robustness.
     - Random Cropping and Rotation: Helps the model learn to detect objects from various angles, increasing robustness.
- Dynamic Model Configurations:YOLO v10 offers various configurations (like yolov10n, yolov10s, yolov10m, etc.) to provide a balance between speed and accuracy, allowing for adaptable performance across different hardware and detection tasks. This flexibility ensures the model is optimized for specific detection scenarios, whether high-speed detection or high-accuracy detection is prioritized.

## Fine Tunning and parameters
1) Learning Rate Schedule:
    - Initial Learning Rate (lr0=0.01): The starting learning rate is set relatively high to accelerate the initial learning phase.
    - Final Learning Rate (lrf=0.0001): The learning rate is gradually decreased to this final value, ensuring stability in the final stages of training and helping the model settle into an optimal minimum.
2) Batch Size (batch=8):
    - A batch size of 8 is used, balancing memory usage and stability during training. Small batch sizes like this are often suitable for more stable convergence on limited hardware.
3) Flip Augmentation:
    - Vertical Flip (flipud=0.5) and Horizontal Flip (fliplr=0.5): These augmentations are applied with a probability of 50%, helping the model generalize better to different object orientations.
4) Color Augmentation:
    - Hue Shift (hsv_h=0.015): Allows hue variation of up to ±1.5%, which helps the model to adapt to different lighting conditions.
    - Saturation Shift (hsv_s=0.7): Allows saturation variation of up to 70%, further enhancing the model’s ability to handle diverse lighting and color conditions.
    - Brightness Shift (hsv_v=0.4): Allows brightness variation of up to 40%, making the model more robust to shadow or illumination changes.
5) Mixup Augmentation (mixup=0.1):
    - Mixup is a data augmentation technique where two images are blended together to create synthetic training data. This helps the model become more robust by learning from blended examples, which is especially helpful for detecting 
      overlapping or partially obscured objects.
7) Auto Augment (auto_augment=randaugment):
    - Random augmentation strategies are applied, further improving the model's ability to generalize. randaugment applies random transformations to images, making the model more adaptable to various environments and conditions.
8) Cosine Learning Rate Scheduler (cos_lr=True):
    - A cosine learning rate scheduler is used to gradually decrease the learning rate in a cosine pattern, which helps prevent the model from getting stuck in suboptimal solutions and improves convergence.
   
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

- [@SHOVIK CHAKRABORTY (RA2211003011403)](https://github.com/cshovik)
- [@Moaz Alimi (RA2211003011348)]
- [@Kush Kokate (RA2211003011382)]


## License

[MIT](https://github.com/cshovik/Gesture-Based-Computer-Control?tab=MIT-1-ov-file#readme)

## Detection image:
![WhatsApp Image 2024-09-19 at 22 41 48_bcdb3ce9](https://github.com/user-attachments/assets/953b20b3-a513-4aab-9760-c73e493adbfb)

![WhatsApp Image 2024-09-19 at 22 38 17_dd9a0a80](https://github.com/user-attachments/assets/ea2b24c9-c7e1-48f1-8853-8385c6115cbf)

![WhatsApp Image 2024-09-19 at 22 41 57_efb116d6](https://github.com/user-attachments/assets/eb0590b4-8d6c-490e-aad3-99071a777a8b)

![img2](https://github.com/user-attachments/assets/97c2685f-8978-4f79-94fb-95087d6fdf02)

![img5](https://github.com/user-attachments/assets/433aee6a-2af7-481d-87dd-612fa5ed8eb4)

![img6](https://github.com/user-attachments/assets/963ad7bb-e56c-4eea-8fe4-cbd73574f884)



