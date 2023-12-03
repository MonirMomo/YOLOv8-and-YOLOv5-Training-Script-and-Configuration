# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone YOLOv5 repository
%cd yolov5

# Install YOLOv5 dependencies
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

# Import necessary libraries
import torch
import os
from IPython.display import Image, clear_output  # to display images

# Print setup information, including Torch version and device information
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Initialize Roboflow client
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")

# Set up environment variables
os.environ["DATASET_DIRECTORY"] = "/content/datasets"

# Importing the Roboflow class from the roboflow library and setting up the API key
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3c1rVplkVlaNwJhcLbGT")
project = rf.workspace("orkhan-aliyev-8nktf").project("fruits-and-vegetables-2vf7u")
dataset = project.version(1).download("yolov5-obb")


# Train YOLOv5 model
!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache

# Start TensorBoard to monitor training progress
# Launch after you have started training, logs save in the folder "runs"
%load_ext tensorboard
%tensorboard --logdir runs

# Run YOLOv5 object detection on test images
!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
