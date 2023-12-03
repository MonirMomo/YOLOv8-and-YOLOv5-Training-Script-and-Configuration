# Importing the necessary libraries
import os

# Getting the current working directory and storing it in the HOME variable
HOME = os.getcwd()

# Printing the current working directory
print(HOME)

# Installing the ultralytics library with a specific version
!pip install ultralytics==8.0.20

# Clearing the output display in the IPython environment
from IPython import display
display.clear_output()

# Importing the ultralytics library and performing checks
import ultralytics
ultralytics.checks()

# Importing the YOLO class from the ultralytics library
from ultralytics import YOLO

# Importing the display and Image classes from IPython
from IPython.display import display, Image

# Creating a 'datasets' directory in the current working directory
!mkdir {HOME}/datasets

# Changing the working directory to the 'datasets' directory
%cd {HOME}/datasets

# Installing the roboflow library without displaying output
!pip install roboflow --quiet

# Importing the Roboflow class from the roboflow library and setting up the API key
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")

# Accessing a specific workspace, project, and version from the Roboflow platform
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
dataset = project.version(1).download("yolov8")

# Changing the working directory back to the original HOME directory
%cd {HOME}

# Running the YOLO training task with specific parameters
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True

# Changing the working directory to the HOME directory
%cd {HOME}

# Running YOLO in validation mode using a specific model and data configuration
!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

# Changing the working directory back to the HOME directory
%cd {HOME}

# Running YOLO in prediction mode using a specific model, confidence threshold, source directory, and saving the results
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True