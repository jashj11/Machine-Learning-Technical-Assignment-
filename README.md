# Stereo Vision Depth Estimation & YOLO Object Distance Tracker

This project implements a real-time computer vision pipeline that combines **OpenCV Stereo Semi-Global Block Matching (SGBM)** with **YOLO object detection** to calculate metric distances (in meters) of objects in a driving scene. It leverages real camera calibration parameters from the **KITTI Vision Benchmark Suite**.

## Features

* **Stereo Disparity & Depth Mapping**: Computes high-density disparity maps and metric depth maps from stereo image pairs (`kitti_1.png` and `kitti_2.png`).

* **Real-World Metric Scaling**: Utilizes precise calibration matrices ($P_2$ and $P_3$) from KITTI to automatically extract focal length and camera baseline.

* **Pre-trained YOLO Object Detection**: Integrates a pre-trained **YOLO11n** model fine-tuned on the **BDD100K** (Berkeley DeepDrive) dataset (`dronefreak/bdd100k-yolo11n`). 
  > *Note on Training:* Custom training from scratch was bypassed because each epoch required approximately 1 hour of compute time, making pre-trained weights the ideal choice for efficient inference.

* **Robust Distance Estimation**: Samples pixel regions inside object bounding boxes and uses robust statistics ($\text{median}$) to filter noise and compute physical distances.

## Project Structure

* `depth_disparity_relation.py`: Standalone script for computing, saving, and visualizing stereo disparity and colormapped depth maps.

* `depth_est+object_detect.py`: Full integrated pipeline performing YOLO object detection, bounding box extraction, depth querying, and visual overlay of distance annotations.

* `kitti_1.png` / `kitti_2.png`: Left and right stereo image inputs.

## Mathematical Background

The depth $Z$ is calculated using the standard stereo triangulation formula:

$$
Z = \frac{f_x \cdot B}{d}
$$

Where:

* $f_x$ is the focal length extracted from projection matrix $P_2[0, 0]$.

* $B$ is the stereo baseline computed from the projection matrix translation components:
  

* $d$ is the disparity in pixels derived from the raw SGBM output:
  

## Requirements

Make sure you have the following Python libraries installed:

```
pip install opencv-python numpy ultralytics huggingface_hub
```

## How to Run

1. Place your stereo image pair (`kitti_1.png` and `kitti_2.png`) in the project root directory.

2. Run the depth and object detection pipeline:

```
python depth_est+object_detect.py<img width="1561" height="714" alt="results_" src="https://github.com/user-attachments/assets/7996c129-736b-4d64-b32d-a30c2061166b" />

```

3. Press any key while the preview windows are active to close the application.

## Results
<img width="1561" height="714" alt="results_" src="https://github.com/user-attachments/assets/c291e365-ed09-438b-b1a4-1598d29696e0" />



## Contact
Name : Jash Jain, 
Phone No. : 8097275568, 
email : jjain0371@gmail.com
