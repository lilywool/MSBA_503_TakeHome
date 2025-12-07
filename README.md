# MSBA_503_TakeHome
# Object Detection Model Comparison: YOLOv8 vs. Faster R-CNN

**Author:** Lillian Wool

**Course:** MSBA 503

**Date:** December 2025

**Language:** Python 3.13

## Project Overview
This project performs a comparative analysis of two state-of-the-art object detection architectures to evaluate the trade-offs between inference speed and detection accuracy.

The analysis runs automated inference on a diverse dataset of images (including cluttered desks, city streets, and sports scenes) using:
1.  **YOLOv8n (You Only Look Once):** A single-stage detector optimized for real-time applications.
2.  **Faster R-CNN (ResNet50):** A two-stage detector known for high accuracy and robustness.

## Repository Structure
* **Take_Home_Wool.py:** The main Python script that automates image downloading, model inference, and data logging.
* **model_comparison_results.csv:** The output dataset containing inference times, object counts, and confidence scores for every image.
* **README.md:** Project documentation.

## Methodology
The script performs the following steps for each image in the dataset:
1.  **Image Acquisition:** Downloads high-resolution images programmatically from Wikimedia Commons and other sources, handling User-Agent headers to prevent access errors.
2.  **Model Inference:**
    * **YOLOv8n:** Implemented via the ultralytics library.
    * **Faster R-CNN:** Implemented via torchvision with pre-trained ResNet50 weights.
3.  **Data Logging:** Captures Inference Time (seconds), Number of Objects Detected, and Average Confidence Score.
4.  **Export:** Aggregates all metrics into a structured CSV file for reporting.

## Installation & Setup
To run this project locally, ensure you have Python installed. You will need the following dependencies:

```bash
pip install torch torchvision ultralytics pandas requests pillow
