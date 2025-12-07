import time
import torch
import torchvision
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# --- SETUP ---
# 1. Define the Image Sources
image_urls = [
    # Ultralytics images (these worked fine)
    "https://ultralytics.com/images/zidane.jpg",
    "https://ultralytics.com/images/bus.jpg",
    
    # Wikimedia images (need headers to work)
    "https://upload.wikimedia.org/wikipedia/commons/a/a5/Red_Kitten_01.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Axis_axis_crossing_the_road.JPG/1200px-Axis_axis_crossing_the_road.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Wooden_table_and_chairs_on_a_balcony_over_the_Mekong_at_sunrise_in_Don_Det_Si_Phan_Don_Laos.jpg/1200px-Wooden_table_and_chairs_on_a_balcony_over_the_Mekong_at_sunrise_in_Don_Det_Si_Phan_Don_Laos.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Sukhoi_SuperJet_100_%285114478300%29.jpg/1200px-Sukhoi_SuperJet_100_%285114478300%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/63/Water_reflection_of_mountains_and_hut_in_a_paddy_field_with_blue_sky_in_Vang_Vieng%2C_Laos.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/b5/Lava_lamp_on_windowsill.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f4/Fruit_bowl.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/Street_Traffic_In_Barcelona_%28166082009%29.jpeg"
]

# 2. Initialize Models
# Model A: YOLOv8 (nano version for speed)
yolo_model = YOLO('yolov8n.pt') 

# Model B: Faster R-CNN
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
frcnn_model = fasterrcnn_resnet50_fpn(weights=weights)
frcnn_model.eval()
frcnn_transform = weights.transforms()

# Store results here
results_data = []

# Define headers to mimic a browser (Fixes the Wikimedia 403 error)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# --- PROCESSING ---
print("Starting analysis...")

for i, url in enumerate(image_urls):
    try:
        # Load Image with Headers
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status() # Check for download errors
        img_raw = Image.open(BytesIO(response.content)).convert("RGB")
        img_name = f"Image_{i+1}"
        
        # ----------------------
        # RUN MODEL 1: YOLOv8
        # ----------------------
        start_time = time.time()
        yolo_results = yolo_model(img_raw, verbose=False)
        yolo_time = time.time() - start_time
        
        # Extract YOLO Data
        y_count = len(yolo_results[0].boxes)
        y_conf = yolo_results[0].boxes.conf.mean().item() if y_count > 0 else 0
        
        results_data.append({
            "Image": img_name,
            "Model": "YOLOv8",
            "Time (sec)": round(yolo_time, 4),
            "Objects Detected": y_count,
            "Avg Probability": round(y_conf, 4)
        })

        # ----------------------
        # RUN MODEL 2: Faster R-CNN
        # ----------------------
        # Preprocess
        img_tensor = frcnn_transform(img_raw).unsqueeze(0)
        
        start_time = time.time()
        with torch.no_grad():
            frcnn_results = frcnn_model(img_tensor)
        frcnn_time = time.time() - start_time
        
        # Extract R-CNN Data (Threshold > 0.5)
        pred_scores = frcnn_results[0]['scores']
        high_conf_indices = [idx for idx, score in enumerate(pred_scores) if score > 0.5]
        
        r_count = len(high_conf_indices)
        if r_count > 0:
            r_conf = pred_scores[high_conf_indices].mean().item()
        else:
            r_conf = 0
            
        results_data.append({
            "Image": img_name,
            "Model": "Faster R-CNN",
            "Time (sec)": round(frcnn_time, 4),
            "Objects Detected": r_count,
            "Avg Probability": round(r_conf, 4)
        })
        
        print(f"Processed {img_name}...")

    except Exception as e:
        print(f"Error processing {url}: {e}")

# --- OUTPUT ---
df = pd.DataFrame(results_data)
df = df.sort_values(by=['Image', 'Model'])

print("\nAnalysis Complete! Here is your table:")
print(df)

df.to_csv("model_comparison_results.csv", index=False)