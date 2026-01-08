import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

# insert documentation here if needed

def extract_metadata(image_path):
    """Extracts basic and EXIF metadata from an image."""
    meta = {"Filename": os.path.basename(image_path)}
    try:
        with Image.open(image_path) as img:
            meta["Size"] = img.size
            meta["Format"] = img.format
            # Attempt to extract EXIF (dates, etc.)
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    decoded = TAGS.get(tag, tag)
                    meta[decoded] = value
    except Exception as e:
        meta["Error"] = str(e)
    return meta

def analyze_ndvi_file(image_path):
    """Calculates a health score based on NDVI visualization intensity."""
    # Read as grayscale; in NDVI exports, brighter usually means higher index
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Calculate basic stats
    avg_intensity = np.mean(img)
    max_intensity = np.max(img)
    # Estimate 'healthy' coverage (pixels above a specific brightness threshold)
    healthy_pixels = np.sum(img > 150) / img.size * 100 
    
    return {
        "Avg_NDVI_Intensity": round(avg_intensity, 2),
        "Max_NDVI_Intensity": max_intensity,
        "Green_Coverage_Pct": round(healthy_pixels, 2)
    }

# --- Execution Block ---
folder_path = 'NVDI'  # Update to your folder name
all_results = []

print(f"Scanning folder: {folder_path}...")

for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        file_path = os.path.join(folder_path, filename)
        
        # 1. Get Metadata
        metadata = extract_metadata(file_path)
        
        # 2. Get Analysis Features
        analysis = analyze_ndvi_file(file_path)
        
        # Combine and store
        if analysis:
            combined = {**metadata, **analysis}
            all_results.append(combined)

# 3. Save to Summary Feature File (CSV)
df = pd.DataFrame(all_results)
df.to_csv("ndvi_batch_summary.csv", index=False)
print("Analysis complete. Summary saved to 'ndvi_batch_summary.csv'.")