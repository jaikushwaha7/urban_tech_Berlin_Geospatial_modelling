import os
import sys
import subprocess
# insert documentation here if needed   

# Add Tesseract to PATH
tesseract_path = r'C:\Program Files\Tesseract-OCR'
if tesseract_path not in os.environ['PATH']:
    os.environ['PATH'] = tesseract_path + os.pathsep + os.environ['PATH']

from PIL import Image, ImageOps
import pytesseract

# Also set the direct path as backup
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_gif_data(gif_path, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    with Image.open(gif_path) as im:
        metadata = []
        for i in range(im.n_frames):
            im.seek(i)
            frame = im.convert('RGB')
            frame_path = f"{output_dir}/frame_{i:03d}.png"
            frame.save(frame_path)
            
            # OCR Preprocessing: Crop to the date area (adjust coordinates as needed)
            # Example: top-right corner [left, top, right, bottom]
            width, height = frame.size
            date_region = frame.crop((width - 300, 0, width, 50)) 
            date_text = pytesseract.image_to_string(date_region).strip()
            
            metadata.append({"frame": i, "path": frame_path, "date": date_text})
            print(f"Processed Frame {i}: {date_text}")
            
    return metadata

metadata = extract_gif_data("Sentinel-2_L2A-202005-202508.gif")