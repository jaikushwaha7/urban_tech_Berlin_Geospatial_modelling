import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

#insert documentation here if needed
"""
This script provides functionalities to analyze changes in satellite imagery,
specifically focusing on urban development and vegetation trends using GIF
time-lapses and NDVI (Normalized Difference Vegetation Index) snapshots.

It includes:
- `get_gif_frames`: Extracts the first and last frames from an animated GIF.
- `generate_rgb_change_map`: Creates a heatmap visualizing pixel-wise RGB
  differences between two images, indicating urban change.
- `generate_ndvi_trend`: Generates a heatmap showing the trend of vegetation
  change (gain or loss) by comparing two NDVI images.

Dependencies:
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`
- `Pillow` (PIL, Image, ImageSequence)

Usage:
1. Ensure you have the required GIF and NDVI image files.
2. Call `generate_rgb_change_map` with the first and last frames of your GIF
   to visualize urban change.
3. Call `generate_ndvi_trend` with two NDVI image paths to visualize vegetation
   gain/loss.
"""


def get_gif_frames(gif_path):
    """Extracts the first and last frames from the Copernicus GIF."""
    with Image.open(gif_path) as im:
        frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(im)]
    return np.array(frames[0]), np.array(frames[-1])

def generate_rgb_change_map(img1, img2, output_name="rgb_change_heatmap.png"):
    """Calculates pixel-wise difference and generates a heatmap."""
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Generate Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(gray_diff, cmap='magma')
    plt.colorbar(label='Change Magnitude')
    plt.title('Urban Change Heatmap (2020 vs 2025)')
    plt.axis('off')
    plt.savefig(output_name)
    print(f"Saved: {output_name}")

def generate_ndvi_trend(ndvi_path1, ndvi_path2, output_name="ndvi_trend_heatmap.png"):
    """Compares two NDVI visualizations to show vegetation gain/loss."""
    # Load images as grayscale (intensity represents NDVI level)
    img1 = cv2.imread(ndvi_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(ndvi_path2, cv2.IMREAD_GRAYSCALE)
    
    # Check if images were loaded successfully
    if img1 is None:
        raise FileNotFoundError(f"Could not read image: {ndvi_path1}")
    if img2 is None:
        raise FileNotFoundError(f"Could not read image: {ndvi_path2}")
    
    # Signed difference: Positive = Gain (Green), Negative = Loss (Red)
    # Note: Using float to allow negative values
    diff = img2.astype(float) - img1.astype(float)
    
    # Create a divergent heatmap (RdYlGn: Red-Yellow-Green)
    plt.figure(figsize=(10, 8))
    # We use a symmetric vmin/vmax to center the 'no-change' at zero
    limit = max(abs(diff.min()), abs(diff.max()))
    plt.imshow(diff, cmap='RdYlGn', vmin=-limit, vmax=limit)
    plt.colorbar(label='Vegetation Index Change (Gain in Green, Loss in Red)')
    plt.title('NDVI Trend Heatmap (2023 vs 2025)')
    plt.axis('off')
    plt.savefig(output_name)
    print(f"Saved: {output_name}")

# --- Execution ---
# 1. Process the Time-lapse GIF
gif_file = 'Sentinel-2_L2A-202005-202508.gif'
first_frame, last_frame = get_gif_frames(gif_file)
generate_rgb_change_map(first_frame, last_frame)

# 2. Process specific NDVI Snapshots (e.g., July 2023 vs August 2025)
generate_ndvi_trend(r'.\NVDI\nvdi_001.png', r'.\NVDI\nvdi_009.png')