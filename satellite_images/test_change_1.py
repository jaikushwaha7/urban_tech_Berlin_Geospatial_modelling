from PIL import Image, ImageChops

def get_change_map(frame1_path, frame2_path):
    img1 = Image.open(frame1_path).convert('L') # Greyscale for comparison
    img2 = Image.open(frame2_path).convert('L')
    
    # Calculate absolute difference
    diff = ImageChops.difference(img1, img2)
    
    # Apply a threshold to remove sensor noise
    diff = diff.point(lambda p: p if p > 30 else 0)
    diff.save("change_map.png")
    print("Change map saved as change_map.png")

get_change_map("frames/frame_000.png", "frames/frame_002.png")