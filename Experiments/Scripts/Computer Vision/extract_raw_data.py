import os
import rawpy
import numpy as np
import cv2
import scipy.io as sio

RAW_PATH = r"c:\Ocean Library\Matlab code practice\Data\images\raw\_DSC5150.ARW"
JPG_PATH = r"c:\Ocean Library\Matlab code practice\Data\images\raw\_DSC5150.JPG"
OUT_PATH = r"c:\Ocean Library\Matlab code practice\Data\images\raw\raw_demosaic_data.mat"

print("Loading RAW and JPEG files...")
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Missing RAW file: {RAW_PATH}")
if not os.path.exists(JPG_PATH):
    raise FileNotFoundError(f"Missing JPEG file: {JPG_PATH}")

with rawpy.imread(RAW_PATH) as raw:
    # 1. Inspect sizes and parameters
    black = float(raw.black_level_per_channel[0]) # Assumes uniform black level
    white = float(raw.white_level)
    wb = [float(g) for g in raw.camera_whitebalance] # [R, G, B, G]
    
    # Normalize WB gains to green (assumed at index 1)
    r_gain = wb[0] / wb[1]
    g_gain = 1.0
    b_gain = wb[2] / wb[1]
    normalized_wb = [r_gain, g_gain, b_gain]
    
    print(f"Sensor Black Level: {black}")
    print(f"Sensor White Level: {white}")
    print(f"Normalized white balance gains: {normalized_wb}")
    
    # 2. Convert raw image to double and subtract black level
    raw_img = raw.raw_image.astype(np.float64)
    raw_norm = (raw_img - black) / (white - black)
    raw_norm = np.clip(raw_norm, 0.0, 1.0)
    
    # 3. Align with camera active image crop
    # Raw active size starts at row 12, col 8 relative to raw sensor size
    cfa_active = raw_norm[12:3644, 8:5464]

# Load camera JPEG
jpg_img = cv2.imread(JPG_PATH)
# Convert BGR to RGB
jpg_rgb = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)

print(f"Aligned active CFA shape: {cfa_active.shape}")
print(f"JPEG shape              : {jpg_rgb.shape}")

# 4. Crop 2048 x 2048 center ROI
H_jpg, W_jpg = jpg_rgb.shape[:2]
H_roi, W_roi = 2048, 2048

r_start = (H_jpg - H_roi) // 2 # (3632 - 2048) // 2 = 792
c_start = (W_jpg - W_roi) // 2 # (5456 - 2048) // 2 = 1704

r_end = r_start + H_roi
c_end = c_start + W_roi

print(f"Cropping center ROI from index ({r_start}, {c_start}) to ({r_end}, {c_end})...")

cfa_crop = cfa_active[r_start:r_end, c_start:c_end]
jpg_crop = jpg_rgb[r_start:r_end, c_start:c_end]

print(f"CFA cropped shape : {cfa_crop.shape}")
print(f"JPEG cropped shape: {jpg_crop.shape}")

# Save to .mat
sio.savemat(OUT_PATH, {
    'cfa_crop': cfa_crop,
    'jpg_crop': jpg_crop,
    'wb_gains': normalized_wb
})

print(f"Successfully saved data to {OUT_PATH}")
