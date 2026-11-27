"""
preprocess_affectnet_landmarks.py
==================================
Preprocessing script for the AffectNet YOLO-format dataset.

PURPOSE:
    Extracts 68 facial landmark (x, y) pixel coordinates from each face image
    using Google MediaPipe FaceMesh, then saves the results as a MATLAB-compatible
    .mat file for downstream Procrustes analysis.

INPUT:
    c:/Ocean Library/Matlab code practice/Data/AffectNet/
        Train/<emotion>/image*.jpg   (face images, already cropped to face)
        labels.csv                   (index, pth, label, relFCs)

OUTPUT:
    c:/Ocean Library/Matlab code practice/Data/AffectNet/AffectNet_landmarks.mat
        X : [68 x 2 x L] float64   — landmark (x, y) coordinates per sample
        Y : [L x 1] int32           — emotion label IDs
        label_names : cell array    — {'anger','happy','neutral','surprise'}
        image_paths : cell array    — relative path per sample (for tracing)

ASSUMPTIONS:
    1. Face images are pre-cropped (AffectNet YOLO format) — face fills most of frame.
       MediaPipe FaceMesh should detect the face reliably without needing a detector.
    2. Ground truth label = subfolder name (NOT the 'label' column in labels.csv,
       which is the YOLO model's prediction, not the AffectNet original annotation).
    3. We use 4 target emotions: anger, happy, neutral, surprise (matches Ex. plan).
       Other classes (contempt, disgust, fear, sad) are skipped.
    4. We sample N_PER_CLASS images per class (balanced dataset).
    5. rng seed fixed for reproducibility: random.seed(42), np.random.seed(42).
    6. Images where MediaPipe finds no face are skipped (counted in failed_count).

DEPENDENCIES:
    pip install mediapipe opencv-python scipy numpy

USAGE:
    py preprocess_affectnet_landmarks.py

REPRODUCIBILITY:
    Run with identical DATA_ROOT and N_PER_CLASS to get the same .mat file.
"""

import os
import random
import numpy as np
import scipy.io as sio
import cv2
import mediapipe as mp

# =========================================================================
# === CONFIGURATION ===
# =========================================================================
DATA_ROOT    = r"c:\Ocean Library\Matlab code practice\Data\AffectNet\Train"
OUTPUT_PATH  = r"c:\Ocean Library\Matlab code practice\Data\AffectNet\AffectNet_landmarks.mat"

# Target emotion classes — must match subfolder names exactly
TARGET_CLASSES = ['anger', 'happy', 'neutral', 'surprise']

# Samples per class (balanced). Total L = N_PER_CLASS * len(TARGET_CLASSES).
# Set to None to use all available images per class (unbalanced).
N_PER_CLASS = 600   # => L = 2400 total

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================================================================
# === 68-POINT SUBSET from MediaPipe FaceMesh 478-point model
# =========================================================================
# MediaPipe FaceMesh returns up to 478 landmarks (468 on the face surface).
# This subset of 68 indices approximates the standard dlib 68-point layout:
#   0-16  : jawline  (17 pts)
#  17-21  : left eyebrow (5 pts)
#  22-26  : right eyebrow (5 pts)
#  27-30  : nose bridge (4 pts)
#  31-35  : nose bottom (5 pts)
#  36-41  : left eye (6 pts)
#  42-47  : right eye (6 pts)
#  48-59  : outer lips (12 pts)
#  60-67  : inner lips (8 pts)

MP_68_SUBSET = [
    # Jawline (17)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 288,
    # Left eyebrow (5)
    70, 63, 105, 66, 107,
    # Right eyebrow (5)
    336, 296, 334, 293, 300,
    # Nose bridge (4)
    168, 6, 197, 195,
    # Nose bottom (5)
    5, 4, 1, 19, 94,
    # Left eye (6)
    33, 160, 158, 133, 153, 144,
    # Right eye (6)
    362, 385, 387, 263, 373, 380,
    # Outer lips (12)
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 181,
    # Inner lips (8)
    78, 95, 88, 178, 87, 14, 317, 402,
]
assert len(MP_68_SUBSET) == 68, f"Expected 68 indices, got {len(MP_68_SUBSET)}"
N_LANDMARKS = len(MP_68_SUBSET)   # 68

# =========================================================================
# === INITIALISE MediaPipe FaceMesh (Tasks API) ===
# =========================================================================
print("[INFO] Initialising MediaPipe FaceLandmarker...")
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

model_path = os.path.join(os.path.dirname(DATA_ROOT), "face_landmarker.task")
if not os.path.exists(model_path):
    print(f"[INFO] Downloading FaceLandmarker model to {model_path}...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# =========================================================================
# === COLLECT IMAGE PATHS PER CLASS ===
# =========================================================================
print("[INFO] Scanning dataset directory...")
class_files = {}
for cls in TARGET_CLASSES:
    cls_dir = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cls_dir):
        raise FileNotFoundError(f"Expected folder not found: {cls_dir}")
    files = sorted([
        os.path.join(cls_dir, f)
        for f in os.listdir(cls_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if N_PER_CLASS is not None:
        random.shuffle(files)
        files = files[:N_PER_CLASS * 3]   # take 3× to allow for failures
    class_files[cls] = files
    print(f"  {cls}: {len(files)} candidates")

# =========================================================================
# === EXTRACT LANDMARKS ===
# =========================================================================
label_names   = TARGET_CLASSES
label_id_map  = {cls: i for i, cls in enumerate(TARGET_CLASSES)}

X_list     = []   # each entry: (68, 2) array
Y_list     = []   # each entry: int
paths_list = []   # each entry: str

for cls in TARGET_CLASSES:
    cls_id       = label_id_map[cls]
    collected    = 0
    failed       = 0
    target_count = N_PER_CLASS if N_PER_CLASS is not None else len(class_files[cls])

    print(f"\n[INFO] Processing class '{cls}' (target={target_count})...")
    for img_path in class_files[cls]:
        if collected >= target_count:
            break

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            failed += 1
            continue

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            failed += 1
            continue

        # Take the first (and only) detected face
        face_lm = detection_result.face_landmarks[0]

        # Extract 68 target landmarks, convert normalized → pixel coords
        pts = np.zeros((N_LANDMARKS, 2), dtype=np.float64)
        for i, idx in enumerate(MP_68_SUBSET):
            lm = face_lm[idx]
            pts[i, 0] = lm.x * w   # x pixel
            pts[i, 1] = lm.y * h   # y pixel

        X_list.append(pts)
        Y_list.append(cls_id)
        rel_path = os.path.relpath(img_path, os.path.dirname(DATA_ROOT))
        paths_list.append(rel_path)
        collected += 1

        if collected % 100 == 0:
            print(f"  [{cls}] {collected}/{target_count} collected, {failed} failed")

    print(f"  [{cls}] DONE: {collected} collected, {failed} failed/skipped")

detector.close()

# =========================================================================
# === PACK INTO TENSORS ===
# =========================================================================
L = len(X_list)
print(f"\n[INFO] Total samples collected: L = {L}")

# MATLAB convention: X [N x p x L]
X = np.stack(X_list, axis=2).astype(np.float64)  # (68, 2, L)
Y = np.array(Y_list, dtype=np.int32).reshape(-1, 1)   # (L, 1)

print(f"[INFO] X shape: {X.shape}")
print(f"[INFO] Y shape: {Y.shape}")
print(f"[INFO] Class distribution:")
for cls in TARGET_CLASSES:
    count = np.sum(Y == label_id_map[cls])
    print(f"  {cls} (id={label_id_map[cls]}): {count} samples")

# =========================================================================
# === SAVE ===
# =========================================================================
print(f"\n[INFO] Saving to: {OUTPUT_PATH}")
sio.savemat(OUTPUT_PATH, {
    'X':           X,
    'Y':           Y,
    'label_names': np.array(label_names, dtype=object),
    'image_paths': np.array(paths_list, dtype=object),
    'N_per_class': N_PER_CLASS,
    'random_seed': RANDOM_SEED,
    'mp_68_subset': np.array(MP_68_SUBSET, dtype=np.int32),
})
print("[INFO] Done. Saved successfully.")
print(f"\n=== PREPROCESSING SUMMARY ===")
print(f"  Source  : {DATA_ROOT}")
print(f"  Classes : {TARGET_CLASSES}")
print(f"  L total : {L}")
print(f"  X shape : {X.shape}  (N={N_LANDMARKS} landmarks, p=2, L={L})")
print(f"  Y shape : {Y.shape}")
print(f"  Output  : {OUTPUT_PATH}")
print(f"  Seed    : {RANDOM_SEED}")
print("==============================")
