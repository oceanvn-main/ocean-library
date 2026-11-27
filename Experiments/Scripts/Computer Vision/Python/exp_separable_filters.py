"""
exp_separable_filters.py

Replicating Szeliski's Figure 3.14 (Separable Linear Filters) using PyTorch.
This represents a standard production implementation utilizing deep learning framework
primitives (tensors, F.pad, and F.conv2d) to perform high-performance 2D image filtering.

Author: Antigravity Code Assistant
Date: 2026-06-15
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def draw_kernel_grid(ax, K, title_str):
    K_np = K.numpy()
    rows, cols = K_np.shape
    max_val = np.max(np.abs(K_np))
    if max_val == 0:
        max_val = 1.0
        
    # Draw heatmap
    im = ax.imshow(K_np, cmap='cool', vmin=-max_val, vmax=max_val)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Draw grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color='#4c4c59', linewidth=1.0)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color='#4c4c59', linewidth=1.0)
        
    # Draw coefficients text
    for r in range(rows):
        for c in range(cols):
            val = K_np[r, c]
            # Print integer value (since grids are multiplied by denominators)
            val_str = f"{int(round(val))}"
            ax.text(c, r, val_str, color='white', fontweight='bold',
                    fontsize=10, ha='center', va='center')
            
    ax.set_title(title_str, color='#e5e5e5', fontsize=11, fontweight='bold', pad=8)

def main():
    # 1. Paths and Directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "..", "..", "..", "..", "Data", "images", "Squid-game-symbols.jpg")
    results_dir = os.path.join(script_dir, "..", "Results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "separable_filters_pytorch.png")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    # 2. Load Image and Transform (Grayscale + Resize to 256x256)
    img_pil = Image.open(image_path).convert('L')
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    
    # Convert to PyTorch Tensor: shape [1, 1, H, W] for Conv2d
    input_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    
    # Downsample to 256x256 using bilinear interpolation
    input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    
    # 3. Define the 5 Filters with Analytical 1D Decompositions
    # We define both the 2D kernels and their 1D horizontal (h) and vertical (v) components.
    # Note that PyTorch Conv2D computes cross-correlation, so we pre-flip Sobel horizontally.
    
    # (a) Box filter (5x5)
    box_h = torch.ones(1, 1, 1, 5) / 5.0
    box_v = torch.ones(1, 1, 5, 1) / 5.0
    box_2d = torch.ones(1, 1, 5, 5) / 25.0
    
    # (b) Bilinear / Triangle filter (3x3)
    bil_h = torch.tensor([[1, 2, 1]]).float().unsqueeze(0).unsqueeze(0) / 4.0
    bil_v = torch.tensor([[1], [2], [1]]).float().unsqueeze(0).unsqueeze(0) / 4.0
    bil_2d = torch.outer(torch.tensor([1, 2, 1]).float(), torch.tensor([1, 2, 1]).float()).unsqueeze(0).unsqueeze(0) / 16.0
    
    # (c) Gaussian binomial filter (5x5)
    gauss_h = torch.tensor([[1, 4, 6, 4, 1]]).float().unsqueeze(0).unsqueeze(0) / 16.0
    gauss_v = torch.tensor([[1], [4], [6], [4], [1]]).float().unsqueeze(0).unsqueeze(0) / 16.0
    gauss_2d = torch.outer(torch.tensor([1, 4, 6, 4, 1]).float(), torch.tensor([1, 4, 6, 4, 1]).float()).unsqueeze(0).unsqueeze(0) / 256.0
    
    # (d) Sobel filter (3x3 horizontal derivative)
    # True mathematical convolution rotates the kernel by 180 degrees.
    # For Sobel horizontal derivative [-1 0 1; -2 0 2; -1 0 1] / 8, true convolution flips it to [1 0 -1; 2 0 -2; 1 0 -1] / 8.
    sobel_h = torch.tensor([[1, 0, -1]]).float().unsqueeze(0).unsqueeze(0) / 4.0
    sobel_v = torch.tensor([[1], [2], [1]]).float().unsqueeze(0).unsqueeze(0) / 2.0
    sobel_2d = torch.outer(torch.tensor([1, 2, 1]).float(), torch.tensor([1, 0, -1]).float()).unsqueeze(0).unsqueeze(0) / 8.0
    
    # (e) Corner filter (3x3 second-order cross-derivative)
    # [1 -2 1; -2 4 -2; 1 -2 1] / 4
    corner_h = torch.tensor([[1, -2, 1]]).float().unsqueeze(0).unsqueeze(0) / 2.0
    corner_v = torch.tensor([[1], [-2], [1]]).float().unsqueeze(0).unsqueeze(0) / 2.0
    corner_2d = torch.outer(torch.tensor([1, -2, 1]).float(), torch.tensor([1, -2, 1]).float()).unsqueeze(0).unsqueeze(0) / 4.0
    
    filters_2d = [box_2d, bil_2d, gauss_2d, sobel_2d, corner_2d]
    filters_h = [box_h, bil_h, gauss_h, sobel_h, corner_h]
    filters_v = [box_v, bil_v, gauss_v, sobel_v, corner_v]
    
    # 4. Perform Separable Convolution in PyTorch
    # In production, separable convolution is executed by running two 1D convolutions:
    #   Step 1: pad and convolve horizontally
    #   Step 2: pad and convolve vertically
    # This reduces complexity from O(K^2) to O(2K).
    
    outputs = []
    for f_h, f_v, f_2d in zip(filters_h, filters_v, filters_2d):
        k_w = f_h.shape[3]
        k_h = f_v.shape[2]
        
        pad_w = k_w // 2
        pad_h = k_h // 2
        
        # Horizontal Pass
        padded_h = F.pad(input_tensor, (pad_w, pad_w, 0, 0), mode='replicate')
        mid = F.conv2d(padded_h, f_h, padding=0)
        
        # Vertical Pass
        padded_v = F.pad(mid, (0, 0, pad_h, pad_h), mode='replicate')
        out_sep = F.conv2d(padded_v, f_v, padding=0)
        
        # Verify correctness against non-separable 2D convolution
        padded_2d = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        out_2d = F.conv2d(padded_2d, f_2d, padding=0)
        
        diff = torch.max(torch.abs(out_sep - out_2d)).item()
        assert diff < 1e-6, f"Separable filtering mismatch: diff = {diff}"
        
        outputs.append(out_sep.squeeze().numpy())
        
    # Scale and shift derivative outputs (Sobel and Corner) to mid-gray offset for display
    # Sobel is outputs[3] (index 3), Corner is outputs[4] (index 4)
    outputs[3] = np.clip(2.0 * outputs[3] + 0.5, 0.0, 1.0)
    outputs[4] = np.clip(4.0 * outputs[4] + 0.5, 0.0, 1.0)
    
    # 5. Plotting (Premium Dark Theme matching MATLAB replica)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 5, figsize=(14, 9), facecolor='#080a0d')
    fig.suptitle('SEPARABLE LINEAR FILTERS (Szeliski Fig 3.14 PyTorch Production Replica)',
                 color='white', fontsize=16, fontweight='bold', y=0.96)
                 
    col_titles = [
        '(a) box, K = 5 (1/25)',
        '(b) bilinear (1/16)',
        '(c) "Gaussian" (1/256)',
        '(d) Sobel (1/8)',
        '(e) corner (1/4)'
    ]
    col_titles_1d = [
        '1D box (1/5)',
        '1D bilinear (1/4)',
        '1D "Gaussian" (1/16)',
        '1D Sobel (1/2)',
        '1D corner (1/2)'
    ]
    
    # Multipliers to convert double matrices to clean integer grids
    mults_2d = [25.0, 16.0, 256.0, 8.0, 4.0]
    mults_1d = [5.0, 4.0, 16.0, 2.0, 2.0]
    
    for col in range(5):
        # Row 1: 2D Grid
        draw_kernel_grid(axes[0, col], filters_2d[col].squeeze(0).squeeze(0) * mults_2d[col], col_titles[col])
        axes[0, col].set_facecolor('#080a0d')
        
        # Row 2: 1D Grid
        draw_kernel_grid(axes[1, col], filters_h[col].squeeze(0).squeeze(0) * mults_1d[col], col_titles_1d[col])
        axes[1, col].set_facecolor('#080a0d')
        
        # Row 3: Output Image
        axes[2, col].imshow(outputs[col], cmap='gray', vmin=0.0, vmax=1.0)
        axes[2, col].axis('image')
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])
        for spine in axes[2, col].spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(1.0)
            
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    plt.savefig(output_path, dpi=100, facecolor='#080a0d')
    plt.show()
    
    print("Separable filters PyTorch replication completed successfully.")
    print(f"Snapshot generated successfully: {output_path}")

if __name__ == "__main__":
    main()
