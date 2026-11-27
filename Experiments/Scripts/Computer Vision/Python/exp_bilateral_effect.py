"""
exp_bilateral_effect.py

Scientific Experiment: Understanding the Edge-Preserving Effect of Bilateral Filtering

This script demonstrates and evaluates the fundamental differences between:
  1. Standard Gaussian Blur (Linear, space-invariant, blurs edges).
  2. Bilateral Filter (Non-linear, space-variant, preserves sharp edges).

It performs two evaluations:
  - Part A: A synthetic 1D step-edge signal to show cross-sectional profile preservation.
  - Part B: Real-world image filtering using 'elon_musk.jpg' to demonstrate performance.

Visualizations are compiled into a comprehensive dark-themed dashboard.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    # -------------------------------------------------------------------------
    # 1. Paths and Setup
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    image_path = os.path.join(project_root, "Data", "images", "elon_musk.jpg")
    results_dir = os.path.join(script_dir, "..", "Results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    output_path = os.path.join(results_dir, "bilateral_edge_preservation_python.png")
    
    print(f"Project root resolved to: {project_root}")
    print(f"Loading real-world image from: {image_path}")

    # -------------------------------------------------------------------------
    # 2. Part A: Generate and Filter Synthetic Step-Edge
    # -------------------------------------------------------------------------
    print("\n--- Part A: Processing Synthetic Step-Edge ---")
    height, width = 256, 256
    I_clean_syn = np.zeros((height, width), dtype=np.float32)
    I_clean_syn[:, width // 2:] = 1.0
    
    # Add Gaussian noise
    np.random.seed(42)
    noise_syn = np.random.normal(0.0, 0.08, (height, width)).astype(np.float32)
    I_noisy_syn = np.clip(I_clean_syn + noise_syn, 0.0, 1.0)
    
    # Apply filters to synthetic edge
    sigma_space_syn = 5.0
    I_gauss_syn = cv2.GaussianBlur(I_noisy_syn, (31, 31), sigma_space_syn)
    
    sigma_range_syn = 0.25
    I_bilat_syn = cv2.bilateralFilter(I_noisy_syn, d=31, sigmaColor=sigma_range_syn, sigmaSpace=sigma_space_syn)
    
    # Extract 1D profiles across center row
    row_idx = height // 2
    profile_clean = I_clean_syn[row_idx, :]
    profile_noisy = I_noisy_syn[row_idx, :]
    profile_gauss = I_gauss_syn[row_idx, :]
    profile_bilat = I_bilat_syn[row_idx, :]

    # -------------------------------------------------------------------------
    # 3. Part B: Load and Filter Elon Musk Image
    # -------------------------------------------------------------------------
    print("\n--- Part B: Processing Elon Musk Image ---")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not locate image at: {image_path}")
        
    I_bgr = cv2.imread(image_path)
    I_rgb = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB)
    I_dbl = I_rgb.astype(np.float32) / 255.0
    
    # Resize high-resolution image to height=512 for memory efficiency
    h_orig, w_orig, c_orig = I_dbl.shape
    h_new = 512
    w_new = int((w_orig / h_orig) * h_new)
    I_resized = cv2.resize(I_dbl, (w_new, h_new), interpolation=cv2.INTER_AREA)
    
    # Add moderate Gaussian noise (0.05 standard deviation)
    noise_elon = np.random.normal(0.0, 0.05, I_resized.shape).astype(np.float32)
    I_noisy_elon = np.clip(I_resized + noise_elon, 0.0, 1.0)
    
    # Apply filters to Elon Musk image
    sigma_space_elon = 5.0
    sigma_range_elon = 0.15
    
    print("Applying Gaussian blur to Elon Musk image...")
    I_gauss_elon = cv2.GaussianBlur(I_noisy_elon, (31, 31), sigma_space_elon)
    
    print("Applying Bilateral filter to Elon Musk image...")
    # cv2.bilateralFilter needs to process float32 directly
    I_bilat_elon = cv2.bilateralFilter(I_noisy_elon, d=31, sigmaColor=sigma_range_elon, sigmaSpace=sigma_space_elon)

    # -------------------------------------------------------------------------
    # 4. Render High-Fidelity Dark-Themed Dashboard
    # -------------------------------------------------------------------------
    print("\nRendering comprehensive comparison dashboard...")
    plt.style.use('dark_background')
    
    # 3 Rows of subplots:
    #   Row 0: Synthetic Step-Edge images
    #   Row 1: 1D Profile plot (spans full width)
    #   Row 2: Elon Musk images
    fig, axs = plt.subplots(3, 4, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1, 2]})
    fig.suptitle("BILATERAL FILTERING SANDBOX: THEORY & REAL-WORLD EVALUATION", fontsize=18, fontweight='bold', color='white', y=0.97)
    
    # Subplot styling helper
    def style_subplot(ax, title):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    # Row 0: Synthetic Edge
    style_subplot(axs[0, 0], "Synthetic: Original")
    axs[0, 0].imshow(I_clean_syn, cmap='gray', vmin=0, vmax=1)
    
    style_subplot(axs[0, 1], "Synthetic: Noisy (std=0.08)")
    axs[0, 1].imshow(I_noisy_syn, cmap='gray', vmin=0, vmax=1)
    
    style_subplot(axs[0, 2], f"Synthetic: Gaussian (sigma_s={sigma_space_syn})")
    axs[0, 2].imshow(I_gauss_syn, cmap='gray', vmin=0, vmax=1)
    
    style_subplot(axs[0, 3], f"Synthetic: Bilateral (sigma_s={sigma_space_syn}, sigma_r={sigma_range_syn})")
    axs[0, 3].imshow(I_bilat_syn, cmap='gray', vmin=0, vmax=1)

    # Row 1: 1D Profile (Remove 4 subplots and replace with a single spanning subplot)
    for col in range(4):
        fig.delaxes(axs[1, col])
    
    # Add single spanning axes for Row 1
    gs = axs[1, 0].get_gridspec()
    ax_profile = fig.add_subplot(gs[1, :])
    
    x_axis = np.arange(width)
    ax_profile.plot(x_axis, profile_clean, label="Original Ideal Step", color='#888888', linestyle='--', linewidth=1.5)
    ax_profile.plot(x_axis, profile_noisy, label="Noisy Profile", color='#5555aa', alpha=0.5, linewidth=1.0)
    ax_profile.plot(x_axis, profile_gauss, label=f"Gaussian Profile (Blurs Edge, sigma_s={sigma_space_syn})", color='#ff5555', linewidth=2.5)
    ax_profile.plot(x_axis, profile_bilat, label=f"Bilateral Profile (Sharp Edge, sigma_r={sigma_range_syn})", color='#55ff55', linewidth=2.5)
    
    ax_profile.set_title("1D Intensity Profile Slice across Step-Edge", fontsize=12, fontweight='bold', pad=10)
    ax_profile.set_xlabel("Pixel Column Coordinate", fontsize=10)
    ax_profile.set_ylabel("Intensity Value [0.0, 1.0]", fontsize=10)
    ax_profile.grid(True, color='#333333', linestyle=':')
    ax_profile.set_ylim(-0.2, 1.2)
    ax_profile.legend(loc="upper left", frameon=True, facecolor='#222222', edgecolor='none')

    # Row 2: Elon Musk
    style_subplot(axs[2, 0], "Elon Musk: Original")
    axs[2, 0].imshow(I_resized)
    
    style_subplot(axs[2, 1], "Elon Musk: Noisy (std=0.05)")
    axs[2, 1].imshow(I_noisy_elon)
    
    style_subplot(axs[2, 2], f"Elon Musk: Gaussian Blur (sigma_s={sigma_space_elon})")
    axs[2, 2].imshow(I_gauss_elon)
    
    style_subplot(axs[2, 3], f"Elon Musk: Bilateral (sigma_s={sigma_space_elon}, sigma_r={sigma_range_elon})")
    axs[2, 3].imshow(I_bilat_elon)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    
    plt.show()
    print(f"\nDashboard figure successfully saved to: {output_path}")
    print("Execution completed successfully.")

if __name__ == "__main__":
    main()
