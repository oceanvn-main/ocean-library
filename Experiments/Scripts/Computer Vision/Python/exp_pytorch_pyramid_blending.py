"""
exp_pytorch_pyramid_blending.py

Solving Exercise 3.19: Pyramid blending in PyTorch.
This script implements multiresolution Laplacian pyramid blending using PyTorch DNN primitives,
compares the results against an identical reference implementation in SciPy/NumPy,
and provides a detailed analysis of the two APIs.

Author: Antigravity Code Assistant
Date: 2026-07-13
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image

def get_gaussian_kernel_2d():
    # 5-tap Gaussian binomial filter coefficients: [1, 4, 6, 4, 1] / 16
    w = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    kernel_2d = np.outer(w, w)
    return kernel_2d

# =====================================================================
# 1. PyTorch Implementation (DNN Primitives)
# =====================================================================

def pytorch_reduce(I, kernel_2d):
    """
    Downsamples tensor by a factor of 2.
    First pads with replicate, then performs strided convolution.
    """
    C = I.shape[1]
    weight = torch.from_numpy(kernel_2d).view(1, 1, 5, 5).repeat(C, 1, 1, 1).to(I.device)
    
    # Pad with replicate (2 pixels on all sides for a 5x5 kernel)
    padded = F.pad(I, (2, 2, 2, 2), mode='replicate')
    
    # Stride of 2 downsamples directly during convolution.
    # groups=C filters each channel separately.
    out = F.conv2d(padded, weight, stride=2, groups=C)
    return out

def pytorch_expand(I, target_shape, kernel_2d):
    """
    Upsamples tensor to target_shape using zero-stuffing and interpolation convolution.
    """
    C = I.shape[1]
    H_t, W_t = target_shape[2], target_shape[3]
    
    # Zero-stuffing (polyphase upsampling base)
    upsampled = torch.zeros((I.shape[0], C, H_t, W_t), dtype=I.dtype, device=I.device)
    upsampled[:, :, ::2, ::2] = I
    
    # Pad
    padded = F.pad(upsampled, (2, 2, 2, 2), mode='replicate')
    
    # Convolve with kernel multiplied by 4.0 to restore correct brightness/energy
    weight = torch.from_numpy(4.0 * kernel_2d).view(1, 1, 5, 5).repeat(C, 1, 1, 1).to(I.device)
    out = F.conv2d(padded, weight, stride=1, groups=C)
    return out

def pytorch_gaussian_pyramid(I, num_levels, kernel_2d):
    gp = [I]
    for _ in range(1, num_levels):
        gp.append(pytorch_reduce(gp[-1], kernel_2d))
    return gp

def pytorch_laplacian_pyramid(I, num_levels, kernel_2d):
    gp = pytorch_gaussian_pyramid(I, num_levels, kernel_2d)
    lp = []
    for l in range(num_levels - 1):
        expanded = pytorch_expand(gp[l+1], gp[l].shape, kernel_2d)
        lp.append(gp[l] - expanded)
    lp.append(gp[-1])  # Residual
    return lp

def pytorch_reconstruct_laplacian(lp, kernel_2d):
    num_levels = len(lp)
    curr = lp[-1]
    for l in reversed(range(num_levels - 1)):
        expanded = pytorch_expand(curr, lp[l].shape, kernel_2d)
        curr = lp[l] + expanded
    return curr

# =====================================================================
# 2. NumPy / SciPy Reference Implementation
# =====================================================================

def scipy_reduce(I, kernel_2d):
    """
    Downsamples NumPy array by factor of 2.
    First blurs with nearest boundary padding, then decimes.
    """
    C = I.shape[2]
    blurred = np.zeros_like(I)
    for c in range(C):
        blurred[:, :, c] = scipy.ndimage.convolve(I[:, :, c], kernel_2d, mode='nearest')
    return blurred[::2, ::2, :]

def scipy_expand(I, target_shape, kernel_2d):
    """
    Upsamples NumPy array to target_shape using zero-stuffing and convolution.
    """
    H_t, W_t, C = target_shape
    upsampled = np.zeros((H_t, W_t, C), dtype=I.dtype)
    upsampled[::2, ::2, :] = I
    
    out = np.zeros_like(upsampled)
    for c in range(C):
        out[:, :, c] = scipy.ndimage.convolve(upsampled[:, :, c], 4.0 * kernel_2d, mode='nearest')
    return out

def scipy_gaussian_pyramid(I, num_levels, kernel_2d):
    gp = [I]
    for _ in range(1, num_levels):
        gp.append(scipy_reduce(gp[-1], kernel_2d))
    return gp

def scipy_laplacian_pyramid(I, num_levels, kernel_2d):
    gp = scipy_gaussian_pyramid(I, num_levels, kernel_2d)
    lp = []
    for l in range(num_levels - 1):
        expanded = scipy_expand(gp[l+1], gp[l].shape, kernel_2d)
        lp.append(gp[l] - expanded)
    lp.append(gp[-1])
    return lp

def scipy_reconstruct_laplacian(lp, kernel_2d):
    num_levels = len(lp)
    curr = lp[-1]
    for l in reversed(range(num_levels - 1)):
        expanded = scipy_expand(curr, lp[l].shape, kernel_2d)
        curr = lp[l] + expanded
    return curr

# =====================================================================
# 3. Main Runner & Validation
# =====================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..", "..", "..")
    image_dir = os.path.join(project_root, "Data", "images")
    results_dir = os.path.join(script_dir, "..", "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Images
    apple_path = os.path.join(image_dir, "apple.jpg")
    orange_path = os.path.join(image_dir, "orange.jpg")
    
    if not os.path.exists(apple_path) or not os.path.exists(orange_path):
        raise FileNotFoundError("apple.jpg and orange.jpg must be present in Data/images/")
        
    img_apple = Image.open(apple_path).resize((512, 512))
    img_orange = Image.open(orange_path).resize((512, 512))
    
    # Convert to double [0, 1] arrays
    np_A = np.array(img_apple, dtype=np.float32) / 255.0
    np_B = np.array(img_orange, dtype=np.float32) / 255.0
    
    # 2. Create Mask (sharp vertical split)
    np_M = np.zeros((512, 512, 3), dtype=np.float32)
    np_M[:, :256, :] = 1.0  # Left side = 1 (Apple), Right side = 0 (Orange)
    
    # Convert to PyTorch Tensors: shape [1, C, H, W]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_A = torch.from_numpy(np_A).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor_B = torch.from_numpy(np_B).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor_M = torch.from_numpy(np_M).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 3. Blending Parameters
    num_levels = 6
    kernel_2d = get_gaussian_kernel_2d()
    
    # --- Execute PyTorch Pipeline ---
    print("Building pyramids and blending in PyTorch...")
    lp_A_torch = pytorch_laplacian_pyramid(tensor_A, num_levels, kernel_2d)
    lp_B_torch = pytorch_laplacian_pyramid(tensor_B, num_levels, kernel_2d)
    gp_M_torch = pytorch_gaussian_pyramid(tensor_M, num_levels, kernel_2d)
    
    lp_blend_torch = []
    for l in range(num_levels):
        # Blend equation: L_blend = GM * LA + (1 - GM) * LB
        blended_level = gp_M_torch[l] * lp_A_torch[l] + (1.0 - gp_M_torch[l]) * lp_B_torch[l]
        lp_blend_torch.append(blended_level)
        
    out_torch_tensor = pytorch_reconstruct_laplacian(lp_blend_torch, kernel_2d)
    out_torch_tensor = torch.clamp(out_torch_tensor, 0.0, 1.0)
    
    # Convert result back to NumPy [H, W, C]
    np_orapple_torch = out_torch_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # --- Execute NumPy/SciPy Reference Pipeline ---
    print("Building pyramids and blending in NumPy/SciPy...")
    lp_A_sci = scipy_laplacian_pyramid(np_A, num_levels, kernel_2d)
    lp_B_sci = scipy_laplacian_pyramid(np_B, num_levels, kernel_2d)
    gp_M_sci = scipy_gaussian_pyramid(np_M, num_levels, kernel_2d)
    
    lp_blend_sci = []
    for l in range(num_levels):
        blended_level = gp_M_sci[l] * lp_A_sci[l] + (1.0 - gp_M_sci[l]) * lp_B_sci[l]
        lp_blend_sci.append(blended_level)
        
    np_orapple_sci = scipy_reconstruct_laplacian(lp_blend_sci, kernel_2d)
    np_orapple_sci = np.clip(np_orapple_sci, 0.0, 1.0)
    
    # --- Compare Results ---
    max_diff = np.max(np.abs(np_orapple_torch - np_orapple_sci))
    print(f"\nVerification Results:")
    print(f"  Maximum Absolute Difference between PyTorch and SciPy: {max_diff:.3e}")
    
    # Identity threshold check
    assert max_diff < 1e-5, f"Verification failed! Difference {max_diff:.2e} is larger than tolerance 1e-5."
    print("  SUCCESS: PyTorch and SciPy blended outputs are mathematically identical!")
    
    # Create a simple sharp splice for reference
    np_splice = np_M * np_A + (1.0 - np_M) * np_B
    
    # 4. Plot and Save Comparison Dashboard
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor='#0b0d10')
    fig.suptitle('Laplacian Pyramid Blending Comparison (Exercise 3.19)', color='white', fontsize=15, fontweight='bold', y=0.96)
    
    # Original Images
    axes[0, 0].imshow(np_A)
    axes[0, 0].set_title('Image A: Apple', color='#e5e5e5')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np_B)
    axes[0, 1].set_title('Image B: Orange', color='#e5e5e5')
    axes[0, 1].axis('off')
    
    # Direct Splice
    axes[0, 2].imshow(np_splice)
    axes[0, 2].set_title('Sharp Direct Splice', color='#e5e5e5')
    axes[0, 2].axis('off')
    
    # PyTorch Blend
    axes[1, 0].imshow(np_orapple_torch)
    axes[1, 0].set_title('PyTorch Pyramid Blend', color='#e5e5e5')
    axes[1, 0].axis('off')
    
    # SciPy Blend
    axes[1, 1].imshow(np_orapple_sci)
    axes[1, 1].set_title('SciPy Reference Blend', color='#e5e5e5')
    axes[1, 1].axis('off')
    
    # Difference Map (Amplified x100 to show any discrepancies)
    diff_map = np.abs(np_orapple_torch - np_orapple_sci) * 100.0
    diff_map = np.clip(diff_map, 0.0, 1.0)
    axes[1, 2].imshow(diff_map, cmap='hot')
    axes[1, 2].set_title('Difference Map (x100)', color='#e5e5e5')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_img_path = os.path.join(results_dir, "orapple_pytorch_vs_scipy.png")
    plt.savefig(output_img_path, dpi=120, facecolor='#0b0d10')
    print(f"Comparison dashboard saved successfully to: {output_img_path}")
    plt.show()

if __name__ == '__main__':
    main()

# =====================================================================
# 4. Discussion: PyTorch vs. SciPy/NumPy/MATLAB APIs (Exercise 3.19, Part 4)
# =====================================================================
"""
API Discussion & Comparison:
-----------------------------
We compare the PyTorch DNN API against NumPy/SciPy/MATLAB for fixed pipeline image tasks:

1. Advantages of PyTorch:
   - GPU Acceleration: Fixed pipeline convolutions, padding, and arithmetic operations
     can be executed on a CUDA device with zero changes to the underlying logic, scaling
     to multi-megapixel images or video sequences effortlessly.
   - Batch Processing: Tensors natively support a batch dimension [B, C, H, W]. Blending
     an entire folder of images is parallelized automatically.
   - Separate Channel Filtering (groups=C): PyTorch conv2d allows processing multiple
     channels in parallel without write loops. By setting groups=in_channels, it acts
     as a depthwise-separable conv, which is cleaner than loops in MATLAB/SciPy.
   - Autodiff Compatibility: If pyramid blending is part of a larger neural network pipeline
     (e.g., style transfer or texture synthesis), gradients flow back through the blending
     and reconstruction steps automatically.

2. Disadvantages of PyTorch:
   - Overhead and Type Matching: Tensors require careful management of dimensions, shapes
     (permutes from HWC to BCHW), and precision types (double vs float32).
   - Padding API: F.pad uses tuples for dimensions in reverse order, which is prone to
     indexing bugs.
   - Lack of Out-of-the-Box Image Formats: PyTorch has no direct imread/imshow. It requires
     importing Pillow/Matplotlib to bridge raw images, creating boilerplate transfer code.

Conclusion:
For lightweight, single-image research tasks, the NumPy/SciPy or MATLAB API is simpler and
less verbose. However, for batch processing, high-resolution pipelines, or deep learning
integrations, PyTorch is much more powerful.
"""
