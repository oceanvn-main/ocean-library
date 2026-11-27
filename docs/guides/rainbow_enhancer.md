# Rainbow Separation and Enhancer Guide (Exercise 3.29)

Step-by-step image linearization, binary mask ROI definition, Laplacian background synthesis, and linear subtraction.

## Overview
This guide covers the extraction and enhancement of weak rainbow profiles from digital images in linear radiometric space. The pipeline consists of:
1. **Ingestion & Linearization**: Removing display gamma encoding via $L_{total} = P_{total}^{2.2}$ to perform calculations in physical light space.
2. **Spatial Definition (ROI)**: Tracing a boundary around the rainbow region to generate a logical binary mask $M(x,y)$.
3. **Background Synthesis (Inpainting)**: Estimating the pure background sky behind the rainbow by solving Laplace's equation ($\nabla^2 L_{bg} = 0$) across the mask boundaries.
4. **Mathematical Subtraction**: Isolating the rainbow light energy via $L_{rainbow} = \max(L_{total} - L_{bg}, 0)$, where zero-clamping prevents negative values.
5. **Re-encoding & Contrast Boost**: Multiplying the isolated signal by a gain factor $k$, compositing, and applying inverse gamma compression: $P_{isolated} = (k \cdot L_{rainbow})^{1/2.2}$.

## Usage
The interactive wizard GUI and pipeline classes are located in:
- App: `C:\Ocean Library\Matlab code practice\+OceanViz\+App\Computer Vision\RainbowEnhancer.m`
- Experiment script: `C:\Ocean Library\Matlab code practice\Experiments\Scripts\Computer Vision\exp_rainbow_enhancer.m`

To launch the script and standalone app:
```matlab
% Set up paths
rootDir = 'C:\Ocean Library\Matlab code practice';
addpath(genpath(rootDir));
addpath(fullfile(rootDir, '+OceanViz', '+App', 'Computer Vision'));

% Run the experiment pipeline and launch the GUI
exp_rainbow_enhancer;
```

## Examples

### 1. Launching the App with a Custom Image
```matlab
% Read image
I = im2double(imread('C:\Ocean Library\Matlab code practice\Data\rainbow_dark.jpg'));

% Instantiate the step-by-step wizard GUI
app = RainbowEnhancer('Image', I);
app.show();
```

### 2. Dashboard Interface Wizard Flow
1. **Step 1: Load Image**: Loads the source file into the 2x3 dashboard.
2. **Step 2: Linearize**: Strips gamma compression to leave raw physical light intensities.
3. **Step 3: Draw Mask**: Clicking the button launches the freehand tracing cursor on Axes 1. Drag to outline the rainbow.
4. **Step 4: Synthesize Background**: Runs Laplacian inpainting (`regionfill`) to reconstruct the sky.
5. **Step 5: Isolate & Boost**: Subtracts the background and applies the slider gain value $k$ in real-time.

## See Also
- [Pixel Transformations Guide](pixel_transforms.md)
- [Spatial Filtering & Lifting Guide](filtering_and_lifting.md)
- [Image Pyramids Guide](pyramids.md)
