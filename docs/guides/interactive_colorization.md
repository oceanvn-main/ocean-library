# Interactive Image Colorization

## Overview

Implements **Szeliski §4.2.4 "Application: Interactive Colorization"** based on the optimization framework by Levin, Lischinski & Weiss (2004).

Given a black-and-white (grayscale) image $Y$ and user-drawn color scribbles at sparse pixel locations, the algorithm propagates chrominance $(U, V)$ values across the entire image by enforcing local smoothness weighted inversely by luminance gradients:

$$w_{rs} = \exp\left( -\frac{(Y(r) - Y(s))^2}{2 \sigma_r^2} \right)$$

---

## Math & Solver

The chrominance channels $U$ and $V$ in YUV space are solved independently via the sparse linear system:

$$\left[ M + \lambda (I - W)^T (I - W) \right] C = M C_{\text{scribble}}$$

where:
- $M$ is the binary diagonal matrix indicating scribble locations
- $W$ is the 8-neighbor row-normalized affinity matrix ($\sum_s w_{rs} = 1$)
- $C \in \{U, V\}$ is the chrominance vector
- $\lambda$ controls data fidelity vs. smoothness

---

## Usage

### 1. Programmatic API (`LevinColorizer`)

```matlab
import OceanMath.Image.Variational.LevinColorizer

colorizer = LevinColorizer('Sigma', 0.05, 'Lambda', 1.0);
I_rgb = colorizer.colorize(I_gray, scribbles_rgb, mask_binary);
```

### 2. Interactive Brush UI (`InteractiveImageCanvas`)

```matlab
import OceanViz.App.InteractiveImageCanvas

% Launch interactive brush canvas
canvas = InteractiveImageCanvas.launch(I_gray);
```

Features:
- Mouse painting brush tool with configurable radius
- Preset color swatches
- Auto-exports results to `OceanData.ExperimentResult` and inspects in `OceanViz.App.ExperimentStudio`

---

## Examples

See [`exp_interactive_colorization.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_interactive_colorization.m) for a complete working demonstration.
