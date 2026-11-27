# Model-Driven Dynamic Visualization Architecture

## Overview

The **Model-Driven Dynamic Visualization Architecture** decouples scientific/ML/vision algorithm logic from GUI visualization code.

Instead of writing custom interactive GUIs for every experiment script, experiment outputs are packaged into a standardized, self-describing container (`OceanData.ExperimentResult`). The master studio app (`OceanViz.App.ExperimentStudio`) then inspects, classifies, and dynamically renders the data using the `OceanViz.Core` engine.

---

## Core Components

### 1. `OceanData.ExperimentResult`
The data transfer object (DTO) that encapsulates experiment outputs and metadata:

```matlab
res = OceanData.ExperimentResult('Experiment Title', 'Description', 'Text description');

% Add hyperparameters & scalar metrics
res.addParam('Lambda', 0.08);
res.addMetric('PSNR_dB', 28.45);

% Add data channels (automatically type-inspected)
res.add('NoisyImage', g_noisy, 'Description', 'Input image');
res.add('Denoised', f_tv, 'Description', 'TV output');
res.add('EnergyHistory', h.Energy, 'Description', 'Energy per iteration');

% Save or load from disk
res.save('experiment_output.mat');
res_loaded = OceanData.ExperimentResult.load('experiment_output.mat');
```

#### Auto-Classification Rules:
- **`scalar`** → Tracked in `.Metrics` / `.Parameters` tables
- **`vector_1d`** → Assigned to `TimeSeries` visualization strategy
- **`image_gray` / `image_rgb`** → Assigned to `Image` visualization strategy
- **`matrix_nd`** → Assigned to `PartialDependence2D` heatmap strategy

---

### 2. `OceanViz.App.ExperimentStudio`
The master generic interactive studio UI.

```matlab
% Launch studio directly on an ExperimentResult object or MAT file
OceanViz.App.ExperimentStudio.inspect(res);
```

#### Features:
- **Variable Inspector Sidebar:** Lists all variables with auto-type badges (`[1D]`, `[IMAGE_GRAY]`, `[IMAGE_RGB]`, `[SCALAR]`).
- **Multi-Selection Viewport:** Select 1, 2, or up to 4 variables simultaneously.
- **Dynamic Layout:** Auto-arranges 1-up, 1x2 split, or 2x2 grid views.
- **Theme Switcher:** One-click toggle between `Dark` and `Scientific` themes.
- **High-Res Export:** Save active viewport image to PNG with one click.

---

### 3. `OceanViz.Strategies.ImageVizStrategy`
Integrated into `OceanViz.Core`:

```matlab
viz = OceanViz.Core('Theme', 'Dark');
viz.plot(ax, 'Image', struct('Image', I, 'Title', 'Reconstructed Output'));
```

---

## Example Script

See [`exp_model_driven_viz_demo.m`](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_model_driven_viz_demo.m) for a complete working example.
