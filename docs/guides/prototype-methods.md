# Prototype Methods & Nearest-Neighbors

## Overview

Prototype methods represent each class by a set of prototype points, then classify new observations by nearest-prototype. Two key algorithms:

- **K-means prototypes**: Run K-means within each class to find R cluster centers. Classify by nearest prototype.
- **LVQ (Learning Vector Quantization)**: Refine K-means prototypes via competitive learning — pull correct prototypes toward data, push incorrect ones away.

## Components

| Class | Location | Purpose |
|-------|----------|---------|
| `ESLMixtureGenerator` | `+OceanData/+Generators/` | K-class mixture data (configurable via `NumClasses`) |
| `LVQ1` | `+OceanStats/+Supervised/+Neighbors/` | LVQ1 algorithm (Kohonen, 1989) |
| `FunctionalDataStrategy` | `+OceanViz/+Strategies/` | Generic overlaid curves by group (spectra, waveforms, etc.) |

## Usage

### Generate K-class mixture data
```matlab
gen = OceanData.Generators.ESLMixtureGenerator('N', 300, 'NumClasses', 3, 'CentersPerClass', 10);
[X, y] = gen.generate();
```

### Train LVQ
```matlab
lvq = OceanStats.Supervised.Neighbors.LVQ1('R', 5, 'MaxEpochs', 50);
lvq.fit(X_train, y_train);
y_pred = lvq.predict(X_test);
```

### Visualize Prototypes (2D)
```matlab
viz = OceanViz.Core('Theme', 'Dark');
data.X = X;  data.Y = y;
data.Prototypes = lvq.Prototypes;
data.ProtoLabels = lvq.PrototypeLabels;
data.Grid = struct('xx', xx, 'yy', yy, 'zz', zz);
viz.plot(ax, 'PrototypeClassifier', data, 'Title', 'LVQ - 5 Prototypes');
```

### Visualize Functional Data (any N x P curves)
```matlab
viz = OceanViz.Core('Theme', 'Scientific');
dataViz = struct('X', X, 'Y', y, 'xAxis', 0:255, ...
    'classNames', {{'aa','ao','dcl','iy','sh'}});
viz.plot(ax, 'FunctionalData', dataViz, 'ShowLegend', true);
```

### MDA2 (Shared-Center Mixture Discriminant)
```matlab
mda2 = OceanStats.Supervised.Discriminant.MDA2('R', 15, 'MaxIter', 50);
mda2.fit(X_train, y_train);
y_pred = mda2.predict(X_test);
coords = mda2.project(X_test, 2);  % canonical projection
```

## Examples

- `exp_prototype_methods.m` — LVQ vs K-means (ESL 13.1)
- `exp_phoneme_prototypes.m` — Smooth prototypes (ESL 12.5)
- `exp_mda2_shared_prototypes.m` — MDA vs MDA2 (ESL 12.11)

## See Also

- `OceanStats.Supervised.Discriminant.MDA` — Standard Mixture Discriminant Analysis
- `OceanStats.Supervised.Discriminant.MDA2` — Shared-center MDA (ESL 12.11)
- `OceanStats.Inference.Recipes.Recipe_MDA2_EM` — EM recipe (IMMRecipe)
- `OceanStats.Unsupervised.KMeans` — Used for prototype initialization
- `OceanStats.NonParametric.getNSBasis` — Natural spline basis for smooth prototypes
- ESL Chapters 12.5, 12.11, and 13.2
