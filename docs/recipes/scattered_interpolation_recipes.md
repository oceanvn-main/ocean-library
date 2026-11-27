# Scattered Data Interpolation Recipes

Copy-paste recipes for common scattered data interpolation tasks in MATLAB using the Ocean Library.

---

## Recipe 1: Fit and Compare Multiple Scattered Interpolants

Fit RBF, Nadaraya-Watson, Delaunay Natural Neighbour, and KNN on noisy 2D scattered data and display a comparison table.

```matlab
% 1. Generate 2D scattered dataset
gen = OceanData.Generators.SmoothFunctionGenerator(...
    'N', 50, 'Dim', 2, 'Mode', 'gaussian', 'NoiseStd', 0.05, 'Seed', 42);
[X_train, Y_train] = gen.generate();
f_true = gen.MetaData.f_true;

% 2. Setup ScatterFitter orchestrator
import OceanMath.Interpolation.ScatteredInterpolant
fitter = OceanStats.Interpolation.ScatterFitter('Dim', 2, 'NGrid', 80);

% 3. Register model configurations
fitter.addModel('RBF-ThinPlate',       ScatteredInterpolant('Method', 'rbf', 'Kernel', 'thinplate'));
fitter.addModel('RBF-Gaussian',        ScatteredInterpolant('Method', 'rbf', 'Kernel', 'gaussian', 'Epsilon', 0.3));
fitter.addModel('KernelRegression',    ScatteredInterpolant('Method', 'kernel-regression', 'Kernel', 'gaussian', 'Epsilon', 0.3));
fitter.addModel('KNN-5-Distance',      ScatteredInterpolant('Method', 'knn', 'K', 5, 'Weights', 'distance'));
fitter.addModel('NaturalNeighbour',   ScatteredInterpolant('Method', 'natural', 'SubMethod', 'natural'));

% 4. Fit all models and evaluate on grid
fitter.fitAll(X_train, Y_train);
fitter.evaluateAll(f_true, 'Grid', true);

% 5. Print RMSE, MAE, MaxErr, R^2 summary table
fitter.printSummary();
```

---

## Recipe 2: Tune RBF Shape Parameter Epsilon via K-Fold Cross-Validation

Automatically select the optimal scale parameter `c` (Epsilon) for a Gaussian RBF model without using test data.

```matlab
% 1. Load or generate scattered training points
gen = OceanData.Generators.SmoothFunctionGenerator('N', 60, 'Dim', 2, 'Seed', 123);
[X_train, Y_train] = gen.generate();

% 2. Initialize cross-validation tuner for RBF Epsilon
cv = OceanStats.Interpolation.InterpolationCV(...
    'ModelType',  'rbf', ...
    'TuneParam',  'rbf-epsilon', ...
    'ParamGrid',  logspace(-2, 1, 20), ...
    'KFolds',     5, ...
    'KernelName', 'gaussian', ...
    'Verbose',    true);

% 3. Run cross-validation sweep
cv.fit(X_train, Y_train);

% 4. Retrieve best tuned model and parameter
fprintf('Optimal Epsilon: %.4f\n', cv.BestParam);
best_model = cv.BestModel;

% 5. Predict on new query points
X_query = rand(100, 2);
Y_pred  = best_model.predict(X_query);
```

---

## Recipe 3: Render OceanViz Dark-Theme 3D Surface & Error Dashboard

Visualize ground truth, fitted surface, pointwise error heatmap, and CV curve in a tiled layout.

```matlab
% 1. Setup data and run CV tuning
gen = OceanData.Generators.SmoothFunctionGenerator('N', 50, 'Dim', 2, 'Seed', 42);
[X_tr, Y_tr] = gen.generate();

cv = OceanStats.Interpolation.InterpolationCV('ModelType', 'rbf', 'TuneParam', 'rbf-epsilon', 'KFolds', 5);
cv.fit(X_tr, Y_tr);

% 2. Pack data struct for OceanViz strategy
data = struct();
data.X_train  = X_tr;
data.Y_train  = Y_tr;
data.model    = cv.BestModel;
data.f_true   = gen.MetaData.f_true;
data.dim      = 2;
data.domain   = [0, 1];
data.ngrid    = 80;
data.CVResult = cv;

% 3. Render dashboard figure
fig = figure('Color', [0.09 0.09 0.12], 'Position', [50, 80, 1400, 420]);
ax_ph = axes(fig);

viz = OceanViz.Core('Theme', 'Dark');
viz.plot(ax_ph, 'ScatterInterp', data, 'Title', 'RBF Gaussian Interpolation Dashboard');
```

---

## Recipe 4: Retrieve Pedagogical Metadata for GUI Display

Access equation LaTeX strings, problem motivations, assumptions, and complexity for learning apps.

```matlab
% Query metadata for ScatteredKernelRegression
meta = OceanMath.Interpolation.ScatteredKernelRegression.getMetadata();

fprintf('Method: %s\n', meta.display_name);
fprintf('Motivation: %s\n', meta.pedagogy.motivation);
fprintf('Nadaraya-Watson Equation (LaTeX): %s\n', meta.equations.estimator.latex);
fprintf('Training Complexity: %s\n', meta.complexity.training);
```
