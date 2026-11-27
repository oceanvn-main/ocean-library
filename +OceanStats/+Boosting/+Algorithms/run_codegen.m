% RUN_CODEGEN  Compile predict_codegen.m → C MEX library using MATLAB Coder.
%
%   PREREQUISITES:
%     - MATLAB Coder license/toolbox
%     - Run this script after training a GBM and calling exportForCodegen()
%
%   WHAT IT DOES:
%     1. Trains a small demo GBM on sine data.
%     2. Exports the flat model struct (no handle classes, just doubles).
%     3. Runs codegen to compile predict_codegen.m to a C MEX file.
%     4. Verifies the MEX output matches the object-based output.
%
%   OUTPUT:
%     codegen/mex/predict_codegen/predict_codegen_mex.mexw64  (Windows)
%     or equivalent .so on Linux/Mac.
%
%   USAGE AFTER COMPILATION:
%     Y_hat = predict_codegen_mex(X, flat);  % ← no MATLAB interpreter needed

clc;
addpath('c:\Ocean Library\Matlab code practice');

% =========================================================================
% STEP 1: Train a GBM
% =========================================================================
rng(42);
N = 300;
X = linspace(0, 4*pi, N)';
Y = sin(X) + 0.3 * randn(N, 1);

loss = OceanStats.Loss.Regression.HuberLoss('Delta', 1.0);
gbm  = OceanStats.Boosting.Algorithms.GradientBoostingMachine(...
    'Loss', loss, 'NumTrees', 50, 'LearningRate', 0.15, 'MaxDepth', 3);
gbm.fit(X, Y);
fprintf('GBM trained. Final loss: %.5f\n', gbm.trainLoss()(end));

% =========================================================================
% STEP 2: Export to flat struct
% =========================================================================
flat = gbm.exportForCodegen();
fprintf('Flat model: %d total nodes across %d trees.\n', ...
    numel(flat.FeatureIdx), flat.NumTrees);

% =========================================================================
% STEP 3: Verify predict_codegen matches the object predict
% =========================================================================
Y_obj    = gbm.predict(X);
Y_flat   = predict_codegen(X, flat);
maxError = max(abs(Y_obj - Y_flat));
fprintf('Max prediction difference (object vs flat): %.2e\n', maxError);
assert(maxError < 1e-10, 'Flat representation does not match object predict!');
fprintf('Verification PASSED.\n\n');

% =========================================================================
% STEP 4: Run codegen
% =========================================================================
fprintf('Compiling to MEX/C ...\n');

% Define input types (fixed size = size at export time)
X_type    = coder.typeof(double(0), [N, 1],       [true, false]);  % variable rows, 1 col
flat_type = coder.typeof(flat);   % MATLAB Coder infers struct layout automatically

cfg         = coder.config('mex');
cfg.GenerateReport = true;

codegen predict_codegen ...
    -config cfg ...
    -args {X_type, flat_type} ...
    -o codegen/mex/predict_codegen/predict_codegen_mex

fprintf('Compilation complete.\n');
fprintf('MEX file: codegen/mex/predict_codegen/predict_codegen_mex.mexw64\n');

% =========================================================================
% STEP 5: Test the compiled MEX
% =========================================================================
Y_mex      = predict_codegen_mex(X, flat);
mexError   = max(abs(Y_obj - Y_mex));
fprintf('Max MEX vs object difference: %.2e\n', mexError);
fprintf('MEX verification %s\n', ternary(mexError < 1e-10, 'PASSED', 'FAILED'));

% =========================================================================
% Helper
% =========================================================================
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
