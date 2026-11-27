% EXP_LANDSAT_SATELLITE  ESL Section 13.3.2 — Landsat Image Scene Classification
%
% Reproduces ESL Figure 13.6:
%   1. Reconstructs the 82x100 Landsat satellite image from
%      overlapping 3x3 neighborhoods (jigsaw puzzle approach).
%   2. Runs 5-nearest-neighbor classification on 36-dimensional
%      feature space (9 pixels x 4 spectral bands).
%   3. Visualizes: 4 spectral band heatmaps + true/predicted land usage.
%
% Pipeline: 8-step Ocean Library ML Pipeline
% Author: Ocean Library   Date: 2026-03-26
% Ref: Hastie et al. (2009), ESL Section 13.3.2, Figure 13.6
%      STATLOG Landsat Satellite Dataset (Michie et al., 1994)
clc; clear; close all;

%% === Step 1: Load Data ===
fprintf('=== Step 1: Load Data ===\n');

dataDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', ...
    'Data', 'statlog+landsat+satellite');

% Load training and test sets
trn = load(fullfile(dataDir, 'sat.trn'));
tst = load(fullfile(dataDir, 'sat.tst'));

X_train = trn(:, 1:36);   y_train = trn(:, 37);
X_test  = tst(:, 1:36);   y_test  = tst(:, 37);

fprintf('  Train: %d samples, Test: %d samples\n', size(X_train,1), size(X_test,1));
fprintf('  Features: %d (9 pixels x 4 bands)\n', size(X_train,2));

classes = unique([y_train; y_test]);
classNames = {'Red Soil', 'Cotton', 'Grey Soil', 'Damp Grey Soil', ...
              'Veg Stubble', 'Mixture', 'V.Damp Grey Soil'};
fprintf('  Classes: %s\n', mat2str(classes'));

%% === Step 2: Reconstruct Spatial Layout (Jigsaw Puzzle) ===
fprintf('\n=== Step 2: Spatial Reconstruction ===\n');

% Combine all data for reconstruction
allData = [trn; tst];
N_all   = size(allData, 1);
X_all   = allData(:, 1:36);
y_all   = allData(:, 37);

fprintf('  Total neighborhoods: %d\n', N_all);
fprintf('  Valid 82x100 positions: %d (80 x 98 = 7840)\n', 80*98);
fprintf('  Missing: %d\n', 80*98 - N_all);

% ---- Attribute layout for 3x3 neighborhood ----
% Attrs  1- 4: top-left pixel     (r-1, c-1)
% Attrs  5- 8: top-middle pixel   (r-1, c  )
% Attrs  9-12: top-right pixel    (r-1, c+1)
% Attrs 13-16: middle-left pixel  (r  , c-1)
% Attrs 17-20: center pixel       (r  , c  )
% Attrs 21-24: middle-right pixel (r  , c+1)
% Attrs 25-28: bottom-left pixel  (r+1, c-1)
% Attrs 29-32: bottom-middle pixel(r+1, c  )
% Attrs 33-36: bottom-right pixel (r+1, c+1)

% ---- Build adjacency via hash matching ----
% Horizontal: Row A's {TM,TR,C,MR,BM,BR} == Row B's {TL,TM,ML,C,BL,BM}
% i.e., A's cols [5:12, 17:24, 29:36] == B's cols [1:8, 13:20, 25:32]

rightSig_cols = [5:12, 17:24, 29:36];   % 24 values
leftSig_cols  = [1:8,  13:20, 25:32];   % 24 values

% Vertical: A's {ML,C,MR,BL,BM,BR} == C's {TL,TM,TR,ML,C,MR}
% i.e., A's cols [13:36] == C's cols [1:24]
bottomSig_cols = 13:36;   % 24 values
topSig_cols    = 1:24;    % 24 values

% Build hash maps for left signatures and top signatures
fprintf('  Building adjacency graph...\n');

% Hash function: convert to string key
makeKey = @(v) sprintf('%d,', v);

% Map: leftSig → row index (for finding right neighbors)
leftMap = containers.Map('KeyType', 'char', 'ValueType', 'int32');
for i = 1:N_all
    key = makeKey(X_all(i, leftSig_cols));
    leftMap(key) = i;
end

% Map: topSig → row index (for finding bottom neighbors)
topMap = containers.Map('KeyType', 'char', 'ValueType', 'int32');
for i = 1:N_all
    key = makeKey(X_all(i, topSig_cols));
    topMap(key) = i;
end

% Find right neighbor for each row
rightNeighbor = zeros(N_all, 1);
for i = 1:N_all
    key = makeKey(X_all(i, rightSig_cols));
    if leftMap.isKey(key)
        rightNeighbor(i) = leftMap(key);
    end
end

% Find bottom neighbor for each row
bottomNeighbor = zeros(N_all, 1);
for i = 1:N_all
    key = makeKey(X_all(i, bottomSig_cols));
    if topMap.isKey(key)
        bottomNeighbor(i) = topMap(key);
    end
end

nHoriz = sum(rightNeighbor > 0);
nVert  = sum(bottomNeighbor > 0);
fprintf('  Horizontal adjacencies found: %d\n', nHoriz);
fprintf('  Vertical adjacencies found: %d\n', nVert);

% ---- Build reverse links: left and top neighbors ----
leftNeighbor = zeros(N_all, 1);
topNeighbor  = zeros(N_all, 1);
for i = 1:N_all
    if rightNeighbor(i) > 0
        leftNeighbor(rightNeighbor(i)) = i;
    end
    if bottomNeighbor(i) > 0
        topNeighbor(bottomNeighbor(i)) = i;
    end
end

% ---- BFS flood-fill to assign (row, col) positions ----
% Start from pixel 1, assign it position (0, 0), then expand in all 4 dirs
fprintf('  Running BFS flood-fill...\n');

posR = NaN(N_all, 1);   % row position for each data index
posC = NaN(N_all, 1);   % col position

% BFS queue
queue = zeros(N_all, 1);
qHead = 1; qTail = 0;

% Seed from index 1
startIdx = 1;
posR(startIdx) = 0;
posC(startIdx) = 0;
qTail = qTail + 1;
queue(qTail) = startIdx;

while qHead <= qTail
    cur = queue(qHead);
    qHead = qHead + 1;
    cr = posR(cur);
    cc = posC(cur);
    
    % Explore 4 neighbors
    neighbors = [rightNeighbor(cur), leftNeighbor(cur), ...
                 bottomNeighbor(cur), topNeighbor(cur)];
    dr = [0, 0, 1, -1];
    dc = [1, -1, 0, 0];
    
    for ni = 1:4
        nb = neighbors(ni);
        if nb > 0 && isnan(posR(nb))
            posR(nb) = cr + dr(ni);
            posC(nb) = cc + dc(ni);
            qTail = qTail + 1;
            queue(qTail) = nb;
        end
    end
end

nPlaced = sum(~isnan(posR));
fprintf('  BFS placed: %d / %d (%.1f%%)\n', nPlaced, N_all, 100*nPlaced/N_all);

% Shift positions so (min_r, min_c) = (1, 1)
placed = find(~isnan(posR));
minR = min(posR(placed));
minC = min(posC(placed));
posR(placed) = posR(placed) - minR + 1;
posC(placed) = posC(placed) - minC + 1;

gridRows = max(posR(placed));
gridCols = max(posC(placed));
fprintf('  Grid dimensions: %d x %d\n', gridRows, gridCols);

% Build grid index array
gridIdx = zeros(gridRows, gridCols);
for i = 1:numel(placed)
    idx = placed(i);
    gridIdx(posR(idx), posC(idx)) = idx;
end

nInGrid = sum(gridIdx(:) > 0);
fprintf('  Pixels in grid: %d / %d (%.1f%%)\n', nInGrid, N_all, 100*nInGrid/N_all);

% ---- Build image arrays directly from BFS grid ----
% The BFS grid IS the spatial layout; gridRows/gridCols ≈ 80x98
spectral = NaN(gridRows, gridCols, 4);
labels   = NaN(gridRows, gridCols);

for r = 1:gridRows
    for c = 1:gridCols
        idx = gridIdx(r, c);
        if idx > 0
            spectral(r, c, :) = X_all(idx, 17:20);
            labels(r, c)      = y_all(idx);
        end
    end
end

% Also fill edge pixels from neighbor values where possible
for r = 1:gridRows
    for c = 1:gridCols
        idx = gridIdx(r, c);
        if idx == 0, continue; end
        
        % Top-left of neighborhood → (r-1, c-1)
        if r > 1 && c > 1 && isnan(spectral(r-1,c-1,1))
            spectral(r-1, c-1, :) = X_all(idx, 1:4);
        end
        % Top-middle → (r-1, c)
        if r > 1 && isnan(spectral(r-1,c,1))
            spectral(r-1, c, :) = X_all(idx, 5:8);
        end
        % Top-right → (r-1, c+1)
        if r > 1 && c < gridCols && isnan(spectral(r-1,c+1,1))
            spectral(r-1, c+1, :) = X_all(idx, 9:12);
        end
        % Middle-left → (r, c-1)
        if c > 1 && isnan(spectral(r,c-1,1))
            spectral(r, c-1, :) = X_all(idx, 13:16);
        end
        % Middle-right → (r, c+1)
        if c < gridCols && isnan(spectral(r,c+1,1))
            spectral(r, c+1, :) = X_all(idx, 21:24);
        end
        % Bottom-left → (r+1, c-1)
        if r < gridRows && c > 1 && isnan(spectral(r+1,c-1,1))
            spectral(r+1, c-1, :) = X_all(idx, 25:28);
        end
        % Bottom-middle → (r+1, c)
        if r < gridRows && isnan(spectral(r+1,c,1))
            spectral(r+1, c, :) = X_all(idx, 29:32);
        end
        % Bottom-right → (r+1, c+1)
        if r < gridRows && c < gridCols && isnan(spectral(r+1,c+1,1))
            spectral(r+1, c+1, :) = X_all(idx, 33:36);
        end
    end
end

nFilled = sum(~isnan(spectral(:,:,1)), 'all');
fprintf('  Spectral pixels filled: %d / %d (%.1f%%)\n', ...
    nFilled, gridRows*gridCols, 100*nFilled/(gridRows*gridCols));

%% === Step 3-6: [SKIP] No feature engineering needed ===
fprintf('\n[Steps 3-6] [SKIP] Raw 36-dim features retained.\n');

%% === Step 7: 5-NN Classification (MATLAB built-in) ===
fprintf('\n=== Step 7: 5-NN Classification (fitcknn) ===\n');

knn5 = fitcknn(X_train, y_train, 'NumNeighbors', 5, 'Distance', 'euclidean');
y_pred = predict(knn5, X_test);

testErr = mean(y_pred ~= y_test);
fprintf('  5-NN Test Error: %.3f (%.1f%%)\n', testErr, testErr*100);
fprintf('  ESL Expected:    ~0.095 (9.5%%)\n');

% Per-class accuracy
fprintf('\n  Per-class accuracy:\n');
for i = 1:numel(classes)
    cl = classes(i);
    mask = y_test == cl;
    acc = mean(y_pred(mask) == cl);
    fprintf('    Class %d (%-20s): %.1f%% (%d samples)\n', ...
        cl, classNames{cl}, acc*100, sum(mask));
end

% Confusion matrix
fprintf('\n  Confusion Matrix (rows=true, cols=pred):\n');
nC = numel(classes);
confMat = zeros(nC);
for i = 1:nC
    for j = 1:nC
        confMat(i,j) = sum(y_test == classes(i) & y_pred == classes(j));
    end
end
fprintf('        ');
for j = 1:nC, fprintf('  C%d  ', classes(j)); end
fprintf('\n');
for i = 1:nC
    fprintf('  C%d  ', classes(i));
    for j = 1:nC
        fprintf('%5d ', confMat(i,j));
    end
    fprintf('\n');
end

%% === Step 8: Visualization (ESL Figure 13.6) ===
fprintf('\n=== Step 8: Visualization ===\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('ESL Figure 13.6 — Landsat Satellite Image Classification');
fig.Position = [30, 30, 1200, 800];

% Panel positions: 2 rows x 3 columns
pw = 0.28; ph = 0.40;
gx = 0.04; gy = 0.06;
positions = {
    [gx + 0*(pw+gx), 0.52, pw, ph],     % top-left: Band 1
    [gx + 1*(pw+gx), 0.52, pw, ph],     % top-mid:  Band 2
    [gx + 2*(pw+gx), 0.52, pw, ph],     % top-right: Band 3
    [gx + 0*(pw+gx), 0.06, pw, ph],     % bot-left: Band 4
    [gx + 1*(pw+gx), 0.06, pw, ph],     % bot-mid:  Land Usage (true)
    [gx + 2*(pw+gx), 0.06, pw, ph],     % bot-right: Land Usage (predicted)
};

bandTitles = {'Spectral Band 1', 'Spectral Band 2', ...
              'Spectral Band 3', 'Spectral Band 4'};

% ---- Custom blue-magenta-yellow colormap (matching ESL book) ----
n = 256;
cmap_esl = zeros(n, 3);
for i = 1:n
    t = (i - 1) / (n - 1);
    if t < 0.33
        % Blue → Magenta
        s = t / 0.33;
        cmap_esl(i,:) = [s, 0, 1-0.3*s];
    elseif t < 0.66
        % Magenta → Pink/Red
        s = (t - 0.33) / 0.33;
        cmap_esl(i,:) = [1, 0.2*s, 0.3*(1-s)];
    else
        % Pink/Red → Yellow
        s = (t - 0.66) / 0.34;
        cmap_esl(i,:) = [1, 0.2 + 0.8*s, 0];
    end
end

% ---- Panels 1-4: Spectral band heatmaps ----
for b = 1:4
    ax = axes('Parent', fig, 'Position', positions{b});
    
    bandImg = spectral(:, :, b);
    % Fill NaN with nearest-neighbor interpolation for clean display
    nanMask = isnan(bandImg);
    if any(nanMask(:))
        [yr, xc] = find(~nanMask);
        vals = bandImg(~nanMask);
        [ynr, xnc] = find(nanMask);
        if ~isempty(yr) && ~isempty(ynr)
            F = scatteredInterpolant(double(xc), double(yr), vals, 'nearest', 'nearest');
            bandImg(nanMask) = F(double(xnc), double(ynr));
        end
    end
    
    imagesc(ax, bandImg);
    colormap(ax, cmap_esl);
    axis(ax, 'image');
    set(ax, 'XTick', [], 'YTick', []);
    title(ax, bandTitles{b}, 'Color', viz.Colors.Text, ...
        'FontSize', 11, 'FontWeight', 'bold');
end

% ---- Class colormap (matching ESL Figure 13.6 appearance) ----
classColors = [
    0.85  0.10  0.10;   % 1: red soil      → red
    1.00  0.85  0.00;   % 2: cotton         → yellow
    0.60  0.60  0.60;   % 3: grey soil      → grey
    1.00  0.50  0.00;   % 4: damp grey soil → orange
    0.00  0.80  0.90;   % 5: veg stubble    → cyan
    0.30  0.70  0.15;   % 6: mixture (N/A)  → green
    0.40  0.25  0.65;   % 7: v.damp grey    → purple
];

% ---- Fill missing label pixels with nearest-neighbor ----
labelsFilled = labels;
nanLabelMask = isnan(labels);
if any(nanLabelMask(:))
    [yr, xc] = find(~nanLabelMask);
    vals = labels(~nanLabelMask);
    [ynr, xnc] = find(nanLabelMask);
    if ~isempty(yr) && ~isempty(ynr)
        F = scatteredInterpolant(double(xc), double(yr), vals, 'nearest', 'nearest');
        labelsFilled(nanLabelMask) = round(F(double(xnc), double(ynr)));
    end
end

% Build label RGB image
labelRGB = zeros(gridRows, gridCols, 3);
for r = 1:gridRows
    for c = 1:gridCols
        cl = labelsFilled(r, c);
        if ~isnan(cl) && cl >= 1 && cl <= 7
            labelRGB(r, c, :) = classColors(cl, :);
        end
    end
end

% ---- Panel 5: True Land Usage ----
ax5 = axes('Parent', fig, 'Position', positions{5});
image(ax5, labelRGB);
axis(ax5, 'image');
set(ax5, 'XTick', [], 'YTick', []);
title(ax5, 'Land Usage', 'Color', viz.Colors.Text, ...
    'FontSize', 11, 'FontWeight', 'bold');

% ---- Panel 6: Predicted Land Usage (5-NN) ----
fprintf('  Classifying all placed pixels with 5-NN...\n');

placedMask = gridIdx > 0;
placedIdxs = gridIdx(placedMask);
X_placed   = X_all(placedIdxs, :);
y_placed_pred = predict(knn5, X_placed);

% Scatter back + fill missing
predLabels = NaN(gridRows, gridCols);
predLabels(placedMask) = y_placed_pred;

predFilled = predLabels;
nanPredMask = isnan(predLabels);
if any(nanPredMask(:))
    [yr, xc] = find(~nanPredMask);
    vals = predLabels(~nanPredMask);
    [ynr, xnc] = find(nanPredMask);
    if ~isempty(yr) && ~isempty(ynr)
        F = scatteredInterpolant(double(xc), double(yr), vals, 'nearest', 'nearest');
        predFilled(nanPredMask) = round(F(double(xnc), double(ynr)));
    end
end

predRGB = zeros(gridRows, gridCols, 3);
for r = 1:gridRows
    for c = 1:gridCols
        cl = predFilled(r, c);
        if ~isnan(cl) && cl >= 1 && cl <= 7
            predRGB(r, c, :) = classColors(cl, :);
        end
    end
end

ax6 = axes('Parent', fig, 'Position', positions{6});
image(ax6, predRGB);
axis(ax6, 'image');
set(ax6, 'XTick', [], 'YTick', []);
title(ax6, 'Predicted Land Usage', 'Color', viz.Colors.Text, ...
    'FontSize', 11, 'FontWeight', 'bold');

% Save Figure 13.6
resultsDir = fullfile(fileparts(mfilename('fullpath')), 'Results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
saveas(fig, fullfile(resultsDir, 'fig13_6_landsat_satellite.png'));
fprintf('[Step 8] Saved: fig13_6_landsat_satellite.png\n');

%% === Figure 2: Confusion Matrix Heatmap ===
fprintf('  Generating confusion matrix heatmap...\n');

fig2 = viz.createFigure('Landsat — Confusion Matrix (5-NN)');
fig2.Position = [50, 50, 700, 600];
ax_cm = axes('Parent', fig2);

confMatNorm = confMat ./ sum(confMat, 2);  % row-normalized
imagesc(ax_cm, confMatNorm);
colormap(ax_cm, flipud(hot(256)));
cb = colorbar(ax_cm);
cb.Color = viz.Colors.Text;
cb.Label.String = 'Proportion';
cb.Label.Color = viz.Colors.Text;

classLabelsShort = {'Red Soil', 'Cotton', 'Grey Soil', 'Damp Grey', 'Veg Stub', 'V.Damp Grey'};
set(ax_cm, 'XTick', 1:nC, 'XTickLabel', classLabelsShort, ...
    'YTick', 1:nC, 'YTickLabel', classLabelsShort, ...
    'XTickLabelRotation', 45, ...
    'XColor', viz.Colors.Text, 'YColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
xlabel(ax_cm, 'Predicted', 'Color', viz.Colors.Text, 'FontSize', 11);
ylabel(ax_cm, 'True', 'Color', viz.Colors.Text, 'FontSize', 11);
title(ax_cm, 'Confusion Matrix — 5-NN (row-normalized)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');

% Overlay counts
for i = 1:nC
    for j = 1:nC
        v = confMat(i,j);
        if v > 0
            textCol = [1 1 1];
            if confMatNorm(i,j) > 0.5, textCol = [0 0 0]; end
            text(ax_cm, j, i, sprintf('%d', v), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, ...
                'FontWeight', 'bold', 'Color', textCol);
        end
    end
end

saveas(fig2, fullfile(resultsDir, 'fig_landsat_confusion.png'));
fprintf('[Step 8] Saved: fig_landsat_confusion.png\n');

%% === Figure 3: Class Spectral Profiles ===
fprintf('  Generating spectral profiles...\n');

fig3 = viz.createFigure('Landsat — Spectral Profiles by Class');
fig3.Position = [80, 80, 900, 500];
ax_sp = axes('Parent', fig3);
hold(ax_sp, 'on'); viz.styleAxes(ax_sp);

% Center pixel is attrs 17-20 (4 spectral bands)
X_center = X_train(:, 17:20);  % only center pixel bands
bandNames = {'Green', 'Red', 'NIR-1', 'NIR-2'};

activeClasses = classes;  % [1,2,3,4,5,7]
activeNames = classNames(activeClasses);
activeColors = classColors(activeClasses, :);

for ci = 1:numel(activeClasses)
    cl = activeClasses(ci);
    X_cl = X_center(y_train == cl, :);
    mu = mean(X_cl, 1);
    se = std(X_cl, 0, 1) / sqrt(size(X_cl, 1));
    
    errorbar(ax_sp, 1:4, mu, se, '-o', ...
        'Color', activeColors(ci,:), 'MarkerFaceColor', activeColors(ci,:), ...
        'MarkerSize', 5, 'LineWidth', 1.8, 'CapSize', 4, ...
        'DisplayName', activeNames{ci});
end

set(ax_sp, 'XTick', 1:4, 'XTickLabel', bandNames);
xlabel(ax_sp, 'Spectral Band', 'Color', viz.Colors.Text, 'FontSize', 11);
ylabel(ax_sp, 'Mean Pixel Intensity', 'Color', viz.Colors.Text, 'FontSize', 11);
title(ax_sp, 'Spectral Profiles by Land-Use Class', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
legend(ax_sp, 'Location', 'northwest', 'FontSize', 9, ...
    'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
hold(ax_sp, 'off');

saveas(fig3, fullfile(resultsDir, 'fig_landsat_spectral_profiles.png'));
fprintf('[Step 8] Saved: fig_landsat_spectral_profiles.png\n');

%% === Figure 4: STATLOG Results Comparison (ESL Figure 13.8) ===
fprintf('  Generating STATLOG comparison chart...\n');

% Published error rates from ESL Figure 13.8 / STATLOG project
methods   = {'DANN', 'K-NN', 'LVQ', 'RBF', 'ALLOC80', 'CART', ...
             'Neural', 'NewID', 'C4.5', 'QDA', 'SMART', 'Logistic', 'LDA'};
errorRates = [0.045, 0.085, 0.097, 0.107, 0.112, 0.118, ...
              0.125, 0.132, 0.134, 0.143, 0.148, 0.153, 0.162];

fig4 = viz.createFigure('ESL Figure 13.8 — STATLOG Results');
fig4.Position = [110, 110, 900, 550];
ax_st = axes('Parent', fig4);
hold(ax_st, 'on'); viz.styleAxes(ax_st);

% Our result
ourErr = testErr;

% Color: NN-based methods in cyan, tree in orange, others in grey
for i = 1:numel(methods)
    if ismember(methods{i}, {'DANN','K-NN','LVQ'})
        c = [0.00 0.80 0.90];  % cyan for NN
    elseif ismember(methods{i}, {'CART','NewID','C4.5'})
        c = [1.00 0.55 0.00];  % orange for trees
    else
        c = [0.55 0.55 0.55];  % grey for others
    end
    scatter(ax_st, i, errorRates(i), 60, c, 'filled', ...
        'HandleVisibility', 'off');
    text(ax_st, i, errorRates(i) + 0.004, methods{i}, ...
        'Color', c, 'FontSize', 9, 'HorizontalAlignment', 'center', ...
        'FontWeight', 'bold');
end

% Mark our 5-NN result
scatter(ax_st, 2, ourErr, 120, [0.85 0.10 0.10], 'd', 'filled', ...
    'DisplayName', sprintf('Our 5-NN (%.1f%%)', ourErr*100));

set(ax_st, 'XTick', 1:numel(methods), 'XTickLabel', []);
xlabel(ax_st, 'Method', 'Color', viz.Colors.Text, 'FontSize', 11);
ylabel(ax_st, 'Test Misclassification Error', 'Color', viz.Colors.Text, 'FontSize', 11);
title(ax_st, 'STATLOG Results (ESL Figure 13.8)', ...
    'Color', viz.Colors.Text, 'FontSize', 13, 'FontWeight', 'bold');
ylim(ax_st, [0.0 0.2]);
legend(ax_st, 'Location', 'northwest', 'FontSize', 10, ...
    'EdgeColor', [0.3 0.3 0.3], 'TextColor', viz.Colors.Text, ...
    'Color', viz.Colors.Background);
hold(ax_st, 'off');

saveas(fig4, fullfile(resultsDir, 'fig13_8_statlog_results.png'));
fprintf('[Step 8] Saved: fig13_8_statlog_results.png\n');

fprintf('\n=== Done ===\n');
