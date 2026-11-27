import OceanStats.Supervised.PRIM.PRIM;
import OceanStats.Supervised.PRIM.Criterion.TargetClass;
import OceanStats.Supervised.PRIM.Selection.CVSelector;
import OceanStats.Selection.CrossValidation.KFold;

% --- Data Loading ---
% Standardize pathing for reliability
currentDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(currentDir, 'Data', 'spambase');
for i = 1:3
    if isfolder(dataDir), break; end
    currentDir = fileparts(currentDir);
    dataDir = fullfile(currentDir, 'Data', 'spambase');
end

dataFile = fullfile(dataDir, 'spambase.data');
if ~isfile(dataFile), error('Spambase data not found.'); end

raw = load(dataFile);
X = raw(:, 1:end-1);
Y = raw(:, end);

namesFile = fullfile(dataDir, 'spambase.names');
fileID = fopen(namesFile, 'r');
namesRaw = textscan(fileID, '%s %*s', 'Delimiter', ':', 'CommentStyle', '|', 'HeaderLines', 32);
fclose(fileID);
featNames = namesRaw{1};
featNames = featNames(end-56:end); 

% Standard Split 70/30 (Matches robustness goal)
rng(1);
cvObj = cvpartition(Y, 'HoldOut', 0.3);
X_train = X(training(cvObj), :);
Y_train = Y(training(cvObj));
X_test  = X(test(cvObj), :);
Y_test  = Y(test(cvObj));

fprintf('Train Size: %d, Test Size: %d\n', length(Y_train), length(Y_test));
fprintf('Global Mean (Train): %.4f\n\n', mean(Y_train));

% --- Configure PRIM ---
% Standard parameters: 5% MinBoxSize, Alpha 0.05, 5 Boxes
minSize = floor(0.05 * length(Y_train));

prim = PRIM('Alpha', 0.05, 'MinBoxSize', minSize, 'MaxBoxes', 5);
% To force textbook-style exploratory continuous traits, uncomment below:
% prim.PeelingStrategy = 'Patient';

prim.ObjectiveFunction = TargetClass(1);

% Standard K-Fold Cross Validation for box selection
prim.BoxSelector = CVSelector(KFold(5), prim);

fprintf('Training PRIM...\n');
prim.fit(X_train, Y_train);

% --- Visualization of Rule 1 (2D Projection) ---
% Find indices for 'remove' and 'money'
idx_remove = find(contains(featNames, 'word_freq_remove'), 1);
idx_money = find(contains(featNames, 'word_freq_money'), 1);

if ~isempty(prim.Boxes) && ~isempty(idx_remove) && ~isempty(idx_money)
    box1 = prim.Boxes{1};
    
    % Initialize the Visualization Engine for professional rendering
    import OceanViz.Core;
    viz = Core('Theme', 'Dark');
    
    fig = viz.createFigure('PRIM Rule 1 Projection');
    ax = axes(fig);
    hold(ax, 'on');
    
    % Plot non-spam (class 0) using VizEngine Secondary Color
    idx0 = Y_train == 0;
    scatter(ax, X_train(idx0, idx_remove), X_train(idx0, idx_money), ...
        20, viz.Colors.Secondary(:)', 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none', 'DisplayName', 'Non-Spam (0)');
    
    % Plot spam (class 1) using VizEngine Primary Color
    idx1 = Y_train == 1;
    scatter(ax, X_train(idx1, idx_remove), X_train(idx1, idx_money), ...
        20, viz.Colors.Primary(:)', 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none', 'DisplayName', 'Spam (1)');
    
    % --- EXPLANATION: How this Bounding Box is Produced ---
    % PRIM defines a high-dimensional hyperrectangle. For a 2D projection,
    % we extract the minimum and maximum boundaries for the two specific features.
    % If a boundary is 'Inf' or '-Inf', it means PRIM did not bound that side,
    % so we pad it to the edge of the visible data.
    
    xMin = box1.Bounds(idx_remove, 1);
    if isinf(xMin), xMin = 0; end
    xMax = box1.Bounds(idx_remove, 2);
    if isinf(xMax), xMax = max(X_train(:, idx_remove)) * 1.1; end
    
    yMin = box1.Bounds(idx_money, 1);
    if isinf(yMin), yMin = 0; end
    yMax = box1.Bounds(idx_money, 2);
    if isinf(yMax), yMax = max(X_train(:, idx_money)) * 1.1; end
    
    % Draw the Bounding Box Outline using the Theme's Boundary color
    rectangle(ax, 'Position', [xMin, yMin, xMax-xMin, yMax-yMin], ...
        'EdgeColor', viz.Colors.Boundary(:)', 'LineWidth', 2, 'LineStyle', '--');
    
    % Fill the inside of the box with a subtle transparent highlight
    patch(ax, [xMin xMax xMax xMin], [yMin yMin yMax yMax], ...
        viz.Colors.Boundary(:)', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    
    xlabel(ax, 'word\_freq\_remove', 'Interpreter', 'latex');
    ylabel(ax, 'word\_freq\_money', 'Interpreter', 'latex');
    title(ax, sprintf('Rule 1: Purity %.1f\\%%', box1.Score * 100), 'Interpreter', 'latex');
    
    legend(ax, 'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
    
    % Define explicit axes limit using percentiles to prevent extreme outliers from compressing the view
    xMaxDisp = prctile(X_train(:, idx_remove), 99.5);
    yMaxDisp = prctile(X_train(:, idx_money), 99.5);
    
    if xMaxDisp <= 0.01; xMaxDisp = 1.0; end
    if yMaxDisp <= 0.01; yMaxDisp = 1.0; end
    
    xlim(ax, [-0.05, xMaxDisp * 1.5]);
    ylim(ax, [-0.05, yMaxDisp * 1.5]);
    
    % Apply the VizEngine unifying style (background, grid, axes colors)
    viz.styleAxes(ax);
    saveas(fig, fullfile(currentDir, 'rule1_box.png'));
end
seq = prim.growBoxSequence(X_train, Y_train);
supports = cellfun(@(b) b.NSamples / length(Y_train), seq);
purities = cellfun(@(b) b.Score, seq);

% figure('Color', 'w');
% plot(supports, purities, 'o-', 'LineWidth', 2);
% set(gca, 'XDir', 'reverse');
% xlabel('Support'); ylabel('Purity'); title('Peeling Sequence - Rule 1');
% grid on;
% % Mark the benchmark point
% hold on; plot(0.1454, 0.9746, 'r*', 'MarkerSize', 10); 
% text(0.1454, 0.9746, ' Benchmark', 'Color', 'r');
% saveas(gcf, 'peeling_sequence.png');
fprintf('\nPRIM Results (Hunting for Spam=1)\n');
fprintf('--------------------------------------------------------------------------------\n');

globalMin = min(X_train);
globalMax = max(X_train);

% Track remaining data for sequential rule eval
maskRemainingTrain = true(length(Y_train), 1);
maskRemainingTest = true(length(Y_test), 1);

for i = 1:length(prim.Boxes)
    box = prim.Boxes{i};
    
    currentTrainPurity = mean(Y_train(maskRemainingTrain));
    
    maskInBoxTrain = box.isInside(X_train(maskRemainingTrain, :));
    Y_box_train = Y_train(maskRemainingTrain);
    Y_box_train = Y_box_train(maskInBoxTrain);
    
    boxPurityTrain = mean(Y_box_train);
    supportTrain = sum(maskInBoxTrain) / length(Y_train);
    
    % Test stats
    maskInBoxTest = box.isInside(X_test(maskRemainingTest, :));
    Y_box_test = Y_test(maskRemainingTest);
    Y_box_test = Y_box_test(maskInBoxTest);
    boxPurityTest = mean(Y_box_test);
    
    fprintf('Rule %d\n', i);
    fprintf('  Train: Purity %.4f, Support %.4f (Global: %.4f)\n', ...
        boxPurityTrain, supportTrain, currentTrainPurity);
    fprintf('  Test:  Purity %.4f\n', boxPurityTest);
    
    printBoxRules(box, globalMin, globalMax, featNames);
    
    % Update remaining for next box
    maskSweepTrain = box.isInside(X_train);
    maskRemainingTrain = maskRemainingTrain & ~maskSweepTrain;
    
    maskSweepTest = box.isInside(X_test);
    maskRemainingTest = maskRemainingTest & ~maskSweepTest;
    
    fprintf('--------------------------------------------------------------------------------\n');
end


function printBoxRules(box, gMin, gMax, featNames)
    D = size(box.Bounds, 1);
    for j = 1:D
        name = featNames{j};
        name = strrep(name, 'word_freq_', '');
        name = strrep(name, 'char_freq_', 'ch');
        
        if box.Bounds(j, 1) > gMin(j) + 1e-9
            fprintf('    { %10s > %6.3f }\n', name, box.Bounds(j, 1));
        end
        if box.Bounds(j, 2) < gMax(j) - 1e-9
            fprintf('    { %10s < %6.3f }\n', name, box.Bounds(j, 2));
        end
    end
end