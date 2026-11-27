% EXP_GMM_NCI
% Tests three mathematical workarounds for deploying Gaussian Mixture Models
% on high-dimensional p >> N microarray datasets where raw Maximum Likelihood
% estimation catastrophically fails.
%
% Ref: System ML Solutions: How to Deploy GMMs in p >> N

clc; clear; rng(42, 'twister');
fprintf('=== GMM High-Dimensional Scaling Experiment ===\n');

%% PHASE 1: Data Ingestion
dataPath = fullfile('c:', 'Ocean Library', 'Matlab code practice', 'Data', 'NCI');
dataFile = fullfile(dataPath, 'nci.data.csv');
lblFile  = fullfile(dataPath, 'nci.label.txt');

trueLabels = strtrim(string(readcell(lblFile)));
trueLabels = strrep(trueLabels, 'K562A-repro', 'K562');
trueLabels = strrep(trueLabels, 'K562B-repro', 'K562');
trueLabels = strrep(trueLabels, 'MCF7A-repro', 'MCF7');
trueLabels = strrep(trueLabels, 'MCF7D-repro', 'MCF7');
uniqueLabels = unique(trueLabels);
L = numel(uniqueLabels);

T = readtable(dataFile, 'ReadVariableNames', true, 'ReadRowNames', true, 'VariableNamingRule', 'preserve');
X = table2array(T)'; 

fprintf('[DataOps] Ingested %d samples across %d genes.\n', size(X, 1), size(X, 2));

%% PHASE 2: Dimensionality Extraction
% We use 10 components for Option A to ensure p < N so the GMM converges cleanly
pcaPipe = OceanStats.Pipeline.PCA(10); 
pcaModel = pcaPipe.train(X, []);
X_pca_10 = pcaPipe.predict(pcaModel, X);
X_pca_2  = X_pca_10(:, 1:2); % Used exclusively for 2D Visual mappings

%% PHASE 3: The 3 Analytical Interventions
% Reverting to the custom explicit OceanStats engine because MATLAB fitgmdist 
% brutally blocks any p > N execution natively avoiding matrix math checks.
K = 3;

experiments = {
    struct('Name', 'Option A: PCA Pre-comp', ...
           'Cov', 'full', 'Reg', 1e-6, 'DataIdx', 1, 'MaxIter', 50),
    struct('Name', 'Option B: Diagonal Cov Constraints', ...
           'Cov', 'diagonal', 'Reg', 1e-6, 'DataIdx', 2, 'MaxIter', 50),
    struct('Name', 'Option C: Diagonal Loading (Shrinkage)', ...
           'Cov', 'full', 'Reg', 1e-2, 'DataIdx', 2, 'MaxIter', 15)
};
% Option C capped severely due to extreme memory loading.

%% PHASE 4: Execution & Evaluation
viz = OceanViz.Core('Theme', 'Dark');
f_scatter = viz.createFigure('GMM High-Dim Comparison Maps');

for i = 1:numel(experiments)
    exp = experiments{i};
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', exp.Name);
    fprintf('=================================================================\n');
    
    if exp.DataIdx == 1, inputData = X_pca_10; else, inputData = X; end
    
    fprintf('  Training OceanStats GMM Engine... (Option C will take >30s)\n');
    
    % Initialize structural model framework
    model = OceanStats.Unsupervised.GMM(K, 'CovarianceType', exp.Cov, 'Regularization', exp.Reg);
    
    % HACK: Quick mathematical initialization. 
    % We use K-means briefly to seed the Mu, giving GMM a strong topological start!
    km = OceanStats.Unsupervised.KMeans();
    km.fit(inputData, K, 'NumRestarts', 3);
    model.Mu = km.Centroids;
    model.Pi = ones(1, K) / K;
    [~, D_] = size(inputData);
    sigma = zeros(D_, D_, K);
    for k=1:K, sigma(:,:,k) = cov(inputData(km.Labels==k, :)) + eye(D_)*exp.Reg; end
    if exp.Cov == "diagonal", for k=1:K, sigma(:,:,k) = diag(diag(sigma(:,:,k))); end; end
    model.Sigma = sigma;

    % Train Engine explicitly via EM Recipe Wrapper
    recipe = OceanStats.Inference.Recipes.Recipe_GMM_EM(model, 1e-4);
    optim = OceanStats.Optimization.GenericOptimizer('MaxIter', exp.MaxIter, 'Verbose', false);
    optim.run(recipe, inputData);

    % Assign hard-clusters natively
    log_probs = zeros(size(inputData, 1), K);
    for k = 1:K
         try log_pdf = model.log_mvnpdf(inputData, model.Mu(k,:), model.Sigma(:,:,k));
         catch, log_pdf = log(mvnpdf(inputData, model.Mu(k,:), model.Sigma(:,:,k)) + eps); end
         log_probs(:,k) = log(model.Pi(k) + eps) + log_pdf;
    end
    [~, labels] = max(log_probs, [], 2);

    % Table Cross-Tabulation
    counts = zeros(K, L);
    for c = 1:K
        for l = 1:L
            counts(c, l) = sum((labels == c) & (trueLabels == uniqueLabels(l)));
        end
    end

    fprintf('\n  Cross-Tabulation Matrix:\n');
    fprintf('  Cluster |');
    for l = 1:L, fprintf(' %9s', uniqueLabels(l)); end
    fprintf('\n  %s\n', repmat('-', 1, 9 + 10*L));
    for c = 1:K
        fprintf('     %-2d   |', c);
        for l = 1:L, fprintf(' %9d', counts(c, l)); end
        fprintf('\n');
    end

    % Render Plotting Scatter map
    ax = subplot(1, 3, i, 'Parent', f_scatter);
    hold(ax, 'on');
    palette = viz.getColorSequence(K);
    
    for k = 1:K
        mask = (labels == k);
        if any(mask)
            scatter(ax, X_pca_2(mask, 1), X_pca_2(mask, 2), 50, palette(k, :), 'filled', ...
                'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.8, 'DisplayName', sprintf('C %d', k));
        end
    end
    viz.styleAxes(ax);
    title(ax, exp.Name(1:15), 'FontWeight', 'bold', 'Color', viz.Colors.Text, 'Interpreter', 'none');
    xlabel(ax, 'PC1', 'Color', viz.Colors.Text);
    if i == 1, ylabel(ax, 'PC2', 'Color', viz.Colors.Text); end 
    if i == 1, legend(ax, 'Location', 'southwest', 'TextColor', viz.Colors.Text, 'Color', 'none', 'EdgeColor', 'none', 'FontSize', 7); end
    hold(ax, 'off');
    
    pause(0.1); % Flush graphics buffer
end

fprintf('\n[Done] High-Dimensional GMM Experiment successfully completed.\n');
