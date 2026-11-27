%% exp_market_basket_analysis.m
%  ============================================================
%  MARKET BASKET ANALYSIS — FULL PIPELINE (ALL 5 PHASES)
%  ============================================================
%
%  End-to-end experiment script:
%    Phase 1: Data Acquisition (TransactionalReader)
%    Phase 2: Geometric Transform (TransactionEncoder)
%    Phase 3: Algorithmic Engine (AprioriMiner)
%    Phase 4: Rule Generation (RuleGenerator)
%    Phase 5: Evaluation & Visualization (RuleFilter + OceanViz)
%
%  Dataset: Instacart, Aisle Level
%
%  Author: Ocean Library System
%  Date:   2026-04-16

clear; clc; close all;

%% ============================================================
%  CONFIGURATION
%  ============================================================
dataDir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    '..', 'Data', 'InstaCart');
dataDir = char(java.io.File(dataDir).getCanonicalPath());

% Pipeline parameters (all tunable)
minSup   = 0.01;   % 1% minimum support
minConf  = 0.30;   % 30% minimum confidence
minLift  = 1.0;    % Only rules better than random
maxRules = 50;     % Top 50 rules

fprintf('=============================================================\n');
fprintf('  MARKET BASKET ANALYSIS — FULL PIPELINE\n');
fprintf('  Dataset:       Instacart (Aisle Level)\n');
fprintf('  minSupport:    %.2f%%\n', minSup * 100);
fprintf('  minConfidence: %.2f%%\n', minConf * 100);
fprintf('  minLift:       %.2f\n', minLift);
fprintf('  maxRules:      %d\n', maxRules);
fprintf('=============================================================\n\n');

assert(isfolder(dataDir), 'Instacart data not found at: %s', dataDir);

%% ============================================================
%  PHASE 1: DATA ACQUISITION. Read raw data and product Shopping baskets
%  ============================================================
fprintf('>>> PHASE 1: DATA ACQUISITION <<<\n');

reader = OceanPipeline.Ingestion.TransactionalReader( ...
    dataDir, ...
    'Mode', 'relational', ...
    'PrimaryFile', 'order_products__prior.csv', ...
    'TransactionCol', 1, ...
    'ItemCol', 2, ...
    'EnrichFile', 'products.csv', ...
    'EnrichFromCol', 1, ...
    'EnrichToCol', 3, ...
    'LookupFile', 'aisles.csv', ...
    'LookupLabelCol', 2);

result = reader.read();
fprintf('  Baskets: %d, Vocabulary: %d items\n\n', ...
    numel(result.baskets), result.vocabulary.Count);

%% ============================================================
%  PHASE 2: GEOMETRIC TRANSFORM. Turn baskets into matrix (sparse)
%  ============================================================
fprintf('>>> PHASE 2: GEOMETRIC TRANSFORM <<<\n');

encoder = OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder();
[X, state] = encoder.fitApply(result.baskets);
fprintf('  Sparse matrix: %d x %d, nnz=%d, sparsity=%.2f%%\n\n', ...
    size(X,1), size(X,2), nnz(X), (1 - nnz(X)/(size(X,1)*size(X,2)))*100);

%% ============================================================
%  PHASE 3: APRIORI MINING
%  ============================================================
fprintf('>>> PHASE 3: APRIORI MINING <<<\n');

miner = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner();
mineResult = miner.mine(X, minSup);

%% ============================================================
%  PHASE 4: RULE GENERATION
%  ============================================================
fprintf('\n>>> PHASE 4: RULE GENERATION <<<\n');

gen = OceanStats.Unsupervised.Rules.RuleGenerator();
rules = gen.generate(mineResult, minConf, ...
    'ColMap', state.colMap, ...
    'Vocabulary', result.vocabulary);

%% ============================================================
%  PHASE 5: EVALUATION & VISUALIZATION
%  ============================================================
fprintf('\n>>> PHASE 5: EVALUATION & VISUALIZATION <<<\n');

% 5a. Filter rules
filt = OceanStats.Unsupervised.Rules.RuleFilter();
filtered = filt.filter(rules, ...
    'MinLift', minLift, ...
    'MaxRules', maxRules, ...
    'RemoveRedundant', true, ...
    'SortBy', 'lift');

% 5b. Print final rule table
fprintf('\n=============================================================\n');
fprintf('  FINAL RULES TABLE (Top %d by Lift)\n', filtered.stats.nFiltered);
fprintf('=============================================================\n\n');

fprintf('  %-4s | %-30s | %-22s | %6s | %6s | %6s\n', ...
    '#', 'IF (buy this)', 'THEN (also buy)', 'Sup', 'Conf', 'Lift');
fprintf('  %s\n', repmat('-', 1, 85));

for i = 1:filtered.stats.nFiltered
    antStr = filtered.antecedentNames{i};
    conStr = filtered.consequentNames{i};
    
    if numel(antStr) > 28, antStr = [antStr(1:25), '...']; end
    if numel(conStr) > 20, conStr = [conStr(1:17), '...']; end
    
    fprintf('  %-4d | %-30s | %-22s | %5.2f%% | %5.1f%% | %5.2f\n', ...
        i, antStr, conStr, ...
        filtered.support(i) * 100, ...
        filtered.confidence(i) * 100, ...
        filtered.lift(i));
end

% 5c. Visualize with OceanViz
fprintf('\n  Rendering scatter plot...\n');

viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('Market Basket Analysis — Association Rules');

ax = axes(fig);

% Prepare data struct for OceanViz
vizData.confidence  = filtered.confidence;
vizData.lift        = filtered.lift;
vizData.support     = filtered.support;
vizData.itemsetSize = filtered.itemsetSize;
vizData.nAnnotate   = 5;

% Build labels: "antecedent → consequent"
vizData.labels = cell(filtered.stats.nFiltered, 1);
for i = 1:filtered.stats.nFiltered
    vizData.labels{i} = sprintf('%s → %s', ...
        filtered.antecedentNames{i}, filtered.consequentNames{i});
end

viz.plot(ax, 'AssociationRule', vizData, ...
    'Title', 'Association Rules: Confidence vs Lift', ...
    'XLabel', 'Confidence', ...
    'YLabel', 'Lift');

fprintf('\n=============================================================\n');
fprintf('  PIPELINE COMPLETE\n');
fprintf('  %d baskets → %d itemsets → %d rules → %d filtered\n', ...
    numel(result.baskets), mineResult.stats.nItemsets, ...
    rules.stats.nRules, filtered.stats.nFiltered);
fprintf('=============================================================\n');
