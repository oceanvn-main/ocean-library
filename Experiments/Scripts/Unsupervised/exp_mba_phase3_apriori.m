%% exp_mba_phase3_apriori.m
%  ============================================================
%  PHASE 3: APRIORI — FREQUENT ITEMSET MINING
%  ============================================================
%
%  Motivation:
%    Apply the Apriori algorithm to the sparse Boolean matrix from Phase 2
%    to discover itemsets that co-occur above a minimum support threshold.
%    Observe the anti-monotone property and level-wise search in action.
%
%  Inputs:
%    - Sparse matrix X from Phase 2 (TransactionEncoder)
%    - minSupport threshold
%
%  Outputs:
%    - Frequent itemsets with support values
%    - k-level breakdown (candidates, pruned, frequent)
%    - Top itemsets with human-readable names
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

minSup = 0.01;  % 1% minimum support → item/pair must appear in >= 34K baskets

fprintf('=============================================================\n');
fprintf('  MARKET BASKET ANALYSIS — PHASE 3: APRIORI MINING\n');
fprintf('  Dataset: Instacart\n');
fprintf('  minSupport: %.2f%% (%d baskets)\n', minSup*100, ceil(minSup*3421083));
fprintf('=============================================================\n\n');

assert(isfolder(dataDir), 'Instacart data not found at: %s', dataDir);

%% ============================================================
%  GRANULARITY CONFIGS (department + aisle only — product too slow for Apriori)
%  ============================================================

configs = struct( ...
    'name',          {'department',     'aisle'}, ...
    'EnrichFile',    {'products.csv',   'products.csv'}, ...
    'EnrichFromCol', {1,                1}, ...
    'EnrichToCol',   {4,                3}, ...
    'LookupFile',    {'departments.csv','aisles.csv'}, ...
    'LookupLabelCol',{2,                2} ...
);

%% ============================================================
%  STEP 1: INGEST + ENCODE + MINE
%  ============================================================

encoder = OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder();
miner   = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner();

miningResults = struct();

for g = 1:numel(configs)
    cfg = configs(g);
    
    fprintf('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n');
    fprintf('  GRANULARITY: %s\n', upper(cfg.name));
    fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n');
    
    % --- Phase 1: Read baskets ---
    readerArgs = { ...
        dataDir, ...
        'Mode', 'relational', ...
        'PrimaryFile', 'order_products__prior.csv', ...
        'TransactionCol', 1, ...
        'ItemCol', 2, ...
        'EnrichFile', cfg.EnrichFile, ...
        'EnrichFromCol', cfg.EnrichFromCol, ...
        'EnrichToCol', cfg.EnrichToCol, ...
        'LookupFile', cfg.LookupFile, ...
        'LookupLabelCol', cfg.LookupLabelCol ...
    };
    
    reader = OceanPipeline.Ingestion.TransactionalReader(readerArgs{:});
    result = reader.read();
    
    % --- Phase 2: Encode ---
    fprintf('\n--- ENCODING ---\n');
    [X, state] = encoder.fitApply(result.baskets);
    
    % --- Phase 3: Mine ---
    fprintf('\n--- APRIORI MINING ---\n');
    mineResult = miner.mine(X, minSup);
    
    % Store
    miningResults.(cfg.name).mineResult = mineResult;
    miningResults.(cfg.name).state      = state;
    miningResults.(cfg.name).vocabulary = result.vocabulary;
    miningResults.(cfg.name).X          = X;
end

%% ============================================================
%  STEP 2: TOP FREQUENT ITEMSETS (AISLE LEVEL)
%  ============================================================

fprintf('\n=============================================================\n');
fprintf('  TOP 20 FREQUENT ITEMSETS (Aisle Level)\n');
fprintf('=============================================================\n\n');

aisleResult = miningResults.aisle;
mr = aisleResult.mineResult;
vocab = aisleResult.vocabulary;
colMap = aisleResult.state.colMap;

% Sort by support descending
[sortedSup, sortIdx] = sort(mr.support, 'descend');

fprintf('  %-4s | %-50s | %8s\n', 'Rank', 'Itemset', 'Support');
fprintf('  %s\n', repmat('-', 1, 68));

nShow = min(20, numel(sortIdx));
for r = 1:nShow
    idx = sortIdx(r);
    itemCols = mr.itemsets{idx};
    
    % Translate column indices → item IDs → names
    names = cell(1, numel(itemCols));
    for j = 1:numel(itemCols)
        itemId = colMap(itemCols(j));
        if vocab.isKey(itemId)
            names{j} = vocab(itemId);
        else
            names{j} = sprintf('ID:%d', itemId);
        end
    end
    
    labelStr = strjoin(names, ' + ');
    if numel(labelStr) > 48
        labelStr = [labelStr(1:45), '...'];
    end
    
    fprintf('  %-4d | %-50s | %7.2f%%\n', r, labelStr, sortedSup(r) * 100);
end

%% ============================================================
%  STEP 3: K-LEVEL BREAKDOWN COMPARISON
%  ============================================================

fprintf('\n=============================================================\n');
fprintf('  K-LEVEL BREAKDOWN — ALL GRANULARITIES\n');
fprintf('=============================================================\n\n');

for g = 1:numel(configs)
    cfg = configs(g);
    stats = miningResults.(cfg.name).mineResult.stats;
    
    fprintf('  --- %s ---\n', upper(cfg.name));
    fprintf('  %-4s | %12s | %12s | %12s | %8s\n', ...
        'k', 'Candidates', 'Pruned', 'Frequent', 'Time');
    fprintf('  %s\n', repmat('-', 1, 56));
    
    for k = 1:numel(stats.kBreakdown)
        fprintf('  %-4d | %12d | %12d | %12d | %6.2f s\n', ...
            k, stats.kCandidates(k), stats.kPruned(k), ...
            stats.kBreakdown(k), stats.kElapsed(k));
    end
    fprintf('  Total: %d frequent itemsets (%.2f s)\n\n', ...
        stats.nItemsets, stats.elapsed);
end

%% ============================================================
%  STEP 4: ANTI-MONOTONE PROPERTY VERIFICATION
%  ============================================================

fprintf('=============================================================\n');
fprintf('  ANTI-MONOTONE PROPERTY CHECK (Aisle Level)\n');
fprintf('=============================================================\n\n');

% Build a Map: itemset string → support
supportMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(mr.itemsets)
    key = mat2str(sort(mr.itemsets{i}));
    supportMap(key) = mr.support(i);
end

antiMonotoneOK = true;
nChecked = 0;

for i = 1:numel(mr.itemsets)
    itemset = sort(mr.itemsets{i});
    k = numel(itemset);
    
    if k < 2, continue; end
    
    % Check all (k-1) subsets
    for j = 1:k
        subset = itemset([1:j-1, j+1:k]);
        key = mat2str(sort(subset));
        
        if ~supportMap.isKey(key)
            fprintf('  [FAIL] Subset %s of %s NOT found in frequent itemsets\n', ...
                key, mat2str(itemset));
            antiMonotoneOK = false;
        else
            subSupport = supportMap(key);
            parentSupport = mr.support(i);
            
            if subSupport < parentSupport - 1e-10
                fprintf('  [FAIL] support(%s)=%.4f < support(%s)=%.4f\n', ...
                    key, subSupport, mat2str(itemset), parentSupport);
                antiMonotoneOK = false;
            end
        end
        nChecked = nChecked + 1;
    end
end

if antiMonotoneOK
    fprintf('  [PASS] Anti-monotone property holds (%d subset checks)\n\n', nChecked);
else
    fprintf('  [FAIL] Anti-monotone violations detected!\n\n');
end

%% ============================================================
%  STEP 5: VERIFICATION ASSERTIONS
%  ============================================================

fprintf('=============================================================\n');
fprintf('  VERIFICATION ASSERTIONS\n');
fprintf('=============================================================\n\n');

counts = [0, 0];

for g = 1:numel(configs)
    cfg = configs(g);
    mr_g = miningResults.(cfg.name).mineResult;
    
    % 1. All support values in [minSup, 1.0]
    allInRange = all(mr_g.support >= minSup - 1e-10) && all(mr_g.support <= 1.0);
    counts = checkAssert(allInRange, ...
        sprintf('%s: all support values in [%.4f, 1.0]', cfg.name, minSup), counts);
    
    % 2. Itemsets are sorted within each set
    allSorted = true;
    for i = 1:numel(mr_g.itemsets)
        if ~issorted(mr_g.itemsets{i})
            allSorted = false;
            break;
        end
    end
    counts = checkAssert(allSorted, ...
        sprintf('%s: all itemsets internally sorted', cfg.name), counts);
    
    % 3. k-levels eventually converge to 0 (algorithm terminates)
    kFreq = mr_g.stats.kBreakdown;
    converges = kFreq(end) < kFreq(1);  % Last level has fewer than first
    counts = checkAssert(converges, ...
        sprintf('%s: k-levels converge (k=1:%d → k=%d:%d)', ...
            cfg.name, kFreq(1), numel(kFreq), kFreq(end)), counts);
    
    % 4. At least some itemsets found
    counts = checkAssert(mr_g.stats.nItemsets > 0, ...
        sprintf('%s: found %d frequent itemsets', cfg.name, mr_g.stats.nItemsets), counts);
    
    % 5. k=1 frequent items <= p
    counts = checkAssert(mr_g.stats.kBreakdown(1) <= size(miningResults.(cfg.name).X, 2), ...
        sprintf('%s: freq-1 items <= p', cfg.name), counts);
end

% 6. Anti-monotone property
counts = checkAssert(antiMonotoneOK, 'aisle: anti-monotone property holds', counts);

% 7. Department: all 21 items should be frequent at 1% support
deptFreq1 = miningResults.department.mineResult.stats.kBreakdown(1);
counts = checkAssert(deptFreq1 == 21, ...
    sprintf('department: all 21 singletons frequent (got %d)', deptFreq1), counts);

nPass = counts(1);
nFail = counts(2);

fprintf('\n  -------------------------------------------\n');
fprintf('  RESULTS: %d PASSED, %d FAILED\n', nPass, nFail);
fprintf('  -------------------------------------------\n\n');

if nFail > 0
    warning('MBA:Phase3', '%d assertion(s) failed. Review output above.', nFail);
else
    fprintf('  Phase 3 Apriori Mining: ALL CHECKS PASSED\n\n');
end

fprintf('=============================================================\n');
fprintf('  Phase 3 complete. Frequent itemsets ready for Phase 4.\n');
fprintf('=============================================================\n');

%% ---- Local function (must be at end of script) ----
function counts = checkAssert(condition, label, counts)
    if condition
        fprintf('  [PASS] %s\n', label);
        counts(1) = counts(1) + 1;
    else
        fprintf('  [FAIL] %s\n', label);
        counts(2) = counts(2) + 1;
    end
end
