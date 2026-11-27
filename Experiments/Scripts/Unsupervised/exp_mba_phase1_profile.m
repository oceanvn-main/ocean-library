%% exp_mba_phase1_profile.m
%  ============================================================
%  PHASE 1: DATA ACQUISITION & DIMENSIONALITY PROFILING
%  ============================================================
%  
%  Motivation:
%    Profile the Instacart dataset at all 3 granularity levels
%    (department, aisle, product) to establish the structural
%    boundaries before any algorithmic processing.
%
%  Outputs:
%    - Dimensional profile table (N, p, sparsity, memory)
%    - Wall-clock timing per read operation
%    - Sample baskets decoded to human-readable strings
%
%  Author: Ocean Library System
%  Date:   2026-04-15

clear; clc; close all;

%% ============================================================
%  CONFIGURATION
%  ============================================================
dataDir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    '..', 'Data', 'InstaCart');

% Resolve to absolute path
dataDir = char(java.io.File(dataDir).getCanonicalPath());

fprintf('=============================================================\n');
fprintf('  MARKET BASKET ANALYSIS — PHASE 1: DATA PROFILING\n');
fprintf('  Dataset: Instacart\n');
fprintf('  Path:    %s\n', dataDir);
fprintf('=============================================================\n\n');

% Verify dataset exists
assert(isfolder(dataDir), ...
    'Instacart data not found at: %s', dataDir);

%% ============================================================
%  GRANULARITY CONFIGURATIONS (generic column-based)
%  ============================================================
%
%  Instacart file layout:
%    order_products__prior.csv : order_id(1), product_id(2), ...
%    products.csv              : product_id(1), product_name(2), aisle_id(3), department_id(4)
%    aisles.csv                : aisle_id(1), aisle(2)
%    departments.csv           : department_id(1), department(2)

configs = struct( ...
    'name',          {'department',     'aisle',         'product'}, ...
    'EnrichFile',    {'products.csv',   'products.csv',  ''}, ...
    'EnrichFromCol', {1,                1,               1}, ...
    'EnrichToCol',   {4,                3,               2}, ...
    'LookupFile',    {'departments.csv','aisles.csv',    'products.csv'}, ...
    'LookupLabelCol',{2,                2,               2} ...
);

%% ============================================================
%  STEP 1: PROFILE AT ALL 3 GRANULARITY LEVELS
%  ============================================================

results = struct();
timings = zeros(1, numel(configs));

for g = 1:numel(configs)
    cfg = configs(g);
    
    fprintf('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n');
    fprintf('  GRANULARITY: %s\n', upper(cfg.name));
    fprintf('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n');
    
    % Build reader with generic parameters
    readerArgs = { ...
        dataDir, ...
        'Mode', 'relational', ...
        'PrimaryFile', 'order_products__prior.csv', ...
        'TransactionCol', 1, ...
        'ItemCol', 2, ...
        'LookupFile', cfg.LookupFile, ...
        'LookupLabelCol', cfg.LookupLabelCol ...
    };
    
    % Add enrichment only if specified
    if ~isempty(cfg.EnrichFile)
        readerArgs = [readerArgs, { ...
            'EnrichFile', cfg.EnrichFile, ...
            'EnrichFromCol', cfg.EnrichFromCol, ...
            'EnrichToCol', cfg.EnrichToCol ...
        }]; %#ok<AGROW>
    end
    
    reader = OceanPipeline.Ingestion.TransactionalReader(readerArgs{:});
    
    tStart = tic;
    result = reader.read();
    timings(g) = toc(tStart);
    
    % Store for comparison
    results.(cfg.name) = result;
    
    fprintf('  Wall-clock time: %.1f seconds\n\n', timings(g));
end

%% ============================================================
%  STEP 2: COMPARISON TABLE
%  ============================================================

fmt  = @OceanPipeline.Ingestion.TransactionalReader.formatNumber;
fmtB = @OceanPipeline.Ingestion.TransactionalReader.formatBytes;

fprintf('\n');
fprintf('=============================================================\n');
fprintf('       INSTACART — DIMENSIONAL PROFILE COMPARISON\n');
fprintf('=============================================================\n');
fprintf('  %-14s | %12s | %10s | %14s | %12s\n', ...
    'Granularity', 'N', 'p', 'Sparsity', 'Dense Memory');
fprintf('  %s\n', repmat('-', 1, 70));

for g = 1:numel(configs)
    cfg = configs(g);
    prof = results.(cfg.name).profile;
    
    fprintf('  %-14s | %12s | %10s | %13.4f%% | %12s\n', ...
        cfg.name, ...
        fmt(prof.nTransactions), ...
        fmt(prof.nUniqueItems), ...
        prof.sparsityRatio * 100, ...
        fmtB(prof.memoryEstimate));
end

fprintf('  %s\n', repmat('-', 1, 70));
fprintf('  %-14s | %12s | %12s | %12s\n', ...
    'Granularity', 'Read Time', 'Avg Basket', 'Sparse?');
fprintf('  %s\n', repmat('-', 1, 55));

for g = 1:numel(configs)
    cfg = configs(g);
    prof = results.(cfg.name).profile;
    
    sparseStr = "NO";
    if prof.sparseRecommended
        sparseStr = "YES";
    end
    
    fprintf('  %-14s | %10.1f s | %10.1f | %12s\n', ...
        cfg.name, timings(g), prof.avgBasketSize, sparseStr);
end

fprintf('=============================================================\n\n');

%% ============================================================
%  STEP 3: SAMPLE BASKETS (SANITY CHECK)
%  ============================================================

fprintf('=============================================================\n');
fprintf('  SAMPLE BASKETS — Decoded to Human-Readable Strings\n');
fprintf('=============================================================\n\n');

% Show samples at aisle level (most interpretable)
aisleResult = results.aisle;
nSamples = min(5, numel(aisleResult.baskets));

for i = 1:nSamples
    basket = aisleResult.baskets{i};
    names = cell(1, numel(basket));
    for j = 1:numel(basket)
        if aisleResult.vocabulary.isKey(basket(j))
            names{j} = aisleResult.vocabulary(basket(j));
        else
            names{j} = sprintf('[ID:%d]', basket(j));
        end
    end
    
    fprintf('  Basket %d (%d items):\n', i, numel(basket));
    fprintf('    %s\n', strjoin(names, ', '));
    fprintf('\n');
end

%% ============================================================
%  STEP 4: AUTOMATED ASSERTIONS
%  ============================================================

fprintf('=============================================================\n');
fprintf('  VERIFICATION ASSERTIONS\n');
fprintf('=============================================================\n\n');

counts = [0, 0];  % [nPass, nFail]

granNames = {configs.name};

% 1. Baskets are non-empty
for g = 1:numel(granNames)
    gran = granNames{g};
    counts = checkAssert(~isempty(results.(gran).baskets), ...
        sprintf('%s: baskets non-empty', gran), counts);
end

% 2. All item IDs are positive
for g = 1:numel(granNames)
    gran = granNames{g};
    allPositive = all(cellfun(@(b) all(b > 0), results.(gran).baskets));
    counts = checkAssert(allPositive, ...
        sprintf('%s: all item IDs positive', gran), counts);
end

% 3. Vocabulary size matches profile.nUniqueItems
for g = 1:numel(granNames)
    gran = granNames{g};
    vocabMatch = results.(gran).vocabulary.Count == ...
        results.(gran).profile.nUniqueItems;
    counts = checkAssert(vocabMatch, ...
        sprintf('%s: vocabulary size matches p', gran), counts);
end

% 4. Sparsity ratio is valid (0 < s < 1)
for g = 1:numel(granNames)
    gran = granNames{g};
    s = results.(gran).profile.sparsityRatio;
    counts = checkAssert(s > 0 && s < 1, ...
        sprintf('%s: sparsity ratio in (0,1)', gran), counts);
end

% 5. Known ground truths
counts = checkAssert(results.department.profile.nUniqueItems == 21, ...
    'department: p == 21 (ground truth)', counts);
counts = checkAssert(results.aisle.profile.nUniqueItems == 134, ...
    'aisle: p == 134 (ground truth)', counts);

% 6. Transaction count consistent across granularities
nDept  = results.department.profile.nTransactions;
nAisle = results.aisle.profile.nTransactions;
nProd  = results.product.profile.nTransactions;
counts = checkAssert(nDept == nAisle && nAisle == nProd, ...
    'N consistent across all granularities', counts);

% 7. Sparsity increases with granularity
sDept  = results.department.profile.sparsityRatio;
sAisle = results.aisle.profile.sparsityRatio;
sProd  = results.product.profile.sparsityRatio;
counts = checkAssert(sDept < sAisle && sAisle < sProd, ...
    'sparsity: department < aisle < product (monotonic)', counts);

nPass = counts(1);
nFail = counts(2);

fprintf('\n  -------------------------------------------\n');
fprintf('  RESULTS: %d PASSED, %d FAILED\n', nPass, nFail);
fprintf('  -------------------------------------------\n\n');

if nFail > 0
    warning('MBA:Phase1', '%d assertion(s) failed. Review output above.', nFail);
else
    fprintf('  Phase 1 Data Acquisition: ALL CHECKS PASSED\n\n');
end

fprintf('=============================================================\n');
fprintf('  Phase 1 complete. Results ready for Phase 2.\n');
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
