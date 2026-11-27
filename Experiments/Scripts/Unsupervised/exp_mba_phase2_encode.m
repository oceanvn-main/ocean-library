%% exp_mba_phase2_encode.m
%  ============================================================
%  PHASE 2: GEOMETRIC TRANSFORMATION — SPARSE BOOLEAN MATRIX
%  ============================================================
%
%  Motivation:
%    Convert the variable-length basket arrays from Phase 1 into a sparse
%    Boolean matrix X(N x p) suitable for vectorized support counting
%    in the Apriori and FP-Growth algorithms.
%
%  Inputs:
%    - result.baskets from Phase 1 (TransactionalReader)
%
%  Outputs:
%    - X_sparse: sparse(N x p) logical matrix
%    - state:    struct with colMap, lookup, p
%    - Matrix statistics: nnz, sparsity, memory
%    - Round-trip validation results
%
%  Author: Ocean Library System
%  Date:   2026-04-15

clear; clc; close all;

%% ============================================================
%  CONFIGURATION
%  ============================================================
dataDir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    '..', 'Data', 'InstaCart');
dataDir = char(java.io.File(dataDir).getCanonicalPath());

fprintf('=============================================================\n');
fprintf('  MARKET BASKET ANALYSIS — PHASE 2: TRANSACTION ENCODING\n');
fprintf('  Dataset: Instacart\n');
fprintf('  Path:    %s\n', dataDir);
fprintf('=============================================================\n\n');

assert(isfolder(dataDir), 'Instacart data not found at: %s', dataDir);

%% ============================================================
%  GRANULARITY CONFIGURATIONS (same as Phase 1)
%  ============================================================

configs = struct( ...
    'name',          {'department',     'aisle',         'product'}, ...
    'EnrichFile',    {'products.csv',   'products.csv',  ''}, ...
    'EnrichFromCol', {1,                1,               1}, ...
    'EnrichToCol',   {4,                3,               2}, ...
    'LookupFile',    {'departments.csv','aisles.csv',    'products.csv'}, ...
    'LookupLabelCol',{2,                2,               2} ...
);

%% ============================================================
%  STEP 1: INGEST + ENCODE AT ALL 3 GRANULARITY LEVELS
%  ============================================================

encoder = OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder();

encResults = struct();
timings = zeros(1, numel(configs));

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
        'LookupFile', cfg.LookupFile, ...
        'LookupLabelCol', cfg.LookupLabelCol ...
    };
    if ~isempty(cfg.EnrichFile)
        readerArgs = [readerArgs, { ...
            'EnrichFile', cfg.EnrichFile, ...
            'EnrichFromCol', cfg.EnrichFromCol, ...
            'EnrichToCol', cfg.EnrichToCol ...
        }]; %#ok<AGROW>
    end
    
    reader = OceanPipeline.Ingestion.TransactionalReader(readerArgs{:});
    result = reader.read();
    
    % --- Phase 2: Encode ---
    fprintf('\n--- ENCODING ---\n');
    tStart = tic;
    [X, state] = encoder.fitApply(result.baskets);
    timings(g) = toc(tStart);
    
    % Store everything
    encResults.(cfg.name).X          = X;
    encResults.(cfg.name).state      = state;
    encResults.(cfg.name).baskets    = result.baskets;
    encResults.(cfg.name).vocabulary = result.vocabulary;
    encResults.(cfg.name).profile    = result.profile;
    
    fprintf('  Encode wall-clock: %.2f seconds\n\n', timings(g));
end

%% ============================================================
%  STEP 2: COMPARISON TABLE
%  ============================================================

fmt = @OceanPipeline.Ingestion.TransactionalReader.formatNumber;

fprintf('\n');
fprintf('=============================================================\n');
fprintf('       TRANSACTION ENCODING — COMPARISON TABLE\n');
fprintf('=============================================================\n');
fprintf('  %-14s | %12s | %8s | %12s | %10s | %8s\n', ...
    'Granularity', 'N', 'p', 'nnz', 'Sparse MB', 'Time');
fprintf('  %s\n', repmat('-', 1, 74));

for g = 1:numel(configs)
    cfg = configs(g);
    X = encResults.(cfg.name).X;
    [N, p] = size(X);
    nz = nnz(X);
    memMB = nz * 16 / 1e6;
    
    fprintf('  %-14s | %12s | %8s | %12s | %8.0f MB | %6.1fs\n', ...
        cfg.name, fmt(N), fmt(p), fmt(nz), memMB, timings(g));
end

fprintf('=============================================================\n\n');

%% ============================================================
%  STEP 3: SAMPLE VERIFICATION (AISLE LEVEL)
%  ============================================================

fprintf('=============================================================\n');
fprintf('  SAMPLE VERIFICATION — First 5 Baskets (Aisle Level)\n');
fprintf('=============================================================\n\n');

aisleData = encResults.aisle;
nSamples = min(5, numel(aisleData.baskets));

for i = 1:nSamples
    originalIds   = sort(aisleData.baskets{i});
    matrixCols    = find(aisleData.X(i, :));
    decodedIds    = sort(aisleData.state.colMap(matrixCols));
    
    match = isequal(originalIds(:)', decodedIds(:)');
    matchStr = 'MATCH';
    if ~match, matchStr = 'MISMATCH'; end
    
    % Decode to human-readable
    names = cell(1, numel(originalIds));
    for j = 1:numel(originalIds)
        if aisleData.vocabulary.isKey(originalIds(j))
            names{j} = aisleData.vocabulary(originalIds(j));
        else
            names{j} = sprintf('[ID:%d]', originalIds(j));
        end
    end
    
    fprintf('  Basket %d (%d items) [%s]:\n', i, numel(originalIds), matchStr);
    fprintf('    Original IDs:  [%s]\n', num2str(originalIds));
    fprintf('    Matrix cols:   [%s]\n', num2str(matrixCols));
    fprintf('    Decoded IDs:   [%s]\n', num2str(decodedIds));
    fprintf('    Names:         %s\n\n', strjoin(names, ', '));
end

%% ============================================================
%  STEP 4: TOP ITEMS (MOST FREQUENT)
%  ============================================================

fprintf('=============================================================\n');
fprintf('  TOP 10 MOST FREQUENT ITEMS (Aisle Level)\n');
fprintf('=============================================================\n\n');

itemFreqs = full(sum(aisleData.X, 1));     % (1 x p) frequency vector
[sortedFreqs, sortIdx] = sort(itemFreqs, 'descend');

fprintf('  %-4s | %-30s | %12s | %8s\n', 'Rank', 'Item', 'Frequency', 'Support%');
fprintf('  %s\n', repmat('-', 1, 62));

N_aisle = size(aisleData.X, 1);
for r = 1:min(10, numel(sortIdx))
    colJ = sortIdx(r);
    itemId = aisleData.state.colMap(colJ);
    
    if aisleData.vocabulary.isKey(itemId)
        itemName = aisleData.vocabulary(itemId);
    else
        itemName = sprintf('Item %d', itemId);
    end
    
    fprintf('  %-4d | %-30s | %12s | %7.2f%%\n', ...
        r, itemName, fmt(sortedFreqs(r)), 100 * sortedFreqs(r) / N_aisle);
end

fprintf('\n');

%% ============================================================
%  STEP 5: ROUND-TRIP VALIDATION
%  ============================================================

fprintf('=============================================================\n');
fprintf('  ROUND-TRIP VALIDATION\n');
fprintf('=============================================================\n\n');

for g = 1:numel(configs)
    cfg = configs(g);
    data = encResults.(cfg.name);
    
    % Full validation on first 1000 baskets (full validation on 3.4M is slow)
    nCheck = min(1000, numel(data.baskets));
    subBaskets = data.baskets(1:nCheck);
    subX = data.X(1:nCheck, :);
    
    ok = encoder.validate(subX, subBaskets, data.state);
    
    if ok
        fprintf('  [PASS] %s: first %d baskets round-trip OK\n', cfg.name, nCheck);
    else
        fprintf('  [FAIL] %s: round-trip FAILED\n', cfg.name);
    end
end

fprintf('\n');

%% ============================================================
%  STEP 6: AUTOMATED ASSERTIONS
%  ============================================================

fprintf('=============================================================\n');
fprintf('  VERIFICATION ASSERTIONS\n');
fprintf('=============================================================\n\n');

counts = [0, 0];  % [nPass, nFail]

granNames = {configs.name};

for g = 1:numel(granNames)
    gran = granNames{g};
    X = encResults.(gran).X;
    baskets = encResults.(gran).baskets;
    state = encResults.(gran).state;
    prof = encResults.(gran).profile;
    
    [N, p] = size(X);
    
    % 1. Matrix is sparse
    counts = checkAssert(issparse(X), sprintf('%s: X is sparse', gran), counts);
    
    % 2. Matrix is logical
    counts = checkAssert(islogical(X), sprintf('%s: X is logical', gran), counts);
    
    % 3. Dimensions match profile
    counts = checkAssert(N == prof.nTransactions, ...
        sprintf('%s: N=%d matches profile', gran, N), counts);
    counts = checkAssert(p == prof.nActiveItems, ...
        sprintf('%s: p=%d matches active items in baskets', gran, p), counts);
    
    % 4. nnz matches total basket items
    totalItems = sum(cellfun(@numel, baskets));
    counts = checkAssert(nnz(X) == totalItems, ...
        sprintf('%s: nnz=%d matches total items', gran, nnz(X)), counts);
    
    % 5. No empty rows
    rowSums = sum(X, 2);
    counts = checkAssert(all(rowSums > 0), sprintf('%s: no empty rows', gran), counts);
    
    % 6. colMap size matches p
    counts = checkAssert(numel(state.colMap) == p, ...
        sprintf('%s: colMap length matches p', gran), counts);
end

nPass = counts(1);
nFail = counts(2);

fprintf('\n  -------------------------------------------\n');
fprintf('  RESULTS: %d PASSED, %d FAILED\n', nPass, nFail);
fprintf('  -------------------------------------------\n\n');

if nFail > 0
    warning('MBA:Phase2', '%d assertion(s) failed. Review output above.', nFail);
else
    fprintf('  Phase 2 Transaction Encoding: ALL CHECKS PASSED\n\n');
end

fprintf('=============================================================\n');
fprintf('  Phase 2 complete. Sparse matrices ready for Phase 3.\n');
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
