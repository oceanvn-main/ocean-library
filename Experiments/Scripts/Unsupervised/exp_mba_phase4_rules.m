%% exp_mba_phase4_rules.m
%  ============================================================
%  PHASE 4: ASSOCIATION RULE GENERATION
%  ============================================================
%
%  Motivation:
%    Convert frequent itemsets from Phase 3 into directed association rules
%    with support, confidence, and lift metrics.
%
%  Pipeline: Reader → Encoder → Apriori → RuleGenerator
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

minSup  = 0.01;   % 1% minimum support
minConf = 0.30;   % 30% minimum confidence

fprintf('=============================================================\n');
fprintf('  MARKET BASKET ANALYSIS — PHASE 4: RULE GENERATION\n');
fprintf('  Dataset: Instacart (Aisle Level)\n');
fprintf('  minSupport:    %.2f%%\n', minSup * 100);
fprintf('  minConfidence: %.2f%%\n', minConf * 100);
fprintf('=============================================================\n\n');

assert(isfolder(dataDir), 'Instacart data not found at: %s', dataDir);

%% ============================================================
%  STEP 1: INGEST (Phase 1)
%  ============================================================
fprintf('--- PHASE 1: INGESTION ---\n');

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

%% ============================================================
%  STEP 2: ENCODE (Phase 2)
%  ============================================================
fprintf('\n--- PHASE 2: ENCODING ---\n');

encoder = OceanStats.Unsupervised.FrequentItemsets.TransactionEncoder();
[X, state] = encoder.fitApply(result.baskets);

%% ============================================================
%  STEP 3: MINE (Phase 3)
%  ============================================================
fprintf('\n--- PHASE 3: APRIORI MINING ---\n');

miner = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner();
mineResult = miner.mine(X, minSup);

%% ============================================================
%  STEP 4: GENERATE RULES (Phase 4)
%  ============================================================
fprintf('\n--- PHASE 4: RULE GENERATION ---\n');

gen = OceanStats.Unsupervised.Rules.RuleGenerator();
rules = gen.generate(mineResult, minConf, ...
    'ColMap', state.colMap, ...
    'Vocabulary', result.vocabulary);

%% ============================================================
%  STEP 5: TOP 20 RULES BY LIFT (Most interesting)
%  ============================================================
fprintf('\n=============================================================\n');
fprintf('  TOP 20 RULES BY LIFT (Most interesting)\n');
fprintf('=============================================================\n\n');

[sortedLift, liftIdx] = sort(rules.lift, 'descend');
printRuleTable(rules, liftIdx, sortedLift, 'Lift', 20);

%% ============================================================
%  STEP 6: TOP 20 RULES BY CONFIDENCE (Most reliable)
%  ============================================================
fprintf('\n=============================================================\n');
fprintf('  TOP 20 RULES BY CONFIDENCE (Most reliable)\n');
fprintf('=============================================================\n\n');

[sortedConf, confIdx] = sort(rules.confidence, 'descend');
printRuleTable(rules, confIdx, sortedConf, 'Conf', 20);

%% ============================================================
%  STEP 7: VERIFICATION ASSERTIONS
%  ============================================================
fprintf('\n=============================================================\n');
fprintf('  VERIFICATION ASSERTIONS\n');
fprintf('=============================================================\n\n');

counts = [0, 0];

% 1. All confidence values in [minConf, 1.0]
counts = checkAssert(all(rules.confidence >= minConf - 1e-10) && all(rules.confidence <= 1.0 + 1e-10), ...
    sprintf('all confidence in [%.2f%%, 100%%]', minConf * 100), counts);

% 2. All lift values > 0
counts = checkAssert(all(rules.lift > 0), 'all lift values > 0', counts);

% 3. All support values in [minSup, 1.0]
counts = checkAssert(all(rules.support >= minSup - 1e-10) && all(rules.support <= 1.0), ...
    sprintf('all support in [%.2f%%, 100%%]', minSup * 100), counts);

% 4. At least some rules generated
counts = checkAssert(rules.stats.nRules > 0, ...
    sprintf('generated %d rules', rules.stats.nRules), counts);

% 5. Spot-check confidence formula: conf = sup(A∪B) / sup(A)
% Build supportMap for verification
supportMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(mineResult.itemsets)
    key = mat2str(sort(mineResult.itemsets{i}));
    supportMap(key) = mineResult.support(i);
end

nCheck = min(100, rules.stats.nRules);
checkIdx = randperm(rules.stats.nRules, nCheck);
formulaOK = true;

for i = 1:nCheck
    idx = checkIdx(i);
    antKey = mat2str(sort(rules.antecedent{idx}));
    consKey = mat2str(sort(rules.consequent{idx}));
    fullKey = mat2str(sort([rules.antecedent{idx}, rules.consequent{idx}]));
    
    supA  = supportMap(antKey);
    supB  = supportMap(consKey);
    supAB = supportMap(fullKey);
    
    expectedConf = supAB / supA;
    expectedLift = expectedConf / supB;
    
    if abs(rules.confidence(idx) - expectedConf) > 1e-10
        fprintf('  [FAIL] Rule %d: conf=%.6f, expected=%.6f\n', idx, rules.confidence(idx), expectedConf);
        formulaOK = false;
    end
    if abs(rules.lift(idx) - expectedLift) > 1e-10
        fprintf('  [FAIL] Rule %d: lift=%.6f, expected=%.6f\n', idx, rules.lift(idx), expectedLift);
        formulaOK = false;
    end
end

counts = checkAssert(formulaOK, ...
    sprintf('confidence & lift formulas verified (%d spot-checks)', nCheck), counts);

% 6. No duplicate rules
ruleKeys = cell(rules.stats.nRules, 1);
for i = 1:rules.stats.nRules
    ruleKeys{i} = [mat2str(sort(rules.antecedent{i})), '->', mat2str(sort(rules.consequent{i}))];
end
nUnique = numel(unique(ruleKeys));
counts = checkAssert(nUnique == rules.stats.nRules, ...
    sprintf('no duplicate rules (%d unique / %d total)', nUnique, rules.stats.nRules), counts);

% 7. Human-readable names exist
counts = checkAssert(~isempty(rules.antecedentNames) && numel(rules.antecedentNames) == rules.stats.nRules, ...
    'human-readable names populated', counts);

nPass = counts(1);
nFail = counts(2);

fprintf('\n  -------------------------------------------\n');
fprintf('  RESULTS: %d PASSED, %d FAILED\n', nPass, nFail);
fprintf('  -------------------------------------------\n\n');

if nFail > 0
    warning('MBA:Phase4', '%d assertion(s) failed. Review output above.', nFail);
else
    fprintf('  Phase 4 Rule Generation: ALL CHECKS PASSED\n\n');
end

fprintf('=============================================================\n');
fprintf('  Phase 4 complete. Rules ready for Phase 5 (filtering/viz).\n');
fprintf('=============================================================\n');

%% ---- Local functions ----

function printRuleTable(rules, sortIdx, sortedVals, metricName, nShow)
    nShow = min(nShow, numel(sortIdx));
    
    fprintf('  %-4s | %-35s | %-25s | %7s | %7s | %6s\n', ...
        'Rank', 'IF (antecedent)', 'THEN (consequent)', 'Sup', 'Conf', metricName);
    fprintf('  %s\n', repmat('-', 1, 95));
    
    for r = 1:nShow
        idx = sortIdx(r);
        
        % Get names (truncate if too long)
        if ~isempty(rules.antecedentNames)
            antStr = rules.antecedentNames{idx};
            conStr = rules.consequentNames{idx};
        else
            antStr = mat2str(rules.antecedent{idx});
            conStr = mat2str(rules.consequent{idx});
        end
        
        if numel(antStr) > 33, antStr = [antStr(1:30), '...']; end
        if numel(conStr) > 23, conStr = [conStr(1:20), '...']; end
        
        fprintf('  %-4d | %-35s | %-25s | %6.2f%% | %6.2f%% | %6.2f\n', ...
            r, antStr, conStr, ...
            rules.support(idx) * 100, ...
            rules.confidence(idx) * 100, ...
            sortedVals(r));
    end
end

function counts = checkAssert(condition, label, counts)
    if condition
        fprintf('  [PASS] %s\n', label);
        counts(1) = counts(1) + 1;
    else
        fprintf('  [FAIL] %s\n', label);
        counts(2) = counts(2) + 1;
    end
end
