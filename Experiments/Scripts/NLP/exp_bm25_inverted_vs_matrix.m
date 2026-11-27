% EXP_BM25_INVERTED_VS_MATRIX  Controlled comparison of the two BM25 index strategies.
%
%   The single independent variable is the index representation inside
%   OceanNLP.Retrieval.BM25Ranker:
%     "inverted_index" — per-term posting lists (docId, tf); BM25 weights are
%                        computed lazily at query time for the query's terms.
%     "matrix"         — (N x V) sparse BM25 weight matrix W precomputed at fit;
%                        a query is one sparse product  s = W(:, q) * qtf'.
%
%   HELD FIXED across both arms (so any difference is due to the representation):
%     - corpus (same string array, same seed per N), vocabulary size, doc length
%     - query set (same 20 queries for every N and both strategies)
%     - tokenizer (one WhitespaceTokenizer configuration)
%     - BM25 parameters k1 = 1.2, b = 0.75, IDF scheme (BM25Idf default)
%     - timing protocol: both arms warmed up; query latency via timeit() on
%       IndexStrategy.score() only — the shared O(N log N) sort inside rank()
%       is excluded because it is identical for both and would dilute the gap
%     - fit order alternates per corpus size (cancels cache/JIT order bias)
%     - memory: actual bytes (whos) of each index's stored payload only;
%       the Vocabulary object is identical in both and is excluded
%
%   KNOWN NON-REPRESENTATION DIFFERENCE (reported, not hidden):
%     MatrixIndexStrategy.fit tokenizes the corpus twice (CountVectorizer.fit
%     then .forward); InvertedIndex.fit tokenizes once. One tokenization pass
%     is timed separately so its share of fit time is visible.
%
%   Outputs: printed table + `results` struct in the workspace + OceanViz figure.

clear; clc; close all;

% -------------------------------------------------------------------------
% FIXED EXPERIMENT CONFIGURATION
% -------------------------------------------------------------------------
masterSeed   = 42;
corpusSizes  = [100, 1000, 10000, 100000, 1000000];
avgDocLen    = 30;       % mean words per document
vocabSize    = 5000;     % synthetic vocabulary size
numQueries   = 20;       % fixed query set size
fitRepsSmall = 3;        % fit repetitions (median) when N <= fitRepCutoff
fitRepCutoff = 10000;

bm25Params = struct('TermSaturation', 1.2, 'LengthNormalization', 0.75);
tokenizer  = OceanNLP.Tokenize.WhitespaceTokenizer();
strategies = ["inverted_index", "matrix"];
numSizes   = numel(corpusSizes);

rng(masterSeed);
vocab = "word_" + string(1:vocabSize);
testQueries = strings(1, numQueries);
for q = 1:numQueries
    testQueries(q) = join(vocab(randi(vocabSize, 1, randi([2, 5]))), " ");
end

fprintf('=====================================================================\n');
fprintf(' BM25 INDEX REPRESENTATION: INVERTED INDEX vs SPARSE WEIGHT MATRIX\n');
fprintf('=====================================================================\n');
fprintf('V = %d | avg |d| = %d | queries = %d | k1 = %.2f | b = %.2f\n\n', ...
    vocabSize, avgDocLen, numQueries, bm25Params.TermSaturation, bm25Params.LengthNormalization);

% Preallocate (column 1 = inverted, column 2 = matrix)
fitTime    = nan(numSizes, 2);    % seconds
queryTime  = nan(numSizes, 2);    % ms / query, score() only
memMB      = nan(numSizes, 2);    % stored index payload
tokPassSec = nan(numSizes, 1);    % one tokenization pass over the corpus
maxAbsDiff = nan(numSizes, 1);    % max |s_inv - s_mat| over all docs and queries
top10Agree = nan(numSizes, 1);    % % of queries with identical top-10 doc sets

% -------------------------------------------------------------------------
% BENCHMARK LOOP
% -------------------------------------------------------------------------
for s = 1:numSizes
    N = corpusSizes(s);
    fprintf('--> [%d/%d] N = %d documents\n', s, numSizes, N);

    try
        corpus = generateSyntheticCorpus(N, avgDocLen, vocabSize, masterSeed + s);

        % Shared cost reference: one tokenization pass over the corpus
        tic;
        for d = 1:N
            tokenizer.forward(corpus(d));
        end
        tokPassSec(s) = toc;

        % Alternate fit order per size to cancel order bias
        order = [1 2];
        if mod(s, 2) == 0
            order = [2 1];
        end
        nReps = 1;
        if N <= fitRepCutoff
            nReps = fitRepsSmall;
        end

        rankers = cell(1, 2);
        for k = order
            repTimes = zeros(1, nReps);
            for r = 1:nReps
                rankers{k} = buildRanker(strategies(k), tokenizer, bm25Params);
                tic;
                rankers{k}.fit(corpus);
                repTimes(r) = toc;
            end
            fitTime(s, k) = median(repTimes);
            fprintf('    fit %-15s %8.3f s\n', strategies(k), fitTime(s, k));
        end

        % Query latency: score() only, timeit handles warm-up and repetition
        for k = order
            idx = rankers{k}.IndexStrategy;
            perQuery = zeros(1, numQueries);
            for q = 1:numQueries
                qStr = testQueries(q);
                perQuery(q) = timeit(@() idx.score(qStr));
            end
            queryTime(s, k) = median(perQuery) * 1000;
        end

        % Memory of stored index payload (Vocabulary excluded: identical in both)
        invIdx     = rankers{1}.IndexStrategy;
        postings   = invIdx.Postings;   %#ok<NASGU> sized by whos below
        docLengths = invIdx.DocLengths; %#ok<NASGU>
        weightMat  = rankers{2}.IndexStrategy.WeightMatrix; %#ok<NASGU>
        memMB(s, 1) = bytesOf('postings') / 2^20 + bytesOf('docLengths') / 2^20;
        memMB(s, 2) = bytesOf('weightMat') / 2^20;

        % Numerical equivalence on unsorted per-document scores
        diffs = zeros(1, numQueries);
        agree = false(1, numQueries);
        for q = 1:numQueries
            sInv = rankers{1}.IndexStrategy.score(testQueries(q));
            sMat = rankers{2}.IndexStrategy.score(testQueries(q));
            diffs(q) = max(abs(sInv - sMat));
            [~, oInv] = maxk(sInv, 10);
            [~, oMat] = maxk(sMat, 10);
            agree(q) = isempty(setxor(oInv, oMat));
        end
        maxAbsDiff(s) = max(diffs);
        top10Agree(s) = mean(agree) * 100;

    catch ME
        fprintf('   [WARNING] failed at N = %d: %s\n', N, ME.message);
    end

    clear corpus rankers invIdx postings docLengths weightMat sInv sMat;
end

% -------------------------------------------------------------------------
% RESULTS TABLE
% -------------------------------------------------------------------------
fprintf('\n');
fprintf('%-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-8s | %-6s\n', ...
    'N', 'TokPass s', 'Fit Inv s', 'Fit Mat s', 'Q Inv ms', 'Q Mat ms', 'Q Inv/Mat', ...
    'Mem Inv', 'Mem Mat', 'max|ds|', 'Top10');
fprintf('%s\n', repmat('-', 1, 118));
for s = 1:numSizes
    fprintf('%-9d | %-9.3f | %-9.3f | %-9.3f | %-9.4f | %-9.4f | %-9.2f | %-9.2f | %-9.2f | %-8.1e | %5.0f%%\n', ...
        corpusSizes(s), tokPassSec(s), fitTime(s, 1), fitTime(s, 2), ...
        queryTime(s, 1), queryTime(s, 2), queryTime(s, 1) / queryTime(s, 2), ...
        memMB(s, 1), memMB(s, 2), maxAbsDiff(s), top10Agree(s));
end
fprintf('Mem in MB (stored index payload). Query time = median over queries of timeit(score).\n\n');

results = table(corpusSizes(:), tokPassSec, fitTime(:, 1), fitTime(:, 2), ...
    queryTime(:, 1), queryTime(:, 2), memMB(:, 1), memMB(:, 2), maxAbsDiff, top10Agree, ...
    'VariableNames', {'N', 'TokPassSec', 'FitInvSec', 'FitMatSec', 'QueryInvMs', ...
    'QueryMatMs', 'MemInvMB', 'MemMatMB', 'MaxAbsScoreDiff', 'Top10AgreePct'});

% -------------------------------------------------------------------------
% PLOTS (OceanViz MultiComparison on log-log axes)
% -------------------------------------------------------------------------
valid = ~any(isnan([fitTime, queryTime]), 2);
xN    = corpusSizes(valid);

viz = OceanViz.Core('Theme', 'Dark');
viz.createFigure('BM25: Inverted Index vs Sparse Matrix');
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
tl.Title.String = 'BM25 index representation (all other factors fixed)';
tl.Title.Color  = viz.Colors.Text;

panels = { ...
    {fitTime(valid, 1), fitTime(valid, 2), tokPassSec(valid)}, ...
        {'Inverted index', 'Sparse matrix', 'One tokenization pass'}, ...
        'Fit time', 'Seconds'; ...
    {queryTime(valid, 1), queryTime(valid, 2)}, {'Inverted index', 'Sparse matrix'}, ...
        'Query latency (score only)', 'ms / query'; ...
    {memMB(valid, 1), memMB(valid, 2)}, {'Inverted index', 'Sparse matrix'}, ...
        'Stored index memory', 'MB'; ...
    {fitTime(valid, 1) ./ fitTime(valid, 2), queryTime(valid, 1) ./ queryTime(valid, 2)}, ...
        {'Fit ratio (inv / mat)', 'Query ratio (inv / mat)'}, ...
        'Inverted / matrix cost ratio (>1: matrix faster)', 'Ratio'};

for p = 1:size(panels, 1)
    ax = nexttile(tl);
    ys = panels{p, 1};
    names = panels{p, 2};
    data = struct('X', xN, 'Lines', {cell(1, numel(ys))});
    for k = 1:numel(ys)
        data.Lines{k} = struct('Y', ys{k}(:)', 'Name', names{k});
    end
    viz.plot(ax, 'MultiComparison', data, 'Title', panels{p, 3}, ...
        'XLabel', 'Corpus size N', 'YLabel', panels{p, 4}, 'ShowLegend', true);
    set(ax, 'XScale', 'log');
    if p < 4
        set(ax, 'YScale', 'log');
    else
        yline(ax, 1, ':', 'Color', viz.Colors.Text, 'HandleVisibility', 'off');
    end
    legend(ax, 'Location', 'best');
end

% -------------------------------------------------------------------------
% LOCAL FUNCTIONS
% -------------------------------------------------------------------------
function ranker = buildRanker(strategyName, tokenizer, params)
    % Every arm is built from the same tokenizer and BM25 parameters.
    ranker = OceanNLP.Retrieval.BM25Ranker('Strategy', strategyName, ...
        'Tokenizer', tokenizer, ...
        'TermSaturation', params.TermSaturation, ...
        'LengthNormalization', params.LengthNormalization);
end

function b = bytesOf(varName)
    % Bytes of a caller variable as reported by whos (includes container overhead).
    info = evalin('caller', sprintf('whos(''%s'')', varName));
    b = info.bytes;
end

function corpus = generateSyntheticCorpus(N, avgDocLen, vocabSize, seed)
    % Uniform-unigram synthetic corpus; doc length ~ max(5, round(N(avgDocLen, 10^2))).
    rng(seed);
    vocab = "word_" + string(1:vocabSize);
    lens = max(5, round(avgDocLen + randn(1, N) * 10));
    corpus = strings(1, N);
    for i = 1:N
        corpus(i) = join(vocab(randi(vocabSize, 1, lens(i))), " ");
    end
end
