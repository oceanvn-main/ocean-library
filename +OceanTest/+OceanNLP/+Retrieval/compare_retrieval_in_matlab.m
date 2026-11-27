% COMPARE_RETRIEVAL_IN_MATLAB  Run OceanStats/OceanNLP's retrieval system on the
% exact same real Cranfield data (1400 real documents, 25 real queries, real
% human relevance judgments) that fetch_and_compute_retrieval_reference.py fed
% to scikit-learn (TF-IDF+cosine) and a hand-verified numpy BM25/TREC-metrics
% reference, and diff the results.
%
% Run fetch_and_compute_retrieval_reference.py first to produce
% cranfield_validation_data.mat.

function compare_retrieval_in_matlab()
here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(here)));  % .../+OceanTest/+OceanNLP/+Retrieval -> project root
addpath(projectRoot);
data = load(fullfile(here, 'cranfield_validation_data.mat'));

N = double(data.N); V = double(data.V); K = double(data.K);
nQueries = numel(data.query_texts);
fprintf('=== Cranfield: %d real documents, %d real queries (of 225), K=%d ===\n\n', N, nQueries, K);

X_counts = sparse(double(data.X_counts));
TOL = 1e-6;
results = struct('name', {}, 'max_abs_diff', {}, 'pass', {});

% --------------------------------------------------------------------- %
% 1. BM25 ranking: OceanStats.Data.Weighting.BM25Weighting vs the Python
%    Lucene-formula reference, on the REAL 1400-doc corpus.
% --------------------------------------------------------------------- %
bm25 = OceanStats.Data.Weighting.BM25Weighting('K1', data.bm25_K1, 'B', data.bm25_B);
bm25.fit(X_counts);
W_bm25_ocean = bm25.forward(X_counts);

d = abs(full(W_bm25_ocean) - double(data.bm25_weights));
results(end+1) = struct('name', "BM25 weight matrix (1400 real docs) vs Lucene-formula reference", ...
    'max_abs_diff', max(d(:)), 'pass', max(d(:)) < TOL);

% --------------------------------------------------------------------- %
% 2. TF-IDF ranking: OceanStats.Data.Weighting.TfidfWeighting vs sklearn.
% --------------------------------------------------------------------- %
tfidf = OceanStats.Data.Weighting.TfidfWeighting();
tfidf.fit(X_counts);
W_tfidf_ocean = tfidf.forward(X_counts);

d = abs(full(W_tfidf_ocean) - double(data.sklearn_tfidf));
results(end+1) = struct('name', "TF-IDF matrix (1400 real docs) vs scikit-learn TfidfTransformer", ...
    'max_abs_diff', max(d(:)), 'pass', max(d(:)) < TOL);

% --------------------------------------------------------------------- %
% 3. Per-query ranking + all 5 metrics, on REAL relevance judgments.
% --------------------------------------------------------------------- %
cos = OceanStats.Similarity.CosineSimilarity();
metricObjs = { ...
    OceanStats.Metrics.Retrieval.PrecisionAtK('K', K), ...
    OceanStats.Metrics.Retrieval.RecallAtK('K', K), ...
    OceanStats.Metrics.Retrieval.AveragePrecision(), ...
    OceanStats.Metrics.Retrieval.ReciprocalRank(), ...
    OceanStats.Metrics.Retrieval.NDCGAtK('K', K)};
metricPyNames = {'precision@k','recall@k','ap','rr','ndcg@k'};

bm25RankDiff = zeros(nQueries,1);
tfidfRankDiff = zeros(nQueries,1);
bm25MetricDiff = zeros(nQueries, 5);
tfidfMetricDiff = zeros(nQueries, 5);
bm25MetricVals = zeros(nQueries, 5);
tfidfMetricVals = zeros(nQueries, 5);
bm25RankedList = cell(1, nQueries);    % kept for MeanAveragePrecision.aggregate below
tfidfRankedList = cell(1, nQueries);
relevanceList = cell(1, nQueries);

for qi = 1:nQueries
    qcounts = data.query_counts(qi, :);
    rel = data.relevance(qi, :);

    % BM25 ranking
    scoresB = full(W_bm25_ocean * qcounts');
    [~, rankedB] = sort(scoresB, 'descend');
    pyRankedB = double(data.bm25_ranked_idx(qi, :));
    bm25RankDiff(qi) = max(abs(rankedB(:) - pyRankedB(:)));

    % TF-IDF + cosine ranking
    qtfidfRow = qcounts .* tfidf.IDF;
    nrm = norm(qtfidfRow); if nrm > 0, qtfidfRow = qtfidfRow / nrm; end
    scoresT = cos.pairwise(qtfidfRow, full(W_tfidf_ocean));
    [~, rankedT] = sort(scoresT, 'descend');
    pyRankedT = double(data.tfidf_ranked_idx(qi, :));
    tfidfRankDiff(qi) = max(abs(rankedT(:) - pyRankedT(:)));

    bm25RankedList{qi} = rankedB(:)';
    tfidfRankedList{qi} = rankedT(:)';
    relevanceList{qi} = rel;

    % All 5 metrics on both rankings, vs the Python numpy reference
    for mi = 1:5
        vB = metricObjs{mi}.forward(rankedB(:)', rel);
        vT = metricObjs{mi}.forward(rankedT(:)', rel);
        bm25MetricVals(qi,mi) = vB;
        tfidfMetricVals(qi,mi) = vT;
        pyB = data.bm25_metric_vals(qi, mi);
        pyT = data.tfidf_metric_vals(qi, mi);
        if isnan(vB) && isnan(pyB), bm25MetricDiff(qi,mi) = 0; else, bm25MetricDiff(qi,mi) = abs(vB - pyB); end
        if isnan(vT) && isnan(pyT), tfidfMetricDiff(qi,mi) = 0; else, tfidfMetricDiff(qi,mi) = abs(vT - pyT); end
    end
end

results(end+1) = struct('name', "BM25 per-query ranking (25 real queries) vs Python", ...
    'max_abs_diff', max(bm25RankDiff), 'pass', max(bm25RankDiff) == 0);
results(end+1) = struct('name', "TF-IDF+cosine per-query ranking (25 real queries) vs Python", ...
    'max_abs_diff', max(tfidfRankDiff), 'pass', max(tfidfRankDiff) == 0);

for mi = 1:5
    results(end+1) = struct('name', sprintf("%s on BM25 rankings vs real Cranfield qrels (Python numpy ref)", metricPyNames{mi}), ...
        'max_abs_diff', max(bm25MetricDiff(:,mi)), 'pass', max(bm25MetricDiff(:,mi)) < TOL);
    results(end+1) = struct('name', sprintf("%s on TF-IDF rankings vs real Cranfield qrels (Python numpy ref)", metricPyNames{mi}), ...
        'max_abs_diff', max(tfidfMetricDiff(:,mi)), 'pass', max(tfidfMetricDiff(:,mi)) < TOL);
end

% --------------------------------------------------------------------- %
% 4. MAP (IIR eq. 11.16) via MeanAveragePrecision.aggregate() on the REAL
%    Cranfield rankings, vs. the Python reference: nanmean of the per-query
%    AP column (metric index 3 = 'ap') computed independently in numpy.
% --------------------------------------------------------------------- %
mapMetric = OceanStats.Metrics.Retrieval.MeanAveragePrecision();
[mapB, ~, exclB] = mapMetric.aggregate(bm25RankedList, relevanceList);
[mapT, ~, exclT] = mapMetric.aggregate(tfidfRankedList, relevanceList);
pyMapB = mean(data.bm25_metric_vals(:, 3), 'omitnan');
pyMapT = mean(data.tfidf_metric_vals(:, 3), 'omitnan');
results(end+1) = struct('name', sprintf("MAP on BM25 rankings (real Cranfield, %d excluded) = %.4f", exclB, mapB), ...
    'max_abs_diff', abs(mapB - pyMapB), 'pass', abs(mapB - pyMapB) < TOL);
results(end+1) = struct('name', sprintf("MAP on TF-IDF rankings (real Cranfield, %d excluded) = %.4f", exclT, mapT), ...
    'max_abs_diff', abs(mapT - pyMapT), 'pass', abs(mapT - pyMapT) < TOL);

% --------------------------------------------------------------------- %
% Report
% --------------------------------------------------------------------- %
fprintf('=== Comparison report (tolerance %.0e where applicable) ===\n', TOL);
allPass = true;
for k = 1:numel(results)
    r = results(k);
    if r.pass, status = "PASS"; else, status = "FAIL"; allPass = false; end
    fprintf('  [%s]  %-70s  max=%.3e\n', status, r.name, r.max_abs_diff);
end

fprintf('\n--- Mean metrics over %d real Cranfield queries (OceanStats/OceanNLP, NaN-omitted) ---\n', nQueries);
fprintf('%-12s %10s %12s\n', 'metric', 'BM25', 'TF-IDF+cos');
for mi = 1:5
    fprintf('%-12s %10.4f %12.4f\n', metricPyNames{mi}, ...
        mean(bm25MetricVals(:,mi), 'omitnan'), mean(tfidfMetricVals(:,mi), 'omitnan'));
end

if allPass
    fprintf('\nOverall: ALL CHECKS PASSED\n');
else
    fprintf('\nOverall: SOME CHECKS FAILED\n');
end
end
