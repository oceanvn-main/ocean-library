% COMPARE_IN_MATLAB  Run OceanStats/OceanNLP on the exact same count matrix
% that fetch_and_compute_reference.py fed to production Python libraries
% (scikit-learn for TF-IDF + cosine similarity, a direct numpy reproduction
% of the Elasticsearch/Lucene BM25 formula, and rank_bm25 for the classic
% Okapi variant shown for context), and diff the results.
%
% Run fetch_and_compute_reference.py first to produce nlp_validation_data.mat.

%% Setup Path & Load Data
workspaceRoot = 'C:\Ocean Library\Matlab code practice';
if ~contains(path, workspaceRoot)
    addpath(workspaceRoot);
end

here = fileparts(mfilename('fullpath'));
% fetch_and_compute_reference.py writes results/nlp_validation_data.mat; a copy may also sit next to this script.
candidates = {fullfile(here, 'results', 'nlp_validation_data.mat'), fullfile(here, 'nlp_validation_data.mat')};
matFile = candidates{find(cellfun(@(f) exist(f, 'file') == 2, candidates), 1)};
if isempty(matFile)
    error('compare_in_matlab:NoReference', ...
        'nlp_validation_data.mat not found. Run fetch_and_compute_reference.py first (looked in %s and %s).', candidates{:});
end
data = load(matFile);
need = {'X_counts', 'sklearn_tfidf', 'sklearn_idf', 'sklearn_cosine', 'K1', 'B', 'bm25_lucene_manual', 'bm25_lucene_idf', ...
        'query_counts', 'bm25_lucene_query_scores', 'rank_bm25_query_scores', 'query_text'};
missing = need(~isfield(data, need));
if ~isempty(missing)
    error('compare_in_matlab:IncompleteReference', '%s lacks: %s. Re-run fetch_and_compute_reference.py.', matFile, strjoin(missing, ', '));
end

results = struct('name', {}, 'max_abs_diff', {}, 'mean_abs_diff', {}, 'pass', {});
TOL = 1e-6;
%%. Download and extract the raw text files

url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz';
archive = '20news-bydate.tar.gz';
dataDir = '20news_raw_data';
if ~exist(archive, 'file')
    disp('Downloading 20 Newsgroups raw dataset (14 MB)...');
    websave(archive, url);
end
if ~exist(dataDir, 'dir')
    disp('Extracting archive...');
    untar(archive, dataDir);
end
% Create a datastore to crawl the folders
% The dataset is organized into folders (one folder per category)
trainPath = fullfile(dataDir, '20news-bydate-train');
% fileDatastore crawls the directory. We tell it to use `fileread` to read text.
fds = fileDatastore(trainPath, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '', ... % The files don't have extensions
    'ReadFcn', @fileread);
%Extract the category labels from the folder names
% The folder name immediately after '20news-bydate-train\' is the category
labels = extractBetween(fds.Files, "20news-bydate-train\", "\");
labels = categorical(labels);

% Load the raw text into memory
% (For 11,000 documents, this easily fits in RAM as a string array)
disp('Reading text files into memory...');
corpus = strings(numel(fds.Files), 1);
for i = 1:numel(fds.Files)
    corpus(i) = string(read(fds));
end
disp("Successfully loaded " + numel(corpus) + " raw documents.");
disp("Categories found: " + numel(categories(labels)));

%% Vectorizer on the full corpus (timing / sanity; NOT compared with the reference below)
% The sklearn reference was computed on a different, small matrix (data.X_counts: 4 categories, English stop words removed,
% top-300 words), so the weighting checks below start from THAT matrix, exactly as fetch_and_compute_reference.py fed it to
% scikit-learn. This section only exercises OceanNLP's CountVectorizer on all documents.
% 1. Instantiate the CountVectorizer
vectorizer = OceanNLP.FeatureExtraction.CountVectorizer();

% 2. Fit the vocabulary on your corpus, 3. transform raw text into the (N x V) sparse word-count matrix
tic; vectorizer.fit(corpus);           tFit = toc;
tic; X_counts_full = vectorizer.forward(corpus);  tFwd = toc;
fprintf('CountVectorizer on %d documents: fit %.1fs, forward %.1fs, size %s, nnz %d\n', ...
    numel(corpus), tFit, tFwd, mat2str(size(X_counts_full)), nnz(X_counts_full));

% The same counts the Python side used: identical input for every weighting comparison below.
X_counts = sparse(double(data.X_counts));

%% --------------------------------------------------------------------- %
% 1. TF-IDF: OceanStats.Data.Weighting.TfidfWeighting vs. sklearn
% --------------------------------------------------------------------- %
tfidf = OceanStats.Data.Weighting.TfidfWeighting.create();  % defaults: Norm="l2", matches sklearn norm="l2"
tfidf.fit(X_counts);
W_tfidf_ocean = full(tfidf.forward(X_counts));
W_tfidf_sklearn = double(data.sklearn_tfidf);

d = abs(W_tfidf_ocean - W_tfidf_sklearn);
results(end+1) = struct('name', "TF-IDF vs scikit-learn TfidfTransformer", ...
    'max_abs_diff', max(d(:)), 'mean_abs_diff', mean(d(:)), 'pass', max(d(:)) < TOL);

idf_diff = max(abs(tfidf.IDF(:) - data.sklearn_idf(:)));
fprintf('TF-IDF idf vector max abs diff vs sklearn: %.3e\n', idf_diff);

%% --------------------------------------------------------------------- %
% 2. Cosine similarity: OceanStats.Similarity.CosineSimilarity vs sklearn
%    (on the OceanStats-computed TF-IDF matrix, to isolate this pillar)
% --------------------------------------------------------------------- %
cos_sim = OceanStats.Similarity.CosineSimilarity();
P_ocean = cos_sim.pairwise(W_tfidf_ocean, W_tfidf_ocean);
P_sklearn = double(data.sklearn_cosine);

d = abs(P_ocean - P_sklearn);
results(end+1) = struct('name', "Cosine similarity vs scikit-learn cosine_similarity", ...
    'max_abs_diff', max(d(:)), 'mean_abs_diff', mean(d(:)), 'pass', max(d(:)) < TOL);

%% --------------------------------------------------------------------- %
% 3. BM25 weight matrix: OceanStats.Data.Weighting.BM25Weighting vs the
%    Elasticsearch/Lucene-formula numpy reference (the variant our class
%    actually implements -- same K1/B as the Python side).
% --------------------------------------------------------------------- %
bm25 = OceanStats.Data.Weighting.BM25Weighting('K1', double(data.K1), 'B', double(data.B));
bm25.fit(X_counts);
W_bm25_ocean = full(bm25.forward(X_counts));
W_bm25_lucene = double(data.bm25_lucene_manual);

d = abs(W_bm25_ocean - W_bm25_lucene);
results(end+1) = struct('name', "BM25 weights vs Lucene/Elasticsearch-formula numpy reference", ...
    'max_abs_diff', max(d(:)), 'mean_abs_diff', mean(d(:)), 'pass', max(d(:)) < TOL);

idf_diff_bm25 = max(abs(bm25.IDF(:) - data.bm25_lucene_idf(:)));
fprintf('BM25 idf vector max abs diff vs Lucene-formula reference: %.3e\n', idf_diff_bm25);

%% --------------------------------------------------------------------- %
% 4. BM25 query ranking: score = W * queryCounts' (as BM25Weighting's own
%    docstring prescribes), vs the same computation done in numpy.
% --------------------------------------------------------------------- %
query_counts = double(data.query_counts(:));
scores_ocean = W_bm25_ocean * query_counts;
scores_lucene = double(data.bm25_lucene_query_scores(:));

d = abs(scores_ocean - scores_lucene);
results(end+1) = struct('name', "BM25 query scores vs Lucene-formula numpy reference", ...
    'max_abs_diff', max(d(:)), 'mean_abs_diff', mean(d(:)), 'pass', max(d(:)) < TOL);

[~, top_ocean] = sort(scores_ocean, 'descend');
[~, top_lucene] = sort(scores_lucene, 'descend');
fprintf('\nQuery: %s\n', data.query_text);
fprintf('OceanStats BM25 top-3 doc idx (0-based to match Python): %s\n', mat2str(top_ocean(1:3)'-1));
fprintf('Lucene-ref  BM25 top-3 doc idx (0-based to match Python): %s\n', mat2str(top_lucene(1:3)'-1));

% For CONTEXT ONLY -- rank_bm25's classic Robertson formula (k1=1.5) is a
% documented different variant and is EXPECTED to diverge numerically from
% our Lucene-style (k1=1.2) implementation. Not scored pass/fail.
scores_rank_bm25 = double(data.rank_bm25_query_scores(:));
d_context = abs(scores_ocean - scores_rank_bm25);
fprintf('\n[context only, not a pass/fail check] OceanStats (Lucene k1=1.2) vs rank_bm25 (classic k1=1.5):\n');
fprintf('  max abs diff = %.4f (expected to be large -- different BM25 idf formula families)\n', max(d_context));
[~, top_rank_bm25] = sort(scores_rank_bm25, 'descend');
fprintf('  rank_bm25 top-3 doc idx (0-based): %s\n', mat2str(top_rank_bm25(1:3)'-1));

%% --------------------------------------------------------------------- %
% Report
% --------------------------------------------------------------------- %
fprintf('\n=== Comparison report (tolerance %.0e) ===\n', TOL);
allPass = true;
for k = 1:numel(results)
    r = results(k);
    status = "PASS";
    if ~r.pass, status = "FAIL"; allPass = false; end
    fprintf('  [%s]  %-55s  max=%.3e  mean=%.3e\n', status, r.name, r.max_abs_diff, r.mean_abs_diff);
end
if allPass
    fprintf('\nOverall: ALL CHECKS PASSED\n');
else
    fprintf('\nOverall: SOME CHECKS FAILED\n');
end

%% --------------------------------------------------------------------- %
% Visualization: Cross-Validation Grid Search Heatmap (if CV data present)
% --------------------------------------------------------------------- %
if isfield(data, 'cv_grid_scores') && ~isempty(data.cv_grid_scores)
    figure('Name', 'BM25 Hyperparameter Tuning Grid Search', 'Color', 'w');
    imagesc(data.k1_vals, data.b_vals, data.cv_grid_scores');
    axis xy;
    colorbar;
    title('BM25 5-Fold Cross-Validation Precision@5 Grid (k1 vs b)');
    xlabel('k1 (Term Frequency Saturation)');
    ylabel('b (Document Length Normalization)');
    hold on;
    plot(data.cv_best_k1, data.cv_best_b, 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r', ...
        'DisplayName', sprintf('Optimal (k1=%.2f, b=%.2f)', double(data.cv_best_k1), double(data.cv_best_b)));
    legend('Location', 'northeast');
    grid on;
end
