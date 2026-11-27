% COMPARE_DENSE_IN_MATLAB  Modern (DENSE) retrieval workflow with OceanNLP, diffed against the Python dense reference.
%
% Companion of compare_in_matlab.m (which validates the classic SPARSE workflow: counts -> TF-IDF / BM25). Here the text goes
% through neural encoders instead, all implemented in OceanNLP (pure MATLAB, no toolbox):
%
%   bi-encoder     OceanNLP.Embedding.BertEncoder + OceanNLP.Retrieval.BiEncoderRanker   (all-MiniLM-L6-v2: mean pool, L2 normalise)
%   cross-encoder  OceanNLP.Embedding.CrossEncoder                                       (ms-marco-TinyBERT-L-2-v2, raw logits)
%   ColBERT        OceanNLP.Embedding.ColBERTEncoder + OceanNLP.Retrieval.ColBERTRanker  (answerai-colbert-small-v1, MaxSim)
%
% TWO references are used:
%   (A) Cranfield, 1,400 documents, 25 queries: neural_reference_data.mat (PyTorch + transformers), covering bi-encoder,
%       cross-encoder and ColBERT (sections 1-3 below).
%   (B) A small 20 Newsgroups subset with the STANDARD library, sentence-transformers: results/nlp_dense_validation_data.mat from
%       fetch_and_compute_dense_reference.py (section 4): SentenceTransformer.encode, .similarity, util.semantic_search.
%
% REFERENCE = what the Python side (PyTorch + transformers, same checkpoints) produced for the SAME texts:
% neural_reference_data.mat, made by neural_reference.py from the weights exported by export_neural_checkpoints.py; both live in
% +OceanTest/+OceanNLP/+Retrieval. This script only READS them; nothing is recomputed in Python here.
%
% OUTPUT = the DIFFERENCE, per component: max / mean absolute difference, the worst position, cosine similarity of embeddings,
% and how ranked lists differ. A component "passes" when its max abs difference is within tolerance. The table is printed and
% written to results/dense_comparison.csv; per-query score differences are plotted to results/dense_score_differences.png.
%
% Set QUICK = true to compare only the first QUICK_DOCS documents / QUICK_QUERIES queries (minutes -> about a minute); rankings
% are skipped then, because they need the whole corpus.
%
% Run:  run('compare_dense_in_matlab.m')      (about 5-10 minutes for the full 1,400-document Cranfield corpus)

%% Setup
QUICK = false;  QUICK_DOCS = 120;  QUICK_QUERIES = 5;

workspaceRoot = 'C:\Ocean Library\Matlab code practice';
if ~contains(path, workspaceRoot), addpath(workspaceRoot); end
here = fileparts(mfilename('fullpath'));
retDir = fullfile(here, '..', '..', '..', '+Retrieval');           % +OceanTest/+OceanNLP/+Retrieval
refFile = fullfile(retDir, 'neural_reference_data.mat');
needFiles = {refFile, fullfile(retDir, 'bienc_weights.mat'), fullfile(retDir, 'bienc_vocab.txt'), fullfile(retDir, 'cross_weights.mat'), ...
             fullfile(retDir, 'cross_vocab.txt'), fullfile(retDir, 'colbert_weights.mat'), fullfile(retDir, 'colbert_vocab.txt')};
missing = needFiles(~cellfun(@(f) exist(f, 'file') == 2, needFiles));
if ~isempty(missing)
    error('compare_dense_in_matlab:MissingReference', ['Missing: %s\nRun export_neural_checkpoints.py and neural_reference.py ' ...
        '(in +OceanTest/+OceanNLP/+Retrieval) first.'], strjoin(string(missing), newline));
end
ref = load(refFile);
load_ = @(tag) OceanTest.OceanNLP.Retrieval.loadCheckpointWeights(fullfile(retDir, tag + "_weights.mat"));
vocab_ = @(tag) string(fullfile(retDir, tag + "_vocab.txt"));

allTexts = string(ref.sub_texts(:))';
allQueries = string(ref.query_texts(:))';
nd = numel(allTexts);  nq = numel(allQueries);
if QUICK
    ndUse = min(QUICK_DOCS, nd);  nqUse = min(QUICK_QUERIES, nq);
else
    ndUse = nd;  nqUse = nq;
end
docs = allTexts(1:ndUse);  queries = allQueries(1:nqUse);
fullRun = ndUse == nd;
fprintf('=== Dense workflow vs Python reference: %d documents, %d queries (Cranfield)%s ===\n', ndUse, nqUse, ternary_(QUICK, '  [QUICK mode]', ''));
fprintf('reference: %s\n           %s\n\n', string(ref.bi_source), string(ref.cb_source));

rows = struct('component', {}, 'quantity', {}, 'max_abs_diff', {}, 'mean_abs_diff', {}, 'tolerance', {}, 'pass', {}, 'worst_at', {});
ranks = struct('component', {}, 'queries', {}, 'positions_differing', {}, 'of', {}, 'max_ref_score_gap', {}, 'mean_topk_overlap', {});
scoreDiffs = struct();

%% 1. Bi-encoder (all-MiniLM-L6-v2)
enc = OceanNLP.Embedding.BertEncoder('Weights', load_("bienc"), 'VocabFile', vocab_("bienc"), ...
    'ModelName', "all-MiniLM-L6-v2", 'NormalizeOutput', true, 'MaxLength', 256);
bi = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', enc);
t0 = tic; bi.fit(docs); fprintf('bi-encoder: encoded %d documents in %.1fs\n', ndUse, toc(t0));
Dref = double(ref.bi_doc(1:ndUse, :));
Dmine = bi.CorpusMatrix;
rows(end+1) = row_("bi-encoder", "document embeddings (N x 384)", abs(Dmine - Dref), 1e-4, @(k) posText_(k, size(Dref), ["doc" "dim"]));
cosDoc = sum(Dmine .* Dref, 2) ./ (vecnorm(Dmine, 2, 2) .* vecnorm(Dref, 2, 2));
qMine = enc.forward(queries);
qRef = double(ref.bi_q(1:nqUse, :));
rows(end+1) = row_("bi-encoder", "query embeddings (Q x 384)", abs(qMine - qRef), 1e-4, @(k) posText_(k, size(qRef), ["query" "dim"]));
Smine = qMine * Dmine';                                   % dense scores: dot product of normalised vectors
Sref = qRef * Dref';
rows(end+1) = row_("bi-encoder", "query-document scores (Q x N)", abs(Smine - Sref), 1e-4, @(k) posText_(k, size(Sref), ["query" "doc"]));
scoreDiffs.bi = max(abs(Smine - Sref), [], 2);
fprintf('bi-encoder: min cosine between my and the reference document embedding = %.8f\n', min(cosDoc));
if fullRun, ranks(end+1) = rankAgreement_("bi-encoder", Smine, Sref); end

%% 2. Cross-encoder (ms-marco-TinyBERT-L-2-v2): raw logits on the reference's BM25 top-20 pairs
ce = OceanNLP.Embedding.CrossEncoder('Weights', load_("cross"), 'VocabFile', vocab_("cross"), ...
    'ModelName', "ms-marco-TinyBERT-L-2-v2", 'MaxLength', 512);
topTexts = string(ref.top_doc_texts);
nqCe = min(nqUse, size(topTexts, 1));
ceMine = zeros(nqCe, size(topTexts, 2));
t0 = tic;
for qi = 1:nqCe, ceMine(qi, :) = ce.scoreDocs(queries(qi), topTexts(qi, :)); end
fprintf('cross-encoder: scored %d query-document pairs in %.1fs\n', numel(ceMine), toc(t0));
ceRef = double(ref.ce_scores(1:nqCe, :));
rows(end+1) = row_("cross-encoder", "relevance logits (Q x 20)", abs(ceMine - ceRef), 1e-3, @(k) posText_(k, size(ceRef), ["query" "rank-in-BM25-top-20"]));
ceOrder = 0;
for qi = 1:nqCe
    [~, a] = sort(ceMine(qi, :), 'descend');  [~, b] = sort(ceRef(qi, :), 'descend');
    ceOrder = ceOrder + nnz(a ~= b);
end
ranks(end+1) = struct('component', "cross-encoder (rerank of 20)", 'queries', nqCe, 'positions_differing', ceOrder, 'of', numel(ceMine), ...
    'max_ref_score_gap', NaN, 'mean_topk_overlap', NaN);

%% 3. ColBERT (answerai-colbert-small-v1): token embeddings + MaxSim
cb = OceanNLP.Embedding.ColBERTEncoder('Weights', load_("colbert"), 'VocabFile', vocab_("colbert"), 'ModelName', "answerai-colbert-small-v1");
qTokRef = double(ref.cb_q_tok(1:nqUse, :, :));
QDiff = zeros(nqUse, 1);  Qmine = cell(1, nqUse);
for qi = 1:nqUse
    Qmine{qi} = cb.encodeQuery(queries(qi));
    if ~isequal(size(Qmine{qi}), [size(qTokRef, 2) size(qTokRef, 3)])
        error('ColBERT query shape %s differs from the reference %s', mat2str(size(Qmine{qi})), mat2str(size(qTokRef)));
    end
    QDiff(qi) = max(abs(Qmine{qi} - squeeze(qTokRef(qi, :, :))), [], 'all');
end
rows(end+1) = row_("ColBERT", "query token embeddings (Q x 32 x 96)", QDiff, 1e-4, @(k) sprintf('query %d', k));
cbRanker = OceanNLP.Retrieval.ColBERTRanker('Encoder', cb);
t0 = tic; cbRanker.fit(docs); fprintf('ColBERT: encoded %d documents in %.1fs\n', ndUse, toc(t0));
lens = double(ref.cb_d_lens(1:ndUse));  lens = lens(:)';  offs = [0 cumsum(lens)];
dRef = double(ref.cb_d_tok_concat(1:offs(end), :));
DDiff = zeros(ndUse, 1);  tokMismatch = 0;
for j = 1:ndUse
    Dm = double(cbRanker.DocMatrices{j});
    if size(Dm, 1) ~= lens(j), tokMismatch = tokMismatch + 1; continue; end
    DDiff(j) = max(abs(Dm - dRef(offs(j)+1:offs(j+1), :)), [], 'all');
end
rows(end+1) = row_("ColBERT", "document token embeddings (single precision)", DDiff, 1e-4, @(k) sprintf('doc %d', k));
rows(end+1) = row_("ColBERT", "documents whose token count differs from the reference", tokMismatch, 0, @(k) '-');
MSmine = zeros(nqUse, ndUse);
for qi = 1:nqUse, MSmine(qi, :) = cbRanker.Interaction.score(Qmine{qi}, cbRanker.DocMatrices); end
MSref = double(ref.cb_maxsim(1:nqUse, 1:ndUse));
rows(end+1) = row_("ColBERT", "MaxSim scores (Q x N)", abs(MSmine - MSref), 1e-3, @(k) posText_(k, size(MSref), ["query" "doc"]));
scoreDiffs.cb = max(abs(MSmine - MSref), [], 2);
if fullRun, ranks(end+1) = rankAgreement_("ColBERT", MSmine, MSref); end

%% 4. Standard-library reference: sentence-transformers on a small 20 Newsgroups subset
% fetch_and_compute_dense_reference.py records everything needed to reproduce its setting (max_seq_length, pooling,
% normalisation), so the encoder is configured from the file, not from assumptions.
stFile = fullfile(here, 'results', 'nlp_dense_validation_data.mat');
if ~isfile(stFile)
    warning('compare_dense_in_matlab:NoSentenceTransformersReference', ...
        'Section 4 skipped: %s not found. Run fetch_and_compute_dense_reference.py first.', stFile);
else
    st = load(stFile);
    needSt = {'X_dense', 'q_dense', 'dense_cosine_matrix', 'dense_query_scores', 'top_k_idx', 'top_k_scores', 'docs', 'query_texts', ...
              'framework', 'model_name', 'max_seq_length', 'pooling', 'normalize_embeddings'};
    missingSt = needSt(~isfield(st, needSt));
    if ~isempty(missingSt)
        error('compare_dense_in_matlab:StaleReference', ['%s is an older reference (no %s). Re-run ' ...
            'fetch_and_compute_dense_reference.py (needs: py -m pip install sentence-transformers).'], stFile, strjoin(missingSt, ', '));
    end
    fprintf('\nsentence-transformers reference: %s | %s | max_seq_length %d | pooling %s | %d truncated docs\n', ...
        string(st.framework), string(st.model_name), st.max_seq_length, string(st.pooling), st.n_truncated_docs);
    stDocs = string(st.docs(:))';  stQueries = string(st.query_texts(:))';
    encSt = OceanNLP.Embedding.BertEncoder('Weights', load_("bienc"), 'VocabFile', vocab_("bienc"), ...
        'ModelName', "all-MiniLM-L6-v2", 'NormalizeOutput', logical(st.normalize_embeddings), 'MaxLength', double(st.max_seq_length));
    encSt.fit(stDocs);                                           % documented no-op: pretrained weights, only marks the encoder ready
    t0 = tic;  Xst = encSt.forward(stDocs);  Qst = encSt.forward(stQueries);
    fprintf('encoded %d documents and %d queries in %.1fs\n', numel(stDocs), numel(stQueries), toc(t0));
    XstRef = double(st.X_dense);  QstRef = double(st.q_dense);
    cmp = 'sentence-transformers';
    rows(end+1) = row_(cmp, "document embeddings (N x D)", abs(Xst - XstRef), 1e-4, @(k) posText_(k, size(XstRef), ["doc" "dim"]));
    rows(end+1) = row_(cmp, "query embeddings (Q x D)", abs(Qst - QstRef), 1e-4, @(k) posText_(k, size(QstRef), ["query" "dim"]));
    CosMine = Xst * Xst';
    rows(end+1) = row_(cmp, "document-document cosine matrix (N x N)", abs(CosMine - double(st.dense_cosine_matrix)), 1e-4, ...
        @(k) posText_(k, size(CosMine), ["doc" "doc"]));
    QSmine = Qst * Xst';
    QSref = double(st.dense_query_scores);
    rows(end+1) = row_(cmp, "query-document scores (Q x N)", abs(QSmine - QSref), 1e-4, @(k) posText_(k, size(QSref), ["query" "doc"]));
    % top-k vector search (util.semantic_search): my ranking read against the reference's own scores
    K = size(st.top_k_idx, 2);  nqSt = size(QSmine, 1);
    posDiff = 0;  gap = 0;  overlap = zeros(nqSt, 1);
    for qi = 1:nqSt
        [~, order] = sort(QSmine(qi, :), 'descend');
        mineK = order(1:K);  refK = double(st.top_k_idx(qi, :)) + 1;            % reference ids are 0-based
        posDiff = posDiff + nnz(mineK ~= refK);
        gap = max(gap, max(abs(QSref(qi, mineK) - double(st.top_k_scores(qi, :)))));
        overlap(qi) = numel(intersect(mineK, refK)) / K;
    end
    ranks(end+1) = struct('component', "sentence-transformers top-" + K, 'queries', nqSt, 'positions_differing', posDiff, 'of', nqSt * K, ...
        'max_ref_score_gap', gap, 'mean_topk_overlap', mean(overlap));
end

%% Report: the differences
T = struct2table(rows);
T.pass = categorical(T.pass, [true false], ["PASS" "FAIL"]);
fprintf('\n=== Difference vs Python dense reference (max / mean absolute difference) ===\n');
for k = 1:height(T)
    fprintf('  [%s]  %-14s %-52s max=%.3e  mean=%.3e  (tol %.0e)  worst at %s\n', string(T.pass(k)), T.component(k), T.quantity(k), ...
        T.max_abs_diff(k), T.mean_abs_diff(k), T.tolerance(k), T.worst_at(k));
end
if ~isempty(ranks)
    fprintf('\n=== Ranked lists: mine vs reference (near-duplicate documents tie within float noise and can swap) ===\n');
    for k = 1:numel(ranks)
        r = ranks(k);
        fprintf('  %-30s %3d queries: %6d of %7d rank positions differ', r.component, r.queries, r.positions_differing, r.of);
        if ~isnan(r.max_ref_score_gap), fprintf(' | max score gap read off the reference order = %.2e | mean top-k overlap = %.3f', r.max_ref_score_gap, r.mean_topk_overlap); end
        fprintf('\n');
    end
end
allPass = all(T.pass == "PASS");
if allPass
    fprintf('\nOverall: ALL DENSE CHECKS PASSED (differences are float rounding, not model differences)\n');
else
    fprintf('\nOverall: SOME DENSE CHECKS FAILED\n');
end

%% Save + plot
outDir = fullfile(here, 'results');
if ~isfolder(outDir), mkdir(outDir); end
writetable(T, fullfile(outDir, 'dense_comparison.csv'));
fprintf('wrote %s\n', fullfile(outDir, 'dense_comparison.csv'));
% the plot is a convenience: it must never block or fail the comparison, so it is timed step by step and guarded
stamp_ = @(msg) fprintf('[%s] %s\n', char(datetime('now', 'Format', 'HH:mm:ss')), msg);
try
    stamp_('plot: creating figure');
    fig = figure('Name', 'Dense workflow: per-query max |score difference| vs Python', 'Color', 'w', 'Visible', 'off');
    semilogy(1:nqUse, max(scoreDiffs.bi, eps), 'o-', 'DisplayName', 'bi-encoder cosine score'); hold on;
    semilogy(1:nqUse, max(scoreDiffs.cb, eps), 's-', 'DisplayName', 'ColBERT MaxSim score');
    xlabel('query'); ylabel('max |mine - Python| over documents'); grid on; legend('Location', 'best');
    title('OceanNLP dense retrieval vs PyTorch reference (float rounding level)');
    stamp_('plot: exporting png');
    exportgraphics(fig, fullfile(outDir, 'dense_score_differences.png'), 'Resolution', 150);
    close(fig);
    stamp_(['wrote ' fullfile(outDir, 'dense_score_differences.png')]);
catch plotErr
    warning('compare_dense_in_matlab:PlotFailed', 'plot skipped: %s', plotErr.message);
end
stamp_('done');

%% Local functions
function r = row_(component, quantity, D, tol, worstFcn)
% one row of the difference table; D = absolute difference array, worstFcn maps a linear index to a readable position
[mx, k] = max(D(:));
r = struct('component', string(component), 'quantity', string(quantity), 'max_abs_diff', mx, 'mean_abs_diff', mean(D(:)), ...
    'tolerance', tol, 'pass', mx <= tol, 'worst_at', string(worstFcn(k)));
end

function s = ternary_(c, a, b)
if c, s = a; else, s = b; end
end

function t = posText_(k, sz, names)
[i, j] = ind2sub(sz(1:2), k);
t = sprintf('%s %d, %s %d', names(1), i, names(2), j);
end

function r = rankAgreement_(name, Smine, Sref)
% my ordering vs the reference ordering of the same score matrix; ties within float noise are judged on the REFERENCE scores
nq = size(Smine, 1);  nd = size(Smine, 2);
pos = 0;  gap = 0;  ov = zeros(nq, 1);
for qi = 1:nq
    [~, a] = sort(Smine(qi, :), 'descend');
    [refSorted, b] = sort(Sref(qi, :), 'descend');
    pos = pos + nnz(a ~= b);
    gap = max(gap, max(abs(Sref(qi, a) - refSorted)));
    ov(qi) = numel(intersect(a(1:10), b(1:10))) / 10;
end
r = struct('component', string(name), 'queries', nq, 'positions_differing', pos, 'of', nq * nd, 'max_ref_score_gap', gap, 'mean_topk_overlap', mean(ov));
end
