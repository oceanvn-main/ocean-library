function compare_neural_in_matlab()
% COMPARE_NEURAL_IN_MATLAB  External validation + benchmark for the neural
% retrieval models (neural-retrieval addendum §8 b/c/d/e), on REAL Cranfield.
%
% Needs (all in this folder): {bienc,cross,colbert}_{weights.mat,vocab.txt}
% from export_neural_checkpoints.py and neural_reference_data.mat from
% neural_reference.py. Run:  OceanTest.OceanNLP.Retrieval.compare_neural_in_matlab()
%
% PASS = agreement with the Python reference (same inputs both sides), not
% retrieval quality. The benchmark table is a separate, secondary output
% (zero-shot checkpoints on a 1960s aerospace collection; 25 queries only, so
% differences between systems are small-sample and not significance-tested).

here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
ref = load(fullfile(here, 'neural_reference_data.mat'));
load_ = @(tag) OceanTest.OceanNLP.Retrieval.loadCheckpointWeights(fullfile(here, tag + "_weights.mat"));
vocab_ = @(tag) string(fullfile(here, tag + "_vocab.txt"));

subTexts = string(ref.sub_texts(:))';
queries = string(ref.query_texts(:))';
nq = numel(queries);  nd = numel(subTexts);
rel = double(ref.relevance_sub);
relCells = arrayfun(@(i) rel(i, :), 1:nq, 'UniformOutput', false);
fprintf('=== Cranfield (full corpus): %d docs, %d queries ===\n', nd, nq);
fprintf('reference sources:\n  bi-encoder: %s\n  ColBERT:    %s\n\n', string(ref.bi_source), string(ref.cb_source));

results = struct('name', {}, 'value', {}, 'tol', {}, 'pass', {});
    function add(name, value, tol)
        results(end+1) = struct('name', name, 'value', value, 'tol', tol, 'pass', value <= tol);
    end
lat = struct(); idxBytes = struct(); metrics = struct();
metricObjs = {OceanStats.Metrics.Retrieval.PrecisionAtK('K', 10), ...
              OceanStats.Metrics.Retrieval.AveragePrecision(), ...
              OceanStats.Metrics.Retrieval.NDCGAtK('K', 10)};

% ------------------------------------------------------------------ %
% (b) bi-encoder: all-MiniLM-L6-v2 (mean pool + L2 normalize, 256 tokens)
% ------------------------------------------------------------------ %
enc = OceanNLP.Embedding.BertEncoder('Weights', load_("bienc"), 'VocabFile', vocab_("bienc"), ...
    'ModelName', "all-MiniLM-L6-v2", 'NormalizeOutput', true, 'MaxLength', 256);
bi = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', enc);
nTok = arrayfun(@(t) numel(enc.Tokenizer.forward(t)) + 2, subTexts);
fprintf('truncation coverage: %d/%d docs exceed 256 tokens, %d exceed 300 (bi-encoder / ColBERT caps)\n', nnz(nTok > 256), nd, nnz(nTok > 300));
t0 = tic; bi.fit(subTexts); tFit = toc(t0);
fprintf('bi-encoder: encoded %d docs in %.1fs (%.2fs/doc)\n', nd, tFit, tFit / nd);
add("bi-encoder doc embeddings vs reference (max abs)", max(abs(bi.CorpusMatrix - double(ref.bi_doc)), [], 'all'), 1e-4);
qEmb = enc.forward(queries);
add("bi-encoder query embeddings vs reference (max abs)", max(abs(qEmb - double(ref.bi_q)), [], 'all'), 1e-4);
orderGap = 0; orderExact = 0;
for qi = 1:nq
    [mine, ~] = bi.rank(queries(qi));
    refS = double(ref.bi_q(qi, :)) * double(ref.bi_doc)';
    [refSorted, theirs] = sort(refS, 'descend');
    orderExact = orderExact + nnz(mine ~= theirs);
    % Near-duplicate documents tie to within float32 noise and can swap; rank
    % agreement is judged on the REFERENCE's own scores (as for ColBERT below).
    orderGap = max(orderGap, max(abs(refS(mine) - refSorted)));
end
add("bi-encoder rankings: max reference-score gap between my order and theirs", orderGap, 1e-5);
fprintf('bi-encoder: %d of %d rank positions differ from the reference order (near-ties within float32 noise)\n', orderExact, nq * nd);
lat.bi = timeQueries(bi, queries);   idxBytes.bi = numel(bi.CorpusMatrix) * 8;
metrics.bi = evalRanker(bi, queries, relCells, metricObjs);

% ------------------------------------------------------------------ %
% (c) cross-encoder: TinyBERT-L-2-v2, raw logits on BM25 top-20 pairs
% ------------------------------------------------------------------ %
ce = OceanNLP.Embedding.CrossEncoder('Weights', load_("cross"), 'VocabFile', vocab_("cross"), ...
    'ModelName', "ms-marco-TinyBERT-L-2-v2", 'MaxLength', 512);
topTexts = string(ref.top_doc_texts);
[nqTop, kTop] = size(topTexts);
ceMine = zeros(nqTop, kTop);
t0 = tic;
for qi = 1:nqTop
    ceMine(qi, :) = ce.scoreDocs(queries(qi), topTexts(qi, :));
end
fprintf('cross-encoder: scored %d pairs in %.1fs\n', nqTop * kTop, toc(t0));
add("cross-encoder logits vs reference (max abs)", max(abs(ceMine - double(ref.ce_scores)), [], 'all'), 1e-3);
ceOrder = 0;
for qi = 1:nqTop
    [~, a] = sort(ceMine(qi, :), 'descend');
    [~, b] = sort(double(ref.ce_scores(qi, :)), 'descend');
    ceOrder = ceOrder + nnz(a ~= b);
end
add("cross-encoder rerank order: positions differing from reference", ceOrder, 0);

% ------------------------------------------------------------------ %
% (d) ColBERT: answerai-colbert-small-v1, per-token embeddings then MaxSim
% ------------------------------------------------------------------ %
cb = OceanNLP.Embedding.ColBERTEncoder('Weights', load_("colbert"), 'VocabFile', vocab_("colbert"), ...
    'ModelName', "answerai-colbert-small-v1");
qTokRef = double(ref.cb_q_tok);                       % (nq x 32 x 96)
qDiff = 0;
for qi = 1:nq
    Q = cb.encodeQuery(queries(qi));
    if ~isequal(size(Q), [size(qTokRef, 2), size(qTokRef, 3)])
        error('ColBERT query shape %s differs from reference %s', mat2str(size(Q)), mat2str(size(qTokRef)));
    end
    qDiff = max(qDiff, max(abs(Q - squeeze(qTokRef(qi, :, :))), [], 'all'));
end
add("ColBERT query token embeddings vs reference (max abs)", qDiff, 1e-4);

cbRanker = OceanNLP.Retrieval.ColBERTRanker('Encoder', cb);
t0 = tic; cbRanker.fit(subTexts); tFit = toc(t0);
fprintf('ColBERT: encoded %d docs in %.1fs (%.2fs/doc)\n', nd, tFit, tFit / nd);
lens = double(ref.cb_d_lens(:))';  offs = [0 cumsum(lens)];
dRef = double(ref.cb_d_tok_concat);
dDiff = 0; shapeBad = 0;
for j = 1:nd
    Dm = double(cbRanker.DocMatrices{j});
    if size(Dm, 1) ~= lens(j), shapeBad = shapeBad + 1; continue; end
    dDiff = max(dDiff, max(abs(Dm - dRef(offs(j)+1:offs(j+1), :)), [], 'all'));
end
add("ColBERT doc matrices with a different token count than reference", shapeBad, 0);
add("ColBERT doc token embeddings vs reference (max abs, single precision)", dDiff, 1e-4);
msDiff = 0; msOrderGap = 0; msExactDiff = 0;
for qi = 1:nq
    Q = cb.encodeQuery(queries(qi));
    mine = cbRanker.Interaction.score(Q, cbRanker.DocMatrices);
    refS = double(ref.cb_maxsim(qi, :));
    msDiff = max(msDiff, max(abs(mine - refS)));
    [~, ia] = sort(mine, 'descend'); [refSorted, ib] = sort(refS, 'descend');
    msExactDiff = msExactDiff + nnz(ia ~= ib);
    % Scores are single precision (~1e-6 noise), so two docs closer than that
    % can swap places. Rank agreement is therefore judged on the REFERENCE's
    % own scores: my ordering, read off the reference scores, must be sorted
    % to within tolerance.
    msOrderGap = max(msOrderGap, max(abs(refS(ia) - refSorted)));
end
add("ColBERT MaxSim scores vs reference (max abs)", msDiff, 1e-3);
add("ColBERT rankings: max reference-score gap between my order and theirs", msOrderGap, 1e-4);
fprintf('ColBERT: %d of %d rank positions differ from the reference order (near-ties within float32 noise)\n', msExactDiff, nq * nd);
lat.cb = timeQueries(cbRanker, queries);
idxBytes.cb = sum(cellfun(@(m) numel(m) * 4, cbRanker.DocMatrices));
metrics.cb = evalRanker(cbRanker, queries, relCells, metricObjs);
% Query-preprocessing sensitivity (checkpoint files disagree, see ColBERTEncoder header).
% Documents do not depend on these options, so the stored doc matrices are reused.
variants = {'ColBERT no expansion', struct('QueryExpansion', false); ...
            'ColBERT attend [MASK]', struct('AttendToMaskTokens', true)};
for v = 1:size(variants, 1)
    opts = namedargs2cell(variants{v, 2});
    encV = OceanNLP.Embedding.ColBERTEncoder('Weights', load_("colbert"), 'VocabFile', vocab_("colbert"), opts{:});
    key = sprintf('cbv%d', v);
    metrics.(key) = zeros(1, numel(metricObjs));
    t0 = tic;
    for qi = 1:nq
        ranked = cbRanker.Interaction.score(encV.encodeQuery(queries(qi)), cbRanker.DocMatrices);
        [~, ranked] = sort(ranked, 'descend');
        for mi = 1:numel(metricObjs)
            metrics.(key)(mi) = metrics.(key)(mi) + metricObjs{mi}.forward(ranked, relCells{qi}) / nq;
        end
    end
    lat.(key) = toc(t0) / nq;  idxBytes.(key) = idxBytes.cb;
end

% ------------------------------------------------------------------ %
% (e) benchmark on the same corpus
% ------------------------------------------------------------------ %
bm = OceanNLP.Retrieval.BM25Ranker();  bm.fit(subTexts);
lat.bm25 = timeQueries(bm, queries);
metrics.bm25 = evalRanker(bm, queries, relCells, metricObjs);
idxBytes.bm25 = NaN;
rr = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', OceanNLP.Retrieval.BM25Ranker(), 'CrossEncoder', ce, 'K', 20);
rr.fit(subTexts);
lat.rerank = timeQueries(rr, queries);
metrics.rerank = evalRanker(rr, queries, relCells, metricObjs);
idxBytes.rerank = NaN;

% ------------------------------------------------------------------ %
fprintf('\n=== Agreement with Python reference ===\n');
allPass = true;
for k = 1:numel(results)
    r = results(k);
    if r.pass, s = "PASS"; else, s = "FAIL"; allPass = false; end
    fprintf('  [%s]  %-72s  %.3e (tol %.0e)\n', s, r.name, r.value, r.tol);
end

fprintf('\n=== Benchmark: Cranfield, %d docs, %d queries ===\n', nd, nq);
fprintf('%-24s %7s %7s %9s %14s %14s\n', 'system', 'P@10', 'MAP', 'NDCG@10', 'latency s/qry', 'index bytes');
rows = {'BM25', 'bm25'; 'bi-encoder MiniLM', 'bi'; 'BM25->cross-enc@20', 'rerank'; 'ColBERT small', 'cb'; ...
        'ColBERT no expansion', 'cbv1'; 'ColBERT attend [MASK]', 'cbv2'};
for k = 1:size(rows, 1)
    m = metrics.(rows{k, 2});
    fprintf('%-24s %7.4f %7.4f %9.4f %14.3f %14s\n', rows{k, 1}, m(1), m(2), m(3), lat.(rows{k, 2}), fmtBytes(idxBytes.(rows{k, 2})));
end
if allPass
    fprintf('\nOverall: ALL AGREEMENT CHECKS PASSED\n');
else
    fprintf('\nOverall: SOME AGREEMENT CHECKS FAILED\n');
end
end

function t = timeQueries(ranker, queries)
t0 = tic;
for qi = 1:numel(queries), ranker.rank(queries(qi)); end
t = toc(t0) / numel(queries);
end

function m = evalRanker(ranker, queries, relCells, metricObjs)
ev = OceanNLP.Retrieval.RetrievalEvaluator(ranker, metricObjs);
rep = ev.evaluate(queries, relCells);
m = table2array(rep(end, :));        % Mean row
end

function s = fmtBytes(b)
if isnan(b), s = 'n/a'; else, s = sprintf('%.2f MB', b / 1e6); end
end
