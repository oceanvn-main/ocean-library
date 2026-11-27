function benchmark_text_in_matlab()
% BENCHMARK_TEXT_IN_MATLAB  What does the analyzer / subword upgrade do to RETRIEVAL quality?
% Real Cranfield (all 1400 docs, the 25 real queries + real qrels used by the other
% validations). BM25 (inverted index) and TF-IDF cosine, each with different text analysis:
%
%   basic            RegexTokenizer only            (today's default)
%   +stop            + Lucene English stop words
%   +stop+porter     + Porter stemming              (Analyzer preset "english")
%   BPE-4000         byte-level BPE trained on the corpus itself (transductive; no reference
%                    system to compare against, reported for completeness)
%
% STATED EXPECTATION (before running): stemming helps a little on Cranfield (published results
% show small gains for BM25); stop words are roughly neutral; BPE-as-index-terms is worse than
% words for lexical matching. A null or negative result is reported as such. 25 queries -> small
% sample, so every difference vs 'basic' is shown with a paired bootstrap 95% CI of the mean AP
% difference (2000 resamples over queries, seeded) and an exact two-sided sign-test p-value.
%
% Run:  OceanTest.OceanNLP.Text.benchmark_text_in_matlab()

here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
R = fullfile(here, '..', '+Retrieval');
data = load(fullfile(R, 'cranfield_validation_data.mat'));
texts = jsondecode(fileread(fullfile(R, 'cranfield_texts.json')));
docs = string(texts.docs(:))';
queries = string(cellfun(@(x) x, data.query_texts(:), 'UniformOutput', false))';
rel = double(data.relevance);
relCells = arrayfun(@(i) rel(i, :), 1:numel(queries), 'UniformOutput', false);
fprintf('=== Cranfield: %d docs, %d queries ===\n', numel(docs), numel(queries));

metrics = {OceanStats.Metrics.Retrieval.PrecisionAtK('K', 10), ...
           OceanStats.Metrics.Retrieval.AveragePrecision(), ...
           OceanStats.Metrics.Retrieval.NDCGAtK('K', 10)};

configs = {};
configs(end+1, :) = {'basic',        OceanNLP.Text.Analyzer.preset("basic")};
configs(end+1, :) = {'+stop',        OceanNLP.Text.Analyzer.fromSpec(struct('Tokenizer', "regex", 'TokenFilters', ["lowercase" "stopword"]))};
configs(end+1, :) = {'+stop+porter', OceanNLP.Text.Analyzer.preset("english")};
t0 = tic;
bpe = OceanNLP.Subword.SubwordTokenizer();
bpe.fit(docs, 'VocabSize', 4000);
fprintf('trained BPE-4000 in %.1fs\n', toc(t0));
configs(end+1, :) = {'BPE-4000', bpe};

fprintf('\n%-14s %-6s %6s %6s %8s %7s  %s\n', 'analysis', 'rank', 'P@10', 'MAP', 'NDCG@10', 'fit(s)', 'dMAP vs basic [95% CI]   better/worse   sign p');
rng(1);
baseAP = struct();
for c = 1:size(configs, 1)
    for rk = ["bm25" "tfidf"]
        tok = configs{c, 2};
        t0 = tic;
        if rk == "bm25"
            ranker = OceanNLP.Retrieval.BM25Ranker('Tokenizer', tok);
        else
            ranker = OceanNLP.Retrieval.SemanticRanker('Tokenizer', tok);
        end
        ranker.fit(docs);
        tFit = toc(t0);
        ev = OceanNLP.Retrieval.RetrievalEvaluator(ranker, metrics);
        rep = ev.evaluate(queries, relCells);
        M = table2array(rep);
        meanRow = M(end, :);
        ap = M(1:end-1, 2);
        if c == 1, baseAP.(rk) = ap; note = ''; else
            d = ap - baseAP.(rk);
            nb = nnz(d > 1e-12);  nw = nnz(d < -1e-12);
            idx = randi(numel(d), numel(d), 2000);
            bm = mean(d(idx), 1);
            ci = prctile_(bm, [2.5 97.5]);
            note = sprintf('%+.3f [%+.3f, %+.3f]   %2d / %2d   p=%.2f', mean(d), ci(1), ci(2), nb, nw, signTestP_(nb, nw));
        end
        fprintf('%-14s %-6s %6.3f %6.3f %8.3f %7.1f  %s\n', configs{c, 1}, rk, meanRow(1), meanRow(2), meanRow(3), tFit, note);
    end
end
end

function q = prctile_(x, p)
% linear-interpolated percentiles without the Statistics Toolbox
x = sort(x(:));  n = numel(x);
pos = 1 + (n - 1) * p / 100;
lo = floor(pos);  hi = ceil(pos);
q = x(lo)' + (pos - lo) .* (x(hi) - x(lo))';
end

function p = signTestP_(nb, nw)
% exact two-sided sign test on the non-tied queries (Binomial(n, 0.5))
n = nb + nw;
if n == 0, p = 1; return; end
k = min(nb, nw);
lg = @(m) gammaln(m + 1);
pk = exp(arrayfun(@(j) lg(n) - lg(j) - lg(n - j), 0:k) - n * log(2));
p = min(1, 2 * sum(pk));
end
