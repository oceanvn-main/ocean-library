classdef (Abstract) IRanker < handle
    % IRANKER  Abstract interface: fit a retrieval index once, rank(query) many times.
    %
    %   PyTorch-style contract, same shape as IVectorizer/IWeighting: fit()
    %   mutates buffers (a fitted vectorizer/weighting, the corpus size) on
    %   self; rank() reads them back — no separate returned state struct,
    %   and no re-indexing of the corpus on every query.
    %
    %   DESIGN NOTE (deviation from the master plan's roster shorthand):
    %   NLP_Library_Master_Plan.md §5 describes this API loosely as
    %   `rank(query, corpus)`. Every other Tier-0/Tier-1 class in this
    %   project (IWeighting, IVectorizer) uses a fit-once-buffers-on-self,
    %   read-many pattern instead: fit() is the expensive step (build a
    %   vocabulary, a count/TF-IDF matrix, a BM25 IDF table), and the
    %   per-query step should be cheap and repeatable without redoing that
    %   work. `rank(query, corpus)` would force every implementation to
    %   silently re-fit (or worse, cache-and-hope) on every single call,
    %   which both violates that convention and wastes work in the one
    %   place retrieval systems are usually latency-sensitive (serving
    %   queries). This interface therefore uses fit(corpus) + rank(query):
    %   index once, serve many queries — the standard shape of a real
    %   retrieval system (build an inverted index / embedding matrix once,
    %   then answer queries against it).
    %
    %   CONVENTION:
    %     fit(corpus)   — corpus: (1 x N) string/cellstr array of raw
    %                     documents, one per corpus position. Builds and
    %                     stores whatever index the concrete ranker needs
    %                     (vocabulary, weighted matrix, TF-IDF matrix, ...)
    %                     on self. Calling fit() again rebuilds the index
    %                     from scratch and clobbers any previous fit,
    %                     matching CountVectorizer/TfidfVectorizer/
    %                     IWeighting fit() semantics elsewhere in OceanNLP.
    %
    %     [rankedIdx, scores] = rank(query)
    %                   — query: char or string scalar, ONE raw query
    %                     document (tokenized/encoded internally against
    %                     the SAME fitted vocabulary used by fit(); tokens
    %                     never seen at fit() time are out-of-vocabulary
    %                     and handled OOV-safely, never an error).
    %                     rankedIdx — (1 x N) double row vector: a
    %                       permutation of 1:N, the ORIGINAL 1-based corpus
    %                       positions (as passed to fit()), sorted by
    %                       DESCENDING relevance to query. rankedIdx(1) is
    %                       the single most relevant document's position in
    %                       the corpus fit() was given.
    %                     scores — (1 x N) double row vector, ALIGNED with
    %                       rankedIdx (not with corpus order): scores(k) is
    %                       the relevance score of document rankedIdx(k).
    %                       Higher is always more relevant, for every
    %                       concrete ranker (BM25 additive weight, cosine
    %                       similarity) — callers may compare/rank across
    %                       IRanker implementations without knowing which
    %                       concrete scoring formula produced the number.
    %                     Ties (equal scores) keep their original corpus
    %                     order (MATLAB's sort() is stable), so results are
    %                     deterministic given the same fit() and query.
    %                     This exact [rankedIdx, scores] shape is the
    %                     interop contract between BM25Ranker and
    %                     SemanticRanker: either may be substituted for the
    %                     other behind this interface.
    %
    %     tf = isFitted(obj) — true once fit() has been called.
    %
    %   See also: OceanNLP.Retrieval.BM25Ranker, OceanNLP.Retrieval.SemanticRanker
    methods (Abstract)
        fit(obj, corpus)
        [rankedIdx, scores] = rank(obj, query)
        tf = isFitted(obj)
        s  = name(obj)
    end
end
