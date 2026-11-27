classdef AttentionBeamDecoder < handle
    % ATTENTIONBEAMDECODER  Greedy / beam search for autoregressive (attention
    %   encoder-decoder) recognisers, driven by a model-agnostic step function.
    %
    %   d   = OceanNLP.Speech.Decoding.AttentionBeamDecoder(vocab, 'Method', "beam", 'BeamWidth', 8, ...
    %           'NBest', [], 'LengthPenalty', 1, 'MaxLength', 200, 'MaxLengthRatio', Inf, 'ModelId', "")
    %   res = d.search(stepFn, srcLengths)     1 x N struct: Labels, LogProb, Score, Length,
    %                                          HitMaxLength, Candidates (N-best, same fields)
    %   tr  = d.decode(stepFn, srcLengths)     1 x N Core.Transcript
    %
    %   STEP FUNCTION  logp = stepFn(prefixes, src)
    %     prefixes  B x u token ids, column 1 = Bos; every row is a live hypothesis
    %     src       B x 1 utterance index (1..N) each row belongs to
    %     logp      K x B next-token scores; log-softmax is (re)applied here, which is
    %               idempotent on log-probabilities, so raw logits are accepted too.
    %   The decoder knows nothing about the model: anything that maps (prefix, utterance)
    %   to next-token log-probabilities can be decoded (tested with a closed-form toy model).
    %
    %   INCREMENTAL (STATEFUL) STEP FUNCTION -- e.g. a Transformer decoder with a key/value
    %   cache:  res = d.search(stepFn, srcLengths, 'Incremental', true, 'State', s0)
    %     [logp, s] = stepFn(prefixes, src, s, parent)
    %     parent    B x 1 row of the PREVIOUS call that each current row extends (first call:
    %               row n of s0 is utterance n, parent = (1:N)'). The decoder threads the
    %               opaque state s and never inspects it; the step function must reorder /
    %               prune / duplicate its per-row state as s_row(parent) before appending the
    %               new token prefixes(:, end). Rows dropped by greedy completion, beam
    %               pruning, Eos or the early stop simply do not appear in parent.
    %   The search itself is identical in both modes (same expansions, ties, scores).
    %
    %   SCORES. A hypothesis y = (y_1..y_L, Eos) has
    %     logP(y) = sum_{i=1}^{L+1} log p(y_i | y_<i, x)      (Eos included)
    %     Score(y) = logP(y) / (L + 1)^alpha,  alpha = LengthPenalty  (0 = raw log-prob,
    %     1 = per-token average). Normalisation counteracts the bias of log-prob ranking
    %     towards short outputs (early Eos); it is applied to FINISHED hypotheses.
    %   Tokens Blank, Pad and Bos are never emitted (score -Inf).
    %   MAX-LENGTH GUARD: utterance n may emit at most Lmax(n) tokens INCLUDING Eos,
    %     Lmax(n) = max(1, min(MaxLength, ceil(MaxLengthRatio * srcLengths(n))));
    %   at the last allowed step only Eos may follow (forced end; HitMaxLength = true and
    %   its log-prob is still counted, so looping hypotheses pay for the forced Eos).
    %
    %   GREEDY: y_i = argmax_k logp_k at each step (ties -> lowest id) until Eos.
    %   BEAM (synchronous, batched over utterances): every live hypothesis of length u is
    %   expanded by every token; per utterance the BeamWidth best expansions by raw logP
    %   are kept (ties -> lower beam row, then lower token id); expansions ending in Eos
    %   move to the finished list. An utterance stops when no live hypothesis is left, or
    %   when at least NBest are finished and no live one can still beat the NBest-th
    %   finished score: since logP only decreases and lengths are <= Lmax, a live
    %   hypothesis with logP = L <= 0 can reach at most L / Lmax^alpha (exact bound, so
    %   the early stop never changes the result). Final ranking by Score.
    %   With BeamWidth = 1 the beam equals greedy; with BeamWidth >= the number of
    %   hypotheses it is exhaustive (both tested against brute-force enumeration).
    %
    %   Transcript: Decoder "attention-greedy" | "attention-beam", DecoderOptions (all
    %   options above), Score (normalised), Candidates; no Segments (no timing claimed).

    properties (SetAccess = immutable)
        Vocabulary
        Method (1,1) string
        BeamWidth (1,1) double
        NBest (1,1) double
        LengthPenalty (1,1) double
        MaxLength (1,1) double
        MaxLengthRatio (1,1) double
        ModelId (1,1) string
    end

    methods
        function obj = AttentionBeamDecoder(vocabulary, opts)
            arguments
                vocabulary (1,1) OceanNLP.Speech.Data.SpeechVocabulary
                opts.Method (1,1) string {mustBeMember(opts.Method, ["greedy", "beam"])} = "beam"
                opts.BeamWidth (1,1) double {mustBePositive, mustBeInteger} = 8
                opts.NBest double {mustBeScalarOrEmpty} = []
                opts.LengthPenalty (1,1) double {mustBeNonnegative, mustBeFinite} = 1
                opts.MaxLength (1,1) double {mustBePositive, mustBeInteger} = 200
                opts.MaxLengthRatio (1,1) double {mustBePositive} = Inf
                opts.ModelId (1,1) string = ""
            end
            nb = opts.NBest;
            if isempty(nb), nb = opts.BeamWidth; end
            if opts.Method == "greedy"
                opts.BeamWidth = 1; nb = 1;
            end
            obj.Vocabulary = vocabulary;
            obj.Method = opts.Method;
            obj.BeamWidth = opts.BeamWidth;
            obj.NBest = max(1, min(nb, opts.BeamWidth));
            obj.LengthPenalty = opts.LengthPenalty;
            obj.MaxLength = opts.MaxLength;
            obj.MaxLengthRatio = opts.MaxLengthRatio;
            obj.ModelId = opts.ModelId;
        end

        function L = maxLengths(obj, srcLengths)
            % MAXLENGTHS  Per-utterance cap on emitted tokens (Eos included).
            L = min(obj.MaxLength, ceil(obj.MaxLengthRatio * double(srcLengths(:).')));
            L = max(1, L);
        end

        function res = search(obj, stepFn, srcLengths, opts)
            arguments
                obj
                stepFn (1,1) function_handle
                srcLengths (1,:) double {mustBeNonnegative}
                opts.Incremental (1,1) logical = false
                opts.State = []
            end
            st = struct('Fn', stepFn, 'Incremental', opts.Incremental, 'State', {opts.State});
            if obj.Method == "greedy"
                res = obj.greedy(st, srcLengths);
            else
                res = obj.beam(st, srcLengths);
            end
        end

        function tr = decode(obj, stepFn, srcLengths, opts)
            arguments
                obj
                stepFn (1,1) function_handle
                srcLengths (1,:) double {mustBeNonnegative}
                opts.Incremental (1,1) logical = false
                opts.State = []
            end
            res = obj.search(stepFn, srcLengths, 'Incremental', opts.Incremental, 'State', opts.State);
            dopts = obj.options();
            tr = OceanNLP.Speech.Core.Transcript.empty(1, 0);
            for n = 1:numel(res)
                c = res(n).Candidates;
                for h = 1:numel(c)
                    c(h).Text = obj.Vocabulary.decode(c(h).TokenIds);
                end
                c = orderfields(c, {'Text', 'TokenIds', 'Score', 'LogProb', 'Length', 'HitMaxLength'});
                tr(n) = OceanNLP.Speech.Core.Transcript(c(1).Text, 'TokenIds', c(1).TokenIds, ...
                    'Decoder', "attention-" + obj.Method, 'DecoderOptions', dopts, 'ModelId', obj.ModelId, ...
                    'Score', c(1).Score, 'Candidates', c);
            end
        end

        function o = options(obj)
            o = struct('Method', obj.Method, 'BeamWidth', obj.BeamWidth, 'NBest', obj.NBest, ...
                'LengthPenalty', obj.LengthPenalty, 'MaxLength', obj.MaxLength, ...
                'MaxLengthRatio', obj.MaxLengthRatio, 'Bos', obj.Vocabulary.Bos, 'Eos', obj.Vocabulary.Eos, ...
                'Disallowed', [obj.Vocabulary.Blank, obj.Vocabulary.Pad, obj.Vocabulary.Bos]);
        end
    end

    methods (Access = private)
        function [lp, st] = callStep(obj, st, prefixes, src, parent)
            if st.Incremental
                [Z, st.State] = st.Fn(prefixes, src, st.State, parent);
            else
                Z = st.Fn(prefixes, src);
            end
            Z = double(gather(OceanDL.Autograd.Tensor.raw(Z)));
            K = obj.Vocabulary.numClasses();
            if size(Z, 1) ~= K || size(Z, 2) ~= size(prefixes, 1)
                error('OceanNLP:Speech:Decoding:AttentionBeamDecoder:StepSize', ...
                    'stepFn must return K x B = %d x %d scores, got %s.', K, size(prefixes, 1), mat2str(size(Z)));
            end
            m = max(Z, [], 1);
            lp = Z - (m + log(sum(exp(Z - m), 1)));
            v = obj.Vocabulary;
            lp([v.Blank, v.Pad, v.Bos], :) = -Inf;
        end

        function s = normScore(obj, logP, len)
            s = logP ./ (len .^ obj.LengthPenalty);
        end

        function res = greedy(obj, st, srcLengths)
            N = numel(srcLengths);
            Lmax = obj.maxLengths(srcLengths);
            eos = obj.Vocabulary.Eos;
            seqs = repmat(obj.Vocabulary.Bos, N, 1);
            logP = zeros(N, 1);
            alive = true(N, 1);
            hitMax = false(N, 1);
            len = zeros(N, 1);
            u = 0;
            prevIdx = (1:N)';
            while any(alive)
                u = u + 1;                                    % emitting token u
                idx = find(alive);
                [~, parent] = ismember(idx, prevIdx);         % rows only drop out: order kept
                [lp, st] = obj.callStep(st, seqs(idx, :), idx, parent);
                prevIdx = idx;
                tok = zeros(numel(idx), 1);
                for j = 1:numel(idx)
                    n = idx(j);
                    if u >= Lmax(n)
                        tok(j) = eos; hitMax(n) = true;
                    else
                        [~, tok(j)] = max(lp(:, j));
                    end
                    logP(n) = logP(n) + lp(tok(j), j);
                end
                seqs(:, end + 1) = obj.Vocabulary.Pad; %#ok<AGROW>
                seqs(idx, end) = tok;
                done = idx(tok == eos);
                len(done) = u;
                alive(done) = false;
            end
            res = obj.emptyResult(N);
            for n = 1:N
                c = obj.candidate(seqs(n, 2:len(n)), logP(n), len(n), hitMax(n));
                res(n) = obj.result(c);
            end
        end

        function res = beam(obj, st, srcLengths)
            N = numel(srcLengths);
            Lmax = obj.maxLengths(srcLengths);
            v = obj.Vocabulary;
            eos = v.Eos;
            W = obj.BeamWidth;
            % live hypotheses of all utterances: rows of one matrix (same length u+1)
            live = struct('Seq', repmat(v.Bos, N, 1), 'LogP', zeros(N, 1), 'Src', (1:N)', 'Parent', (1:N)');
            finished = cell(1, N);
            u = 0;
            while ~isempty(live.Src)
                u = u + 1;
                [lp, st] = obj.callStep(st, live.Seq, live.Src, live.Parent);  % K x B
                K = size(lp, 1);
                newSeq = zeros(0, u + 1); newLogP = zeros(0, 1); newSrc = zeros(0, 1); newParent = zeros(0, 1);
                for n = unique(live.Src).'
                    rows = find(live.Src == n);
                    cand = live.LogP(rows).' + lp(:, rows);          % K x b
                    if u >= Lmax(n)
                        mask = -Inf(K, 1); mask(eos) = 0;
                        cand = cand + mask;
                    end
                    [tk, bk] = ndgrid(1:K, 1:numel(rows));
                    ok = isfinite(cand(:));
                    tbl = sortrows([-cand(ok), bk(ok), tk(ok)]);     % best first; ties: lower row, lower token
                    tbl = tbl(1:min(W, size(tbl, 1)), :);
                    for r = 1:size(tbl, 1)
                        row = rows(tbl(r, 2)); tok = tbl(r, 3); lpv = -tbl(r, 1);
                        if tok == eos
                            finished{n}(end + 1) = obj.candidate(live.Seq(row, 2:end), lpv, u, u >= Lmax(n));
                        else
                            newSeq(end + 1, :) = [live.Seq(row, :), tok]; %#ok<AGROW>
                            newLogP(end + 1, 1) = lpv; %#ok<AGROW>
                            newSrc(end + 1, 1) = n; %#ok<AGROW>
                            newParent(end + 1, 1) = row; %#ok<AGROW>
                        end
                    end
                    % exact early stop for utterance n
                    f = finished{n};
                    keep = newSrc == n;
                    if numel(f) >= obj.NBest && any(keep)
                        sc = sort([f.Score], 'descend');
                        bound = max(newLogP(keep)) / Lmax(n) ^ obj.LengthPenalty;
                        if bound < sc(obj.NBest)
                            newSeq = newSeq(~keep, :); newLogP = newLogP(~keep); newSrc = newSrc(~keep);
                            newParent = newParent(~keep);
                        end
                    end
                end
                live = struct('Seq', newSeq, 'LogP', newLogP, 'Src', newSrc, 'Parent', newParent);
            end
            res = obj.emptyResult(N);
            for n = 1:N
                f = finished{n};
                [~, order] = sort([f.Score], 'descend');        % stable: earlier-finished first on ties
                f = f(order(1:min(obj.NBest, numel(order))));
                res(n) = obj.result(f);
            end
        end

        function c = candidate(obj, ids, logP, len, hitMax)
            c = struct('Text', "", 'TokenIds', double(ids), 'Score', obj.normScore(logP, len), ...
                'LogProb', logP, 'Length', len, 'HitMaxLength', logical(hitMax));
        end

        function r = result(~, cands)
            r = struct('Labels', cands(1).TokenIds, 'LogProb', cands(1).LogProb, 'Score', cands(1).Score, ...
                'Length', cands(1).Length, 'HitMaxLength', cands(1).HitMaxLength, 'Candidates', cands);
        end

        function r = emptyResult(~, N)
            r = repmat(struct('Labels', [], 'LogProb', 0, 'Score', 0, 'Length', 0, 'HitMaxLength', false, ...
                'Candidates', []), 1, N);
        end
    end
end
