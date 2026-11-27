classdef CTCBeamSearch
    % CTCBEAMSEARCH  CTC prefix beam search (Graves 2012, "Supervised Sequence
    %   Labelling", ch. 7.5; Hannun, Maas, Jurafsky & Ng 2014, arXiv:1408.2873),
    %   with optional shallow fusion of a prefix scorer (e.g. a language model).
    %
    %   WHY NOT GREEDY: CTC defines  p(l | x) = sum_{pi : B(pi) = l} prod_t y_{pi_t}(t),
    %   B = "merge repeats, drop blanks". Greedy decoding returns B(best path), but
    %   the most probable LABELLING may own many mediocre paths instead of one good
    %   one. Prefix beam search scores LABELLINGS: all paths that collapse to the
    %   same prefix are summed.
    %
    %   STATE per prefix l (log space):
    %     p_b(l)   mass of all path prefixes that collapse to l and END IN BLANK
    %     p_nb(l)  mass of those that END IN l's last label
    %   (the split is needed because the next symbol c = last(l) either repeats
    %   the label -- merged -- or, after a blank, starts a new copy of it).
    %   STEP t, for every prefix l in the beam, with y = y(t):
    %     blank:                  p_b'(l)      += (p_b + p_nb)(l) * y_blank
    %     c = last(l):            p_nb'(l)     += p_nb(l) * y_c          (repeat merges)
    %                             p_nb'(l + c) += p_b(l)  * y_c          (needs the blank)
    %     c ~= last(l), c ~= b:   p_nb'(l + c) += (p_b + p_nb)(l) * y_c
    %   then keep the BeamWidth prefixes with the highest Score. At the end
    %     LogProb(l) = log(p_b(l) + p_nb(l))
    %   which is exactly log p(l | x) whenever no path mass of l was pruned
    %   (BeamWidth >= number of live prefixes, Prune = -inf); otherwise it is a
    %   lower bound of it (paths through pruned prefixes are lost).
    %
    %   SHALLOW FUSION (off by default): each label appended to a prefix adds
    %     Weight * PrefixScorer(l, c) + Bonus
    %   to a per-prefix fusion score F(l), so the beam is ranked by
    %     Score(l) = LogProb(l) + F(l) = log p_ctc(l|x) + Weight * log p_lm(l) + Bonus * |l|
    %   (Hannun et al. eq. 2 with alpha = Weight, beta = Bonus). Because F depends
    %   only on the prefix, it is kept separate from the CTC masses: LogProb stays
    %   a pure CTC probability. The scorer is called only when Weight ~= 0, once
    %   per new prefix: logScore = PrefixScorer(prevPrefix (1 x L double), nextLabel).
    %
    %   PRUNE: at each step only labels with logProbs(c, t) >= Prune are tried as
    %   extensions (blank is always propagated, so the beam never empties).
    %
    %   OUTPUT: struct array (best first, sorted by Score, at most BeamWidth) with
    %     Labels (1 x L double), LogProb, Score.
    %   Ties are broken deterministically by candidate order (stable sort):
    %   surviving prefixes first, then extensions in (prefix, label) order.
    %   COST per step: O(B C) vectorised (B = BeamWidth, C = labels tried) plus
    %   B scorer calls per label when fusion is on.
    %
    % EXAMPLE:
    %   lp = logits - log(sum(exp(logits), 1));          % K x T log-softmax
    %   hyps = OceanDL.Recurrent.CTCBeamSearch.decode(lp, 'BeamWidth', 16);
    %   hyps(1).Labels
    %
    % See also: OceanDL.Recurrent.CTCDecoder (greedy), OceanDL.Autograd.Functions.CTCLoss

    methods (Static)
        function hyps = decode(logProbs, opts)
            % DECODE  logProbs (K x T log-probabilities of one sequence) -> N-best struct array.
            arguments
                logProbs
                opts.Blank (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.BeamWidth (1,1) double {mustBePositive, mustBeInteger} = 8
                opts.Prune (1,1) double = -inf
                opts.PrefixScorer = []
                opts.Weight (1,1) double {mustBeFinite} = 0
                opts.Bonus (1,1) double {mustBeFinite} = 0
            end
            import OceanDL.Recurrent.CTCBeamSearch
            LP = double(OceanDL.Autograd.Tensor.raw(logProbs));
            CTCBeamSearch.validate(LP, opts);
            [~, T] = size(LP);
            blank = opts.Blank;
            useScorer = ~isempty(opts.PrefixScorer) && opts.Weight ~= 0;

            K = size(LP, 1);
            % Beam = prefixes P (1 x B cell), column vectors Pb, Pnb (log masses), F (fusion).
            P = {zeros(1, 0)};   Pb = 0;   Pnb = -inf;   F = 0;          % the empty prefix, p = 1
            for t = 1:T
                lp = LP(:, t);
                cand = find(lp >= opts.Prune & (1:K).' ~= blank).';      % 1 x C labels tried
                C = numel(cand);
                B = numel(P);
                last = zeros(B, 1);                                       % last label (0 = empty prefix)
                for i = 1:B
                    if ~isempty(P{i}), last(i) = P{i}(end); end
                end
                tot = CTCBeamSearch.lse(Pb, Pnb);
                % (1) blank keeps l and ends in blank; (2) repeating last(l) merges into l
                sPb = tot + lp(blank);
                sPnb = -inf(B, 1);
                ne = last > 0;
                sPnb(ne) = Pnb(ne) + lp(last(ne));
                % extensions l + c (B x C): (3) c ~= last(l) from all mass of l,
                % (2') c == last(l) only from the paths of l that end in blank
                E = tot + lp(cand).';
                rep = last == cand;
                fromBlank = Pb + lp(cand).';
                E(rep) = fromBlank(rep);
                % an extension may BE another prefix already in the beam (l + c == l2):
                % its mass then belongs to l2's non-blank-ending mass (same labelling)
                keys = strings(1, B);
                for i = 1:B, keys(i) = sprintf('%d,', P{i}); end
                for j = find(ne).'
                    [isIn, i] = ismember(string(sprintf('%d,', P{j}(1:end-1))), keys);
                    c = find(cand == last(j), 1);
                    if isIn && ~isempty(c)
                        sPnb(j) = CTCBeamSearch.lse(sPnb(j), E(i, c));
                        E(i, c) = -inf;                                   % merged, not a new prefix
                    end
                end
                FE = repmat(F + opts.Bonus, 1, C);
                if useScorer
                    for i = 1:B
                        for c = find(isfinite(E(i, :)))
                            FE(i, c) = FE(i, c) + opts.Weight * double(opts.PrefixScorer(P{i}, cand(c)));
                        end
                    end
                end
                % candidates in a fixed order: current prefixes, then extensions
                % prefix-major (stable sort => deterministic ties); keep the
                % BeamWidth best by fused score, dropping zero-mass ones
                ET = E.'; FT = FE.';
                score = [CTCBeamSearch.lse(sPb, sPnb) + F; ET(:) + FT(:)];
                live = find(isfinite(score));
                [~, order] = sort(score(live), 'descend');
                keep = live(order(1:min(opts.BeamWidth, numel(order))));
                nB = numel(keep);
                nP = cell(1, nB); nPb = zeros(nB, 1); nPnb = zeros(nB, 1); nF = zeros(nB, 1);
                for q = 1:nB
                    k = keep(q);
                    if k <= B
                        nP{q} = P{k}; nPb(q) = sPb(k); nPnb(q) = sPnb(k); nF(q) = F(k);
                    else
                        [c, i] = ind2sub([C B], k - B);
                        nP{q} = [P{i}, cand(c)]; nPb(q) = -inf; nPnb(q) = E(i, c); nF(q) = FE(i, c);
                    end
                end
                P = nP; Pb = nPb; Pnb = nPnb; F = nF;
            end
            logProb = CTCBeamSearch.lse(Pb, Pnb).';
            score = logProb + F.';
            [~, order] = sort(score, 'descend');
            hyps = struct('Labels', P(order), 'LogProb', num2cell(logProb(order)), ...
                'Score', num2cell(score(order)));
        end

        function out = decodeBatch(logProbs, lengths, varargin)
            % DECODEBATCH  logProbs (K x N x T), lengths (1 x N, default T) ->
            %   1 x N cell of N-best struct arrays (decode() on each valid prefix).
            LP = OceanDL.Autograd.Tensor.raw(logProbs);
            [K, N, T] = size(LP, 1:3);
            if nargin < 2 || isempty(lengths), lengths = T * ones(1, N); end
            if numel(lengths) ~= N || any(lengths < 0 | lengths > T | lengths ~= round(lengths))
                error('OceanDL:Recurrent:CTCBeamSearch:BadLengths', ...
                    'lengths must be N = %d integers in [0, %d].', N, T);
            end
            out = cell(1, N);
            for n = 1:N
                out{n} = OceanDL.Recurrent.CTCBeamSearch.decode( ...
                    reshape(LP(:, n, 1:lengths(n)), K, lengths(n)), varargin{:});
            end
        end
    end

    methods (Static, Access = private)
        function validate(LP, opts)
            if ~ismatrix(LP) || any(isnan(LP(:))) || any(LP(:) == inf)
                error('OceanDL:Recurrent:CTCBeamSearch:BadInput', ...
                    'logProbs must be a K x T matrix of log-probabilities (no NaN / +Inf).');
            end
            K = size(LP, 1);
            if opts.Blank > K
                error('OceanDL:Recurrent:CTCBeamSearch:BadBlank', 'Blank = %d exceeds K = %d.', opts.Blank, K);
            end
            if ~isempty(LP)
                colLse = OceanDL.Recurrent.CTCBeamSearch.lseColumns(LP);
                if ~all(abs(colLse) <= 1e-3)                     % also catches an all -Inf (NaN) column
                    error('OceanDL:Recurrent:CTCBeamSearch:NotLogProbabilities', ...
                        'Columns must be log-probabilities (logsumexp = 0); max |logsumexp| = %.3g. Apply log-softmax first.', ...
                        max(abs(colLse)));
                end
            end
            if ~isempty(opts.PrefixScorer) && ~isa(opts.PrefixScorer, 'function_handle')
                error('OceanDL:Recurrent:CTCBeamSearch:BadScorer', ...
                    'PrefixScorer must be a function handle @(prefix, label) -> log score.');
            end
        end

        function r = lse(a, b)
            % log(exp(a) + exp(b)), elementwise, -inf safe.
            m = max(a, b);
            r = m + log(exp(a - m) + exp(b - m));
            r(m == -inf) = -inf;
        end

        function r = lseColumns(X)
            m = max(X, [], 1);
            r = m + log(sum(exp(X - m), 1));
        end
    end
end
