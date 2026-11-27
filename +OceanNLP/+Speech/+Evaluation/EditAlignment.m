classdef EditAlignment
    % EDITALIGNMENT  Levenshtein alignment of a reference and a hypothesis token sequence.
    %
    %   a = OceanNLP.Speech.Evaluation.EditAlignment.align(ref, hyp)
    %   c = OceanNLP.Speech.Evaluation.EditAlignment.corpusCounts(refTokens, hypTokens)
    %
    %   ref, hyp: token arrays -- both string arrays (words, characters) or both
    %   numeric arrays (token ids, Unicode code points). Empty arrays are allowed.
    %
    %   DISTANCE (Levenshtein 1966; Jurafsky & Martin, SLP3 sec. 2.8):
    %
    %     D(i,0) = i,   D(0,j) = j
    %     D(i,j) = min( D(i-1,j) + 1,                 deletion of r_i
    %                   D(i,j-1) + 1,                 insertion of h_j
    %                   D(i-1,j-1) + [r_i ~= h_j] )   hit / substitution
    %
    %   and the alignment is one minimum-cost path, so S + D + I = D(m,n) and
    %   H + S + D = m (reference length), H + S + I = n (hypothesis length).
    %
    %   DETERMINISTIC TIE-BREAK (reproduces rapidfuzz.distance.Levenshtein.opcodes,
    %   which jiwer 4.x uses, so the S/D/I split equals jiwer's, not only the total):
    %     1. the longest common prefix and suffix of ref and hyp are hits;
    %     2. D is filled on the remaining cores; the path is traced back from
    %        (m', n') with this priority at each cell (i, j), i, j > 0:
    %          a. deletion     if D(i,j) - D(i-1,j) = +1
    %          b. insertion    if j > 1 and D(i,j-1) - D(i-1,j-1) = -1
    %          c. otherwise the diagonal (hit if r_i = h_j, else substitution);
    %        once i = 0 the rest are insertions, once j = 0 the rest deletions.
    %   (Rule b inspects column j-1: it is the rapidfuzz bit-parallel backtrace
    %   expressed on the explicit DP matrix.) The cost of the traced path is checked
    %   against D(m', n') on every call.
    %
    %   OUTPUT (struct): Ops (1 x K string, "C" hit | "S" | "D" | "I", in reading
    %   order), RefIndex / HypIndex (1 x K, 1-based; 0 where the op consumes no
    %   reference / hypothesis token), Substitutions, Deletions, Insertions, Hits,
    %   Distance, RefLength, HypLength.

    methods (Static)
        function a = align(ref, hyp)
            arguments
                ref
                hyp
            end
            [r, h] = OceanNLP.Speech.Evaluation.EditAlignment.toCodes(ref, hyp);
            m = numel(r); n = numel(h);

            % 1. common prefix / suffix are hits (rapidfuzz affix removal)
            p = 0;
            while p < m && p < n && r(p+1) == h(p+1)
                p = p + 1;
            end
            s = 0;
            while s < m - p && s < n - p && r(m-s) == h(n-s)
                s = s + 1;
            end
            rc = r(p+1:m-s); hc = h(p+1:n-s);
            mc = numel(rc); nc = numel(hc);

            % 2. DP on the cores; D(i+1, j+1) holds D(i, j)
            D = zeros(mc + 1, nc + 1);
            D(:, 1) = (0:mc)';
            D(1, :) = 0:nc;
            for i = 1:mc
                ri = rc(i);
                for j = 1:nc
                    D(i+1, j+1) = min([D(i, j+1) + 1, D(i+1, j) + 1, D(i, j) + (ri ~= hc(j))]);
                end
            end

            % backtrace (ops collected in reverse)
            K = mc + nc;
            ops = strings(1, K); ri = zeros(1, K); hi = zeros(1, K); k = 0;
            i = mc; j = nc;
            while i > 0 && j > 0
                k = k + 1;
                if D(i+1, j+1) - D(i, j+1) == 1
                    ops(k) = "D"; ri(k) = i; hi(k) = 0;
                    i = i - 1;
                elseif j > 1 && D(i+1, j) - D(i, j) == -1
                    ops(k) = "I"; ri(k) = 0; hi(k) = j;
                    j = j - 1;
                else
                    if rc(i) == hc(j), ops(k) = "C"; else, ops(k) = "S"; end
                    ri(k) = i; hi(k) = j;
                    i = i - 1; j = j - 1;
                end
            end
            while i > 0
                k = k + 1; ops(k) = "D"; ri(k) = i; hi(k) = 0; i = i - 1;
            end
            while j > 0
                k = k + 1; ops(k) = "I"; ri(k) = 0; hi(k) = j; j = j - 1;
            end
            ops = fliplr(ops(1:k)); ri = fliplr(ri(1:k)); hi = fliplr(hi(1:k));
            ri(ri > 0) = ri(ri > 0) + p;              % back to full-sequence indices
            hi(hi > 0) = hi(hi > 0) + p;

            % prefix and suffix hits
            ops = [repmat("C", 1, p), ops, repmat("C", 1, s)];
            ri = [1:p, ri, (m-s+1):m];
            hi = [1:p, hi, (n-s+1):n];

            a = struct('Ops', ops, 'RefIndex', ri, 'HypIndex', hi, ...
                'Substitutions', nnz(ops == "S"), 'Deletions', nnz(ops == "D"), ...
                'Insertions', nnz(ops == "I"), 'Hits', nnz(ops == "C"), ...
                'Distance', D(end, end), 'RefLength', m, 'HypLength', n);
            if a.Substitutions + a.Deletions + a.Insertions ~= a.Distance
                error('OceanNLP:Speech:Evaluation:EditAlignment:Internal', ...
                    'Traced path cost %d differs from the edit distance %d.', ...
                    a.Substitutions + a.Deletions + a.Insertions, a.Distance);
            end
        end

        function c = corpusCounts(refTokens, hypTokens)
            % CORPUSCOUNTS  Align each pair; per-utterance and summed S/D/I/H/N.
            %   refTokens, hypTokens: 1 x U cell arrays of token arrays.
            %   c.PerUtterance: table S, D, I, H, N, Errors, Rate (NaN when N = 0),
            %   EmptyReference. c.S ... c.N: corpus sums. Rate is not computed here.
            arguments
                refTokens cell
                hypTokens cell
            end
            U = numel(refTokens);
            if numel(hypTokens) ~= U
                error('OceanNLP:Speech:Evaluation:EditAlignment:CountMismatch', ...
                    '%d references but %d hypotheses.', U, numel(hypTokens));
            end
            S = zeros(U, 1); Dl = zeros(U, 1); I = zeros(U, 1); H = zeros(U, 1);
            for u = 1:U
                a = OceanNLP.Speech.Evaluation.EditAlignment.align(refTokens{u}, hypTokens{u});
                S(u) = a.Substitutions; Dl(u) = a.Deletions; I(u) = a.Insertions; H(u) = a.Hits;
            end
            N = S + Dl + H;
            E = S + Dl + I;
            rate = E ./ N;
            rate(N == 0) = NaN;
            c = struct('S', sum(S), 'D', sum(Dl), 'I', sum(I), 'H', sum(H), 'N', sum(N), ...
                'PerUtterance', table(S, Dl, I, H, N, E, rate, N == 0, 'VariableNames', ...
                ["S", "D", "I", "H", "N", "Errors", "Rate", "EmptyReference"]));
        end
    end

    methods (Static, Access = private)
        function [r, h] = toCodes(ref, hyp)
            % Map both sequences onto one integer alphabet (row vectors).
            if iscellstr(ref) || ischar(ref), ref = string(ref); end %#ok<ISCLSTR>
            if iscellstr(hyp) || ischar(hyp), hyp = string(hyp); end %#ok<ISCLSTR>
            ref = reshape(ref, 1, []); hyp = reshape(hyp, 1, []);
            if isempty(ref) && ~isstring(ref), ref = zeros(1, 0); end
            if isempty(hyp) && ~isstring(hyp), hyp = zeros(1, 0); end
            okType = @(x) isstring(x) || (isnumeric(x) && isreal(x)) || islogical(x);
            if ~okType(ref) || ~okType(hyp)
                error('OceanNLP:Speech:Evaluation:EditAlignment:TokenType', ...
                    'Tokens must be string or real numeric arrays.');
            end
            if isstring(ref) ~= isstring(hyp)
                if isempty(ref) && ~isstring(ref), ref = strings(1, 0);
                elseif isempty(hyp) && ~isstring(hyp), hyp = strings(1, 0);
                else
                    error('OceanNLP:Speech:Evaluation:EditAlignment:TokenType', ...
                        'Reference and hypothesis must both be string or both be numeric.');
                end
            end
            if isstring(ref)
                if any(ismissing([ref, hyp]))
                    error('OceanNLP:Speech:Evaluation:EditAlignment:Missing', 'Tokens must not be <missing>.');
                end
                [~, ~, code] = unique([ref, hyp]);
                code = reshape(code, 1, []);
                r = code(1:numel(ref)); h = code(numel(ref)+1:end);
            else
                r = double(ref); h = double(hyp);
                if any(~isfinite([r, h]))
                    error('OceanNLP:Speech:Evaluation:EditAlignment:Nonfinite', 'Numeric tokens must be finite.');
                end
            end
        end
    end
end
