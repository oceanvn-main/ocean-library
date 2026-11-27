classdef CTCDecoder
    % CTCDECODER  Turn per-step CTC scores into label sequences, and score them.
    %
    %   greedy (best-path) decoding: take the most likely class at every
    %   step, then apply the CTC collapse B: merge adjacent repeats, drop blanks.
    %     argmax path  (-, 3, 3, -, -, 3, 7, 7)  ->  (3, -, 3, 7)  ->  (3, 3, 7)
    %   (Best path is not always the most probable LABELLING -- beam search
    %   sums over paths -- but it is the standard fast decoder and exact when
    %   the per-step distributions are peaked.)
    %
    %   editDistance: Levenshtein distance (insertions, deletions,
    %   substitutions) by dynamic programming, O(|a| |b|).
    %   errorRates:   CER = total edits / total target labels,
    %                 SequenceAccuracy = fraction decoded exactly.

    methods (Static)
        function seqs = greedy(logits, blank, inputLengths)
            arguments
                logits
                blank (1,1) double = 1
                inputLengths (1,:) double = []
            end
            % logits (K x N x T, Tensor or array); inputLengths (1 x N, default T):
            % only steps 1..T_n of example n are decoded; T_n = 0 (or T = 0)
            % gives the empty sequence zeros(1, 0).
            Z = OceanDL.Autograd.Tensor.raw(logits);
            [~, N, T] = size(Z, 1:3);
            [~, best] = max(Z, [], 1);                            % 1 x N x T
            best = reshape(double(best), N, T);
            if isempty(inputLengths), inputLengths = T * ones(1, N); end
            if numel(inputLengths) ~= N || any(inputLengths < 0 | inputLengths > T ...
                    | inputLengths ~= round(inputLengths))
                error('OceanDL:Recurrent:CTCDecoder:BadLengths', ...
                    'inputLengths must be N = %d integers in [0, %d].', N, T);
            end
            seqs = cell(1, N);
            for n = 1:N
                path = best(n, 1:inputLengths(n));
                keep = [true, path(2:end) ~= path(1:end-1)];      % merge repeats
                path = path(keep(1:numel(path)));                 % (keep is 1 x 1 for an empty path)
                seqs{n} = reshape(path(path ~= blank), 1, []);    % drop blanks; always a row
            end
        end

        function d = editDistance(a, b)
            a = a(:).'; b = b(:).';
            D = zeros(numel(a) + 1, numel(b) + 1);
            D(:, 1) = 0:numel(a);
            D(1, :) = 0:numel(b);
            for i = 1:numel(a)
                for j = 1:numel(b)
                    D(i+1, j+1) = min([D(i, j+1) + 1, D(i+1, j) + 1, D(i, j) + (a(i) ~= b(j))]);
                end
            end
            d = D(end, end);
        end

        function r = errorRates(predicted, targets)
            edits = 0; labels = 0; exact = 0;
            for n = 1:numel(targets)
                e = OceanDL.Recurrent.CTCDecoder.editDistance(predicted{n}, targets{n});
                edits = edits + e;
                labels = labels + numel(targets{n});
                exact = exact + (e == 0);
            end
            r = struct('CER', edits / labels, 'SequenceAccuracy', exact / numel(targets));
        end
    end
end
