classdef ShannonTextGenerator
    % SHANNONTEXTGENERATOR Compiles N-gram transition probabilities and generates text.
    %
    % This class analyzes a sanitized text corpus, builds transition tables
    % for order 0, 1, 2, and 3 models, computes the analytical conditional
    % entropies, and allows generating text based on these distributions.
    %
    % See also: OceanMath.Information.Processes.CorpusSanitizer,
    %           OceanMath.Information.Processes.MonteCarloSearchEngine

    properties
        Corpus char
        Alphabet (1, 27) char = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',' ']
        UnigramProbs (1, 27) double
        BigramProbs (27, 27) double
        TrigramProbs (27, 27, 27) double
        Entropies (1, 4) double
    end

    methods
        function obj = ShannonTextGenerator(corpus)
            % SHANNONTEXTGENERATOR Constructor.
            %
            % Sanitizes corpus, counts N-gram occurrences, builds normalized 
            % probability distributions, and computes the 4 analytical entropies.
            arguments
                corpus char
            end

            % Sanitize corpus using CorpusSanitizer
            obj.Corpus = OceanMath.Information.Processes.CorpusSanitizer.sanitize(corpus);
            N = length(obj.Corpus);

            % Pre-allocate alphabet and index conversion
            indices = obj.charToIdx(obj.Corpus);

            % Count occurrences
            if N == 0
                count1 = zeros(1, 27);
                count2 = zeros(27, 27);
                count3 = zeros(27, 27, 27);
            else
                count1 = accumarray(indices', 1, [27, 1])';
                if N >= 2
                    count2 = accumarray([indices(1:end-1)', indices(2:end)'], 1, [27, 27]);
                else
                    count2 = zeros(27, 27);
                end
                if N >= 3
                    count3 = accumarray([indices(1:end-2)', indices(2:end-1)', indices(3:end)'], 1, [27, 27, 27]);
                else
                    count3 = zeros(27, 27, 27);
                end
            end

            % 1. Unigram probabilities
            if N > 0
                obj.UnigramProbs = count1 / N;
            else
                obj.UnigramProbs = ones(1, 27) / 27;
            end

            % 2. Bigram transition probabilities (conditional)
            % p(x_i | x_{i-1}) -> row: x_{i-1}, col: x_i
            obj.BigramProbs = zeros(27, 27);
            for j = 1:27
                row_sum = sum(count2(j, :));
                if row_sum > 0
                    obj.BigramProbs(j, :) = count2(j, :) / row_sum;
                end
            end

            % 3. Trigram transition probabilities (conditional)
            % p(x_i | x_{i-1}, x_{i-2}) -> dim1: x_{i-2}, dim2: x_{i-1}, dim3: x_i
            obj.TrigramProbs = zeros(27, 27, 27);
            for j = 1:27
                for k = 1:27
                    slice_sum = sum(count3(j, k, :));
                    if slice_sum > 0
                        obj.TrigramProbs(j, k, :) = count3(j, k, :) / slice_sum;
                    end
                end
            end

            % 4. Compute Entropies
            % H_0: Zero-Order (Uniform)
            H_0 = log2(27);

            % H_1: First-Order
            H_1 = -sum(obj.UnigramProbs .* log2(obj.UnigramProbs + 1e-12));

            % H_2: Second-Order (Bigram conditional)
            if N >= 2
                joint2 = count2 / (N - 1);
                H_2 = -sum(joint2(:) .* log2(obj.BigramProbs(:) + 1e-12));
            else
                H_2 = 0.0;
            end

            % H_3: Third-Order (Trigram conditional)
            if N >= 3
                joint3 = count3 / (N - 2);
                H_3 = -sum(joint3(:) .* log2(obj.TrigramProbs(:) + 1e-12));
            else
                H_3 = 0.0;
            end

            obj.Entropies = [H_0, H_1, H_2, H_3];
        end

        function [nextChar, candidateProbs] = generateNextAnalytical(obj, context, order)
            % GENERATENEXTANALYTICAL Samples next character using transition tables.
            %
            % context: Character array of history.
            % order: 0, 1, 2, or 3.
            arguments
                obj (1,1) OceanMath.Information.Processes.ShannonTextGenerator
                context char
                order (1,1) double {mustBeInteger, mustBeMember(order, [0, 1, 2, 3])}
            end

            % Validate context based on order
            if order == 0 || order == 1
                if ~isempty(context)
                    error('OceanMath:Information:ShannonTextGenerator:InvalidContext', ...
                        'Context must be empty for order 0 and 1.');
                end
            elseif order == 2
                if length(context) ~= 1
                    error('OceanMath:Information:ShannonTextGenerator:InvalidContext', ...
                        'Context must be of length 1 for order 2 (Bigram).');
                end
            elseif order == 3
                if length(context) ~= 2
                    error('OceanMath:Information:ShannonTextGenerator:InvalidContext', ...
                        'Context must be of length 2 for order 3 (Trigram).');
                end
            end

            % Normalize context characters to uppercase
            context = upper(context);

            % Build candidate probability vector
            switch order
                case 0
                    candidateProbs = ones(1, 27) / 27;
                case 1
                    candidateProbs = obj.UnigramProbs;
                case 2
                    idx1 = obj.charToIdx(context(1));
                    candidateProbs = obj.BigramProbs(idx1, :);
                    % Hierarchical fallback
                    if sum(candidateProbs) == 0
                        candidateProbs = obj.UnigramProbs;
                    end
                case 3
                    idx1 = obj.charToIdx(context(1)); % x_{i-2}
                    idx2 = obj.charToIdx(context(2)); % x_{i-1}
                    candidateProbs = reshape(obj.TrigramProbs(idx1, idx2, :), 1, 27);
                    % Hierarchical fallback to order 2
                    if sum(candidateProbs) == 0
                        candidateProbs = obj.BigramProbs(idx2, :);
                    end
                    % Fallback to order 1
                    if sum(candidateProbs) == 0
                        candidateProbs = obj.UnigramProbs;
                    end
            end

            % Absolute fallback to uniform if still all-zero (e.g. empty corpus distributions)
            if sum(candidateProbs) == 0
                candidateProbs = ones(1, 27) / 27;
            end

            % Normalize candidateProbs to sum to exactly 1.0
            candidateProbs = candidateProbs / sum(candidateProbs);

            % Sample next character
            cdf = cumsum(candidateProbs);
            cdf = cdf / cdf(end); % Normalize for numerical precision
            r = rand();
            idx = find(r <= cdf, 1, 'first');
            nextChar = obj.idxToChar(idx);
        end

        function idx = charToIdx(obj, c)
            % CHARTOIDX Maps A-Z to 1-26, and space to 27.
            arguments
                obj
                c char
            end
            idx = zeros(size(c));
            isSpace = (c == ' ');
            isLetter = (c >= 'A' & c <= 'Z');
            idx(isSpace) = 27;
            idx(isLetter) = double(c(isLetter) - 'A' + 1);
        end

        function c = idxToChar(obj, idx)
            % IDXTOCHAR Maps 1-26 to A-Z, and 27 to space.
            arguments
                obj
                idx double
            end
            c = char(zeros(size(idx)));
            isSpace = (idx == 27);
            isLetter = (idx >= 1 & idx <= 26);
            c(isSpace) = ' ';
            c(isLetter) = char(idx(isLetter) + 'A' - 1);
        end
    end
end
