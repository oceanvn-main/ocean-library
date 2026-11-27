classdef test_shannon_processes < OceanTest.BaseTestCase
    % TEST_SHANNON_PROCESSES Unit tests for Shannon Approximation tools.
    %
    % These tests verify that the text sanitization, conditional entropy decay,
    % Monte Carlo search, and analytical text generation models behave in
    % accordance with Shannon's original 1948 information theory paper.
    %
    % See also: OceanMath.Information.Processes.CorpusSanitizer,
    %           OceanMath.Information.Processes.ShannonTextGenerator,
    %           OceanMath.Information.Processes.MonteCarloSearchEngine

    methods (Test)
        function test_corpus_sanitization(testCase)
            % Verify that raw strings with numbers, punctuation, carriage returns,
            % symbols, and double spaces normalize strictly to A-Z and space.
            raw = sprintf('Hello,\r\n  World! 123;  Testing... \t#Symbol.');
            clean = OceanMath.Information.Processes.CorpusSanitizer.sanitize(raw);

            % 1. Characters must be strictly uppercase letters or spaces
            for c = clean
                testCase.verifyTrue((c >= 'A' && c <= 'Z') || c == ' ', ...
                    'Sanitized character must be uppercase letter or space.');
            end

            % 2. No consecutive double spaces
            testCase.verifyFalse(contains(clean, '  '), ...
                'Consecutive double spaces must be compressed.');

            % 3. Check specific sanitization logic
            simpleRaw = 'Testing 1, 2, 3... can you hear me?';
            % Expected: 'TESTING CAN YOU HEAR ME'
            % Let's see: 'Testing 1, 2, 3... can you hear me?' ->
            % 'TESTING 1, 2, 3... CAN YOU HEAR ME?'
            % Replace digits, punctuation -> 'TESTING          CAN YOU HEAR ME '
            % Strip non-A-Z/space: no-op
            % Compress spaces -> 'TESTING CAN YOU HEAR ME '
            simpleClean = OceanMath.Information.Processes.CorpusSanitizer.sanitize(simpleRaw);
            testCase.verifyEqual(simpleClean, 'TESTING CAN YOU HEAR ME ');
        end

        function test_analytical_entropies(testCase)
            % Asserts H_0 = 4.7549 and that H_0 >= H_1 >= H_2 >= H_3 for a general corpus,
            % validating the law of conditional uncertainty reduction.
            corpusText = 'THE CAT IN THE HAT SAT ON THE MAT WITH A RAT AND A BAT';
            gen = OceanMath.Information.Processes.ShannonTextGenerator(corpusText);

            % Verify H_0 matches log2(27)
            testCase.verifyEqual(gen.Entropies(1), log2(27), 'AbsTol', 1e-4);

            % Verify monotonic decay of conditional entropies (H_0 >= H_1 >= H_2 >= H_3)
            testCase.verifyGreaterThanOrEqual(gen.Entropies(1), gen.Entropies(2), ...
                'Zero-order entropy H_0 must be >= First-order H_1.');
            testCase.verifyGreaterThanOrEqual(gen.Entropies(2), gen.Entropies(3), ...
                'First-order entropy H_1 must be >= Second-order H_2.');
            testCase.verifyGreaterThanOrEqual(gen.Entropies(3), gen.Entropies(4), ...
                'Second-order entropy H_2 must be >= Third-order H_3.');

            % Log the results without using LaTeX formatting
            fprintf('Entropy decay verification:\n');
            fprintf('  H(0) = %.4f bits\n', gen.Entropies(1));
            fprintf('  H(1) = %.4f bits\n', gen.Entropies(2));
            fprintf('  H(2) = %.4f bits\n', gen.Entropies(3));
            fprintf('  H(3) = %.4f bits\n', gen.Entropies(4));
        end

        function test_monte_carlo_search_correctness(testCase)
            % Verifies that searching for a context (e.g., 'T') finds it,
            % wraps around properly, and extracts the correct subsequent character.
            corpusText = 'ABCDE';
            engine = OceanMath.Information.Processes.MonteCarloSearchEngine(corpusText);

            % 1. Search for 'E' (at the end). The next character must wrap around to 'A'.
            for i = 1:10
                [nextChar, logData] = engine.searchNext('E');
                testCase.verifyEqual(nextChar, 'A', 'Character after E must wrap around to A.');
                testCase.verifyEqual(logData.foundChar, 'A');
            end

            % 2. Search for 'B'. The next character must be 'C'.
            for i = 1:10
                [nextChar, logData] = engine.searchNext('B');
                testCase.verifyEqual(nextChar, 'C', 'Character after B must be C.');
                testCase.verifyEqual(logData.foundChar, 'C');
            end
        end

        function test_monte_carlo_equivalence(testCase)
            % Compares frequencies of 2000 draws using MonteCarloSearchEngine
            % against the analytical conditional probability slice from ShannonTextGenerator.
            corpusText = 'A B A B A B C A B A B C';
            sanitized = OceanMath.Information.Processes.CorpusSanitizer.sanitize(corpusText);

            gen = OceanMath.Information.Processes.ShannonTextGenerator(sanitized);
            engine = OceanMath.Information.Processes.MonteCarloSearchEngine(sanitized);

            % Analytical transition probabilities for context 'A' (Bigram, Order 2)
            [~, analyticalProbs] = gen.generateNextAnalytical('A', 2);

            % Draw 2000 samples using the Monte Carlo search engine
            numDraws = 2000;
            draws = char(zeros(1, numDraws));
            for i = 1:numDraws
                draws(i) = engine.searchNext('A');
            end

            % Compute empirical probabilities
            empiricalProbs = zeros(1, 27);
            for k = 1:27
                charK = gen.Alphabet(k);
                empiricalProbs(k) = sum(draws == charK) / numDraws;
            end

            % Verify maximum absolute error is bounded by statistical fluctuation (0.04 is ~3.6 standard deviations)
            maxDiff = max(abs(empiricalProbs - analyticalProbs));
            testCase.verifyLessThan(maxDiff, 0.04, ...
                sprintf('Empirical probabilities deviate too much from analytical (Max Diff: %.4f).', maxDiff));
        end
    end
end
