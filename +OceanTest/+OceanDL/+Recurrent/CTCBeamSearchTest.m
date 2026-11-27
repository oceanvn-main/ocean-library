classdef CTCBeamSearchTest < OceanTest.BaseTestCase
    % CTCBEAMSEARCHTEST  Prefix beam search proven against EXHAUSTIVE enumeration:
    %   for small K, T every one of the K^T alignment paths is collapsed
    %   (merge repeats, drop blanks) and its probability added to its
    %   labelling, giving the exact p(l | x) of every labelling. Then
    %   (a) a beam wide enough to hold every labelling returns exactly those
    %       labellings with LogProb = log p(l | x), best first;
    %   (b) on cases FOUND by search (best path ~= best labelling) the beam
    %       recovers the exact best labelling and greedy does not;
    %   plus shallow fusion (Score decomposition, fused argmax), pruning and
    %   narrow beams (lower bounds), empty input, batching, validation.

    methods (Test)
        function testFullBeamReproducesExactLabellingDistribution(testCase)
            import OceanDL.Recurrent.CTCBeamSearch
            for cfg = {[3 5 1], [4 4 1], [4 4 3], [3 6 2]}               % K, T, Blank
                K = cfg{1}(1); T = cfg{1}(2); blank = cfg{1}(3);
                for seed = 1:3
                    LP = testCase.randomLogProbs(K, T, seed, 1.5);
                    [labs, logp] = testCase.exactLabellings(LP, blank);
                    hyps = CTCBeamSearch.decode(LP, 'Blank', blank, 'BeamWidth', numel(labs) + 5);
                    tag = sprintf('K=%d T=%d blank=%d seed=%d', K, T, blank, seed);
                    testCase.verifyEqual(numel(hyps), numel(labs), tag);       % every labelling, none extra
                    for h = 1:numel(hyps)
                        j = find(cellfun(@(l) isequal(l, hyps(h).Labels), labs));
                        testCase.verifyNumElements(j, 1, tag);
                        testCase.verifyEqual(hyps(h).LogProb, logp(j), 'AbsTol', 1e-11, tag);
                        testCase.verifyEqual(hyps(h).Score, hyps(h).LogProb, tag);   % no fusion
                    end
                    [~, best] = max(logp);
                    testCase.verifyEqual(hyps(1).Labels, labs{best}, tag);
                    testCase.verifyTrue(issorted([hyps.Score], 'descend'), tag);
                    testCase.verifyEqual(sum(exp([hyps.LogProb])), 1, 'AbsTol', 1e-12, tag);   % a distribution
                end
            end
        end

        function testBeamBeatsGreedyWhenBestPathIsNotBestLabelling(testCase)
            % Search seeded random posteriors for cases where the collapsed best
            % path differs from the exact most probable labelling.
            import OceanDL.Recurrent.CTCBeamSearch
            K = 3; T = 5; found = 0;
            for seed = 1:400
                LP = testCase.randomLogProbs(K, T, seed, 1.0);
                [labs, logp] = testCase.exactLabellings(LP, 1);
                [~, best] = max(logp);
                greedy = OceanDL.Recurrent.CTCDecoder.greedy(reshape(LP, K, 1, T), 1);
                greedy = greedy{1};
                if isequal(greedy, labs{best})
                    continue
                end
                found = found + 1;
                hyps = CTCBeamSearch.decode(LP, 'BeamWidth', 8);          % default width
                jg = find(cellfun(@(l) isequal(l, greedy), labs));
                testCase.verifyEqual(hyps(1).Labels, labs{best}, sprintf('seed %d', seed));
                testCase.verifyEqual(hyps(1).LogProb, logp(best), 'AbsTol', 1e-11);
                testCase.verifyGreaterThan(hyps(1).LogProb, logp(jg));   % strictly better than greedy's labelling
                if found >= 5, break; end
            end
            testCase.verifyGreaterThanOrEqual(found, 5, 'search found too few best-path ~= best-labelling cases');
        end

        function testNarrowBeamAndPruningGiveLowerBounds(testCase)
            import OceanDL.Recurrent.CTCBeamSearch
            LP = testCase.randomLogProbs(4, 6, 9, 1.2);
            [labs, logp] = testCase.exactLabellings(LP, 1);
            for opts = {{'BeamWidth', 2}, {'BeamWidth', 3}, {'BeamWidth', 50, 'Prune', log(0.2)}}
                hyps = CTCBeamSearch.decode(LP, opts{1}{:});
                testCase.verifyLessThanOrEqual(numel(hyps), opts{1}{2});
                for h = 1:numel(hyps)
                    j = cellfun(@(l) isequal(l, hyps(h).Labels), labs);
                    testCase.verifyLessThanOrEqual(hyps(h).LogProb, logp(j) + 1e-12);   % lost paths only lower it
                end
            end
        end

        function testShallowFusionScoreDecompositionAndFusedArgmax(testCase)
            % Bigram "LM" over labels (random, seeded, normalised): the fused beam
            % must rank by  log p_ctc(l|x) + Weight * sum log p_lm + Bonus * |l|.
            import OceanDL.Recurrent.CTCBeamSearch
            K = 4; T = 5; w = 0.8; bonus = 0.4;
            s = RandStream('mt19937ar', 'Seed', 21);
            B = rand(s, K, K) + 0.05;  B(:, 1) = 0;                         % column 1 = blank: never a label
            B = log(B ./ sum(B, 2));                                        % row = previous label (1 = start)
            scorer = @(prefix, c) B(testCase.lastOr(prefix, 1), c);
            LP = testCase.randomLogProbs(K, T, 22, 1.0);
            [labs, logp] = testCase.exactLabellings(LP, 1);
            fusion = cellfun(@(l) w * testCase.lmScore(B, l) + bonus * numel(l), labs);
            hyps = CTCBeamSearch.decode(LP, 'BeamWidth', numel(labs) + 1, 'PrefixScorer', scorer, ...
                'Weight', w, 'Bonus', bonus);
            for h = 1:numel(hyps)
                j = cellfun(@(l) isequal(l, hyps(h).Labels), labs);
                testCase.verifyEqual(hyps(h).LogProb, logp(j), 'AbsTol', 1e-11);       % still pure CTC
                testCase.verifyEqual(hyps(h).Score, logp(j) + fusion(j), 'AbsTol', 1e-11);
            end
            [~, best] = max(logp + fusion);
            testCase.verifyEqual(hyps(1).Labels, labs{best});
        end

        function testScorerIsOffByDefault(testCase)
            import OceanDL.Recurrent.CTCBeamSearch
            LP = testCase.randomLogProbs(4, 6, 3, 1.0);
            failing = @(prefix, c) testCase.failScorer();
            plain = CTCBeamSearch.decode(LP);
            withScorer = CTCBeamSearch.decode(LP, 'PrefixScorer', failing);   % Weight = 0 -> never called
            testCase.verifyEqual(withScorer, plain);
            testCase.verifyError(@() CTCBeamSearch.decode(LP, 'PrefixScorer', failing, 'Weight', 1), ...
                'test:scorerCalled');
        end

        function testEmptyInputAndBatchDecoding(testCase)
            import OceanDL.Recurrent.CTCBeamSearch
            h = CTCBeamSearch.decode(zeros(5, 0));
            testCase.verifyEqual(h, struct('Labels', {zeros(1, 0)}, 'LogProb', 0, 'Score', 0));   % p(empty | T = 0) = 1
            K = 4; T = 6;
            LP = zeros(K, 3, T);
            for n = 1:3, LP(:, n, :) = reshape(testCase.randomLogProbs(K, T, 30 + n, 1.0), K, 1, T); end
            lens = [6 0 3];
            out = CTCBeamSearch.decodeBatch(OceanDL.Autograd.Tensor(LP), lens, 'BeamWidth', 4);
            for n = 1:3
                one = CTCBeamSearch.decode(reshape(LP(:, n, 1:lens(n)), K, lens(n)), 'BeamWidth', 4);
                testCase.verifyEqual(out{n}, one);
            end
        end

        function testRejectsInvalidInput(testCase)
            import OceanDL.Recurrent.CTCBeamSearch
            testCase.verifyError(@() CTCBeamSearch.decode(randn(4, 5)), ...
                'OceanDL:Recurrent:CTCBeamSearch:NotLogProbabilities');      % raw logits, not log-softmax
            testCase.verifyError(@() CTCBeamSearch.decode(log(ones(3, 2) / 3), 'Blank', 4), ...
                'OceanDL:Recurrent:CTCBeamSearch:BadBlank');
            testCase.verifyError(@() CTCBeamSearch.decode(log(ones(3, 2, 2) / 3)), ...
                'OceanDL:Recurrent:CTCBeamSearch:BadInput');
        end
    end

    methods (Static, Access = private)
        function LP = randomLogProbs(K, T, seed, scale)
            s = RandStream('mt19937ar', 'Seed', seed);
            Z = scale * randn(s, K, T);
            LP = Z - (max(Z, [], 1) + log(sum(exp(Z - max(Z, [], 1)), 1)));
        end

        function [labs, logp] = exactLabellings(LP, blank)
            % Enumerate all K^T paths; p(l|x) = sum of path probabilities collapsing to l.
            [K, T] = size(LP);
            map = containers.Map('KeyType', 'char', 'ValueType', 'double');
            labs = {}; p = [];
            for code = 0:K^T - 1
                path = mod(floor(code ./ K.^(0:T-1)), K) + 1;
                c = path([true, path(2:end) ~= path(1:end-1)]);
                c = reshape(c(c ~= blank), 1, []);                  % 1 x 0 for the empty labelling
                key = sprintf('%d,', c);
                pp = exp(sum(LP(sub2ind([K T], path, 1:T))));
                if isKey(map, key)
                    p(map(key)) = p(map(key)) + pp; %#ok<AGROW>
                else
                    labs{end+1} = c; p(end+1) = pp; map(key) = numel(p); %#ok<AGROW>
                end
            end
            logp = log(p);
        end

        function v = failScorer()
            v = 0;
            error('test:scorerCalled', 'scorer must not be called');
        end

        function v = lastOr(prefix, default)
            if isempty(prefix), v = default; else, v = prefix(end); end
        end

        function s = lmScore(B, l)
            if isempty(l), s = 0; return; end
            prev = [1, l(1:end-1)];
            s = sum(B(sub2ind(size(B), prev, l)));
        end
    end
end
