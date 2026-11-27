classdef CTCPrefixBeamDecoderTest < OceanTest.BaseTestCase
    % CTCPREFIXBEAMDECODERTEST  Speech CTC decoding adapter + token n-gram LM, checked against
    %   brute-force enumeration of CTC paths and closed-form counts.

    properties
        Vocab       % 5 specials + units {a, b, |}  -> K = 8
    end

    methods (TestClassSetup)
        function build(testCase)
            testCase.Vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(["ab", "ba a"], 'Unit', "char");
        end
    end

    methods (Test)
        function testGreedyAgainstIndependentCollapse(testCase)
            v = testCase.Vocab; K = v.numClasses();
            rs = RandStream('mt19937ar', 'Seed', 1);
            Z = 3 * randn(rs, K, 3, 9);
            lens = [9 5 0];
            tr = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(v, 'Method', "greedy", 'ModelId', "m1").decode(Z, lens);
            testCase.verifySize(tr, [1 3]);
            for n = 1:3
                z = reshape(Z(:, n, 1:lens(n)), K, lens(n));
                [~, path] = max(z, [], 1);
                lab = [];
                prev = -1;
                for t = 1:numel(path)
                    if path(t) ~= prev && path(t) ~= v.Blank, lab(end+1) = path(t); end %#ok<AGROW>
                    prev = path(t);
                end
                lp = z - log(sum(exp(z), 1));
                testCase.verifyEqual(tr(n).TokenIds, reshape(lab, 1, []));
                testCase.verifyEqual(tr(n).Text, v.decode(lab));
                testCase.verifyEqual(tr(n).Score, sum(max(lp, [], 1)), 'AbsTol', 1e-12);
                testCase.verifyEqual(tr(n).Decoder, "ctc-greedy");
                testCase.verifyEqual(tr(n).ModelId, "m1");
                testCase.verifyEmpty(tr(n).Segments);
                testCase.verifyEqual(tr(n).Language, "");
            end
        end

        function testBeamEqualsBruteForce(testCase)
            v = testCase.Vocab; K = v.numClasses(); T = 4;
            rs = RandStream('mt19937ar', 'Seed', 2);
            Z = 2 * randn(rs, K, 1, T);
            exact = OceanTest.OceanNLP.Speech.Decoding.CTCPrefixBeamDecoderTest.bruteForce( ...
                reshape(Z, K, T), v.Blank);
            % beam wider than the number of labellings: exact best labelling and exact log p
            tr = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(v, 'BeamWidth', 4000, 'NBest', 5).decode(Z, T);
            keysSorted = exact.keys; [~, o] = sort(exact.logp, 'descend');
            testCase.verifyEqual(tr.TokenIds, keysSorted{o(1)});
            testCase.verifyEqual(tr.Score, exact.logp(o(1)), 'AbsTol', 1e-10);
            testCase.verifyNumElements(tr.Candidates, 5);
            for h = 1:5
                c = tr.Candidates(h);
                j = find(cellfun(@(k) isequal(k, c.TokenIds), exact.keys));
                testCase.verifyEqual(c.Score, exact.logp(j), 'AbsTol', 1e-10);
                testCase.verifyEqual(c.Text, v.decode(c.TokenIds));
            end
            testCase.verifyEqual(tr.Decoder, "ctc-beam");
            testCase.verifyEqual(tr.DecoderOptions.BeamWidth, 4000);
            testCase.verifyEmpty(tr.Segments);
        end

        function testPaddingIgnored(testCase)
            v = testCase.Vocab; K = v.numClasses();
            rs = RandStream('mt19937ar', 'Seed', 3);
            Z = randn(rs, K, 2, 7);
            lens = [7 4];
            Zg = Z; Zg(:, 2, 5:7) = 50 * randn(rs, K, 1, 3);      % garbage past the valid length
            for method = ["greedy", "beam"]
                d = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(v, 'Method', method, 'BeamWidth', 6);
                a = d.decode(Z, lens); b = d.decode(Zg, lens);
                one = d.decode(Z(:, 2, 1:4), 4);
                testCase.verifyEqual(b.texts(), a.texts());
                testCase.verifyEqual([b.Score], [a.Score]);
                testCase.verifyEqual(a(2).Score, one.Score);
            end
        end

        function testLMFusionScoreDecomposition(testCase)
            v = testCase.Vocab; K = v.numClasses();
            lm = OceanNLP.Speech.Decoding.TokenNGramLM.fit(v.encode(["ab", "ba a", "abab"]), v, 'Order', 2, 'Alpha', 0.5);
            rs = RandStream('mt19937ar', 'Seed', 4);
            Z = randn(rs, K, 1, 4);
            w = 0.7; bonus = 0.3;
            tr = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(v, 'BeamWidth', 4000, 'NBest', 4, ...
                'LanguageModel', lm, 'LMWeight', w, 'Bonus', bonus).decode(Z, 4);
            exact = OceanTest.OceanNLP.Speech.Decoding.CTCPrefixBeamDecoderTest.bruteForce(reshape(Z, K, 4), v.Blank);
            for h = 1:numel(tr.Candidates)
                lab = tr.Candidates(h).TokenIds;
                j = find(cellfun(@(k) isequal(k, lab), exact.keys));
                lmSum = 0;
                for i = 1:numel(lab)
                    lmSum = lmSum + lm.logProb(lab(1:i-1), lab(i));
                end
                testCase.verifyEqual(tr.Candidates(h).Score, exact.logp(j) + w * lmSum + bonus * numel(lab), 'AbsTol', 1e-9);
            end
            testCase.verifyEqual(tr.DecoderOptions.LanguageModel.Type, "TokenNGramLM");
        end

        function testTokenNGramCounts(testCase)
            v = testCase.Vocab;
            seqs = v.encode(["ab", "ba a"]);          % chars: a b | ...
            lm = OceanNLP.Speech.Decoding.TokenNGramLM.fit(seqs, v, 'Order', 2, 'Alpha', 1);
            a = v.encode("a"); b = v.encode("b");
            V = v.numClasses() - 5;
            % bigram counts by hand: contexts (bos,a) (a,b) | (bos,b) (b,a) (a,|) (|,a)
            % c(a -> b) = 1, c(a -> *) = 2  =>  p(b | a) = (1 + 1) / (2 + V)
            testCase.verifyEqual(lm.logProb(a, b), log(2 / (2 + V)), 'AbsTol', 1e-14);
            % c(bos -> a) = 1 of 2 sentence starts
            testCase.verifyEqual(lm.logProb([], a), log(2 / (2 + V)), 'AbsTol', 1e-14);
            % unseen context: uniform 1/V
            testCase.verifyEqual(lm.logProb(b, b), log(1 / (1 + V)), 'AbsTol', 1e-14);   % c(b->*) = 1
            testCase.verifyError(@() OceanNLP.Speech.Decoding.TokenNGramLM.fit({[1 6]}, v), ...
                'OceanNLP:Speech:Decoding:TokenNGramLM:Ids');
        end

        function testErrors(testCase)
            v = testCase.Vocab;
            testCase.verifyError(@() OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(v, 'Method', "greedy", 'Bonus', 1), ...
                'OceanNLP:Speech:Decoding:CTCPrefixBeamDecoder:GreedyFusion');
            testCase.verifyError(@() OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(v, 'LMWeight', 1), ...
                'OceanNLP:Speech:Decoding:CTCPrefixBeamDecoder:LanguageModel');
            d = OceanNLP.Speech.Decoding.CTCPrefixBeamDecoder(v);
            testCase.verifyError(@() d.decode(randn(v.numClasses() + 1, 1, 3), 3), ...
                'OceanNLP:Speech:Data:SpeechVocabulary:Incompatible');
            testCase.verifyError(@() d.decode(randn(v.numClasses(), 2, 3), [3 4]), ...
                'OceanNLP:Speech:Decoding:CTCPrefixBeamDecoder:Lengths');
        end
    end

    methods (Static)
        function r = bruteForce(z, blank)
            % Exact log p(labelling | x) by enumerating all K^T paths and collapsing them.
            [K, T] = size(z);
            lp = z - log(sum(exp(z), 1));
            keys = {}; mass = [];
            idx = ones(1, T);
            for p = 1:K^T
                path = idx;
                lab = path([true, path(2:end) ~= path(1:end-1)]);
                lab = lab(lab ~= blank);
                s = exp(sum(lp(sub2ind([K T], path, 1:T))));
                j = find(cellfun(@(k) isequal(k, lab), keys), 1);
                if isempty(j)
                    keys{end+1} = lab; mass(end+1) = s; %#ok<AGROW>
                else
                    mass(j) = mass(j) + s;
                end
                for t = T:-1:1                     % next path (odometer)
                    idx(t) = idx(t) + 1;
                    if idx(t) <= K, break; end
                    idx(t) = 1;
                end
            end
            keys = cellfun(@(k) reshape(double(k), 1, []), keys, 'UniformOutput', false);
            r = struct('keys', {keys}, 'logp', log(mass));
        end
    end
end
