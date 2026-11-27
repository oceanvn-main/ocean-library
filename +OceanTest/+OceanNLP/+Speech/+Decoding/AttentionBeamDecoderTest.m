classdef AttentionBeamDecoderTest < OceanTest.BaseTestCase
    % ATTENTIONBEAMDECODERTEST  Greedy / beam search over a closed-form toy autoregressive
    %   model (next-token logits = table(last token, position) + per-utterance offset):
    %   beam width 1 == greedy; wide beam == brute-force enumeration of every hypothesis
    %   under the same length-normalised score (several LengthPenalty values, N-best order);
    %   max-length guard; Transcript fields; step-size errors.

    properties
        Vocab
        Table
        Offset
    end

    methods (TestClassSetup)
        function build(testCase)
            testCase.Vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit("ab", 'Unit', "char");   % K = 7
            K = testCase.Vocab.numClasses();
            s = RandStream('mt19937ar', 'Seed', 21);
            testCase.Table = 1.5 * randn(s, K, K, 12);      % logits(next | last, position)
            testCase.Offset = randn(s, K, 3);               % utterance-specific shift
        end
    end

    methods (Test)
        function testBeamWidthOneEqualsGreedy(testCase)
            step = testCase.stepFn();
            for lp = [0 1]
                g = OceanNLP.Speech.Decoding.AttentionBeamDecoder(testCase.Vocab, 'Method', "greedy", ...
                    'LengthPenalty', lp, 'MaxLength', 6).search(step, [5 9 2]);
                b = OceanNLP.Speech.Decoding.AttentionBeamDecoder(testCase.Vocab, 'Method', "beam", 'BeamWidth', 1, ...
                    'LengthPenalty', lp, 'MaxLength', 6).search(step, [5 9 2]);
                testCase.verifyEqual({b.Labels}, {g.Labels});
                testCase.verifyEqual([b.LogProb], [g.LogProb], 'AbsTol', 1e-12);
                testCase.verifyEqual([b.Score], [g.Score], 'AbsTol', 1e-12);
            end
            % greedy against a hand-rolled argmax loop for utterance 2
            r = OceanNLP.Speech.Decoding.AttentionBeamDecoder(testCase.Vocab, 'Method', "greedy", 'MaxLength', 6).search(step, [5 9 2]);
            [ids, logP] = testCase.manualGreedy(2, 6);
            testCase.verifyEqual(r(2).Labels, ids);
            testCase.verifyEqual(r(2).LogProb, logP, 'AbsTol', 1e-12);
        end

        function testWideBeamEqualsBruteForce(testCase)
            step = testCase.stepFn();
            Lmax = 4;
            for alpha = [0 0.5 1 2]
                d = OceanNLP.Speech.Decoding.AttentionBeamDecoder(testCase.Vocab, 'BeamWidth', 64, 'NBest', 5, ...
                    'LengthPenalty', alpha, 'MaxLength', Lmax);
                res = d.search(step, [10 10 10]);
                for n = 1:3
                    [seqs, scores, logPs] = testCase.bruteForce(n, Lmax, alpha);
                    [sc, order] = sort(scores, 'descend');
                    testCase.verifyEqual(res(n).Labels, seqs{order(1)}, sprintf('alpha %g utt %d', alpha, n));
                    testCase.verifyEqual(res(n).Score, sc(1), 'AbsTol', 1e-12);
                    testCase.verifyEqual(res(n).LogProb, logPs(order(1)), 'AbsTol', 1e-12);
                    testCase.verifyEqual([res(n).Candidates.Score], sc(1:5), 'AbsTol', 1e-12);
                    testCase.verifyEqual({res(n).Candidates.TokenIds}, seqs(order(1:5)));
                end
                % a narrow beam can only be worse or equal
                r2 = OceanNLP.Speech.Decoding.AttentionBeamDecoder(testCase.Vocab, 'BeamWidth', 2, ...
                    'LengthPenalty', alpha, 'MaxLength', Lmax).search(step, [10 10 10]);
                testCase.verifyLessThanOrEqual([r2.Score], [res.Score] + 1e-12);
            end
        end

        function testIncrementalStateProtocol(testCase)
            % A stateful step whose state is each row's own token history, reordered by the
            % parent indices: it errors unless the reordering reproduces the prefixes exactly
            % (beam reordering, duplication, pruning, Eos, greedy drop-out), computes the
            % logits from its state only, and must give results identical to the stateless step.
            T = testCase.Table; O = testCase.Offset;
            inc = @(p, src, st, parent) statefulToyStep(T, O, p, src, st, parent);
            stateless = testCase.stepFn();
            for cfg = {{'Method', "greedy"}, {'BeamWidth', 3, 'NBest', 2}, {'BeamWidth', 64, 'NBest', 5}, ...
                       {'BeamWidth', 4, 'LengthPenalty', 0}}
                d = OceanNLP.Speech.Decoding.AttentionBeamDecoder(testCase.Vocab, cfg{1}{:}, 'MaxLength', 5);
                src = [5 9 2];
                a = d.search(stateless, src);
                b = d.search(inc, src, 'Incremental', true, 'State', repmat(testCase.Vocab.Bos, 3, 0));
                testCase.verifyEqual(b, a);
                ta = d.decode(stateless, src);
                tb = d.decode(inc, src, 'Incremental', true, 'State', zeros(3, 0));
                testCase.verifyEqual(tb.texts(), ta.texts());
                testCase.verifyEqual([tb.Score], [ta.Score]);
            end
        end

        function testMaxLengthGuard(testCase)
            v = testCase.Vocab; K = v.numClasses();
            step = @(p, src) repmat([-Inf; -Inf; -Inf; -20; 0; 1; 0.5], 1, size(p, 1)) + 0 * K;   % Eos very unlikely
            d = OceanNLP.Speech.Decoding.AttentionBeamDecoder(v, 'BeamWidth', 3, 'MaxLength', 50, 'MaxLengthRatio', 0.5);
            testCase.verifyEqual(d.maxLengths([10 3 0 1000]), [5 2 1 50]);
            r = d.search(step, [10 3]);
            testCase.verifyEqual([r.Length], [5 2]);
            testCase.verifyTrue(all([r.HitMaxLength]));
            testCase.verifyEqual(numel(r(1).Labels), 4);
            lp = [-Inf; -Inf; -Inf; -20; 0; 1; 0.5]; lp = lp - log(sum(exp(lp(isfinite(lp)))));
            testCase.verifyEqual(r(1).LogProb, 4 * lp(6) + lp(4), 'AbsTol', 1e-12);   % forced Eos is paid for
            g = OceanNLP.Speech.Decoding.AttentionBeamDecoder(v, 'Method', "greedy", 'MaxLength', 3).search(step, 7);
            testCase.verifyEqual(g.Labels, [6 6]);
            testCase.verifyTrue(g.HitMaxLength);
        end

        function testTranscriptAndErrors(testCase)
            d = OceanNLP.Speech.Decoding.AttentionBeamDecoder(testCase.Vocab, 'BeamWidth', 4, 'NBest', 3, ...
                'MaxLength', 5, 'ModelId', "toy");
            tr = d.decode(testCase.stepFn(), [4 4]);
            testCase.verifySize(tr, [1 2]);
            testCase.verifyEqual(tr(1).Decoder, "attention-beam");
            testCase.verifyEqual(tr(1).ModelId, "toy");
            testCase.verifyEqual(tr(1).DecoderOptions.BeamWidth, 4);
            testCase.verifyEqual(tr(1).DecoderOptions.LengthPenalty, 1);
            testCase.verifyEqual(numel(tr(1).Candidates), 3);
            testCase.verifyEqual(tr(1).Text, testCase.Vocab.decode(tr(1).TokenIds));
            testCase.verifyEqual(tr(1).Score, tr(1).Candidates(1).Score);
            testCase.verifyEmpty(tr(1).Segments);
            bad = @(p, src) zeros(3, size(p, 1));
            testCase.verifyError(@() d.search(bad, 4), 'OceanNLP:Speech:Decoding:AttentionBeamDecoder:StepSize');
        end
    end

    methods
        function f = stepFn(testCase)
            T = testCase.Table; O = testCase.Offset;
            f = @(p, src) toyStep(T, O, p, src);
        end

        function [ids, logP] = manualGreedy(testCase, n, Lmax)
            v = testCase.Vocab; prefix = v.Bos; logP = 0; ids = zeros(1, 0);
            for u = 1:Lmax
                lp = testCase.logp(prefix, n);
                if u == Lmax, k = v.Eos; else, [~, k] = max(lp); end
                logP = logP + lp(k);
                if k == v.Eos, return; end
                ids(end + 1) = k; %#ok<AGROW>
                prefix(end + 1) = k; %#ok<AGROW>
            end
        end

        function lp = logp(testCase, prefix, n)
            z = toyStep(testCase.Table, testCase.Offset, prefix, n);
            lp = z - log(sum(exp(z)));
            v = testCase.Vocab;
            lp([v.Blank v.Pad v.Bos]) = -Inf;
        end

        function [seqs, scores, logPs] = bruteForce(testCase, n, Lmax, alpha)
            % every y = (y_1..y_L, Eos), L <= Lmax - 1, y_i in {Unk, units}
            v = testCase.Vocab; units = [v.Unk, 6:v.numClasses()];
            seqs = {}; scores = []; logPs = [];
            for L = 0:Lmax - 1
                grid = cell(1, L);
                [grid{:}] = ndgrid(units);
                combos = reshape(cat(L + 1, grid{:}), [], L);
                if L == 0, combos = zeros(1, 0); end
                for c = 1:size(combos, 1)
                    y = combos(c, :); prefix = v.Bos; lp = 0;
                    for i = 1:L + 1
                        l = testCase.logp(prefix, n);
                        if i <= L, tok = y(i); else, tok = v.Eos; end
                        lp = lp + l(tok);
                        prefix(end + 1) = tok; %#ok<AGROW>
                    end
                    seqs{end + 1} = y; %#ok<AGROW>
                    logPs(end + 1) = lp; %#ok<AGROW>
                    scores(end + 1) = lp / (L + 1) ^ alpha; %#ok<AGROW>
                end
            end
        end
    end
end

function [Z, st] = statefulToyStep(T, O, prefixes, src, st, parent)
% state: B x (u-1) token history (Bos first) of each row of the PREVIOUS call
hist = [st(parent, :), prefixes(:, end)];
if ~isequal(hist, prefixes)
    error('toy:StateMismatch', 'Reordered state does not reproduce the prefixes.');
end
st = hist;
Z = toyStep(T, O, hist, src);
end

function Z = toyStep(T, O, prefixes, src)
B = size(prefixes, 1); u = size(prefixes, 2);
Z = zeros(size(T, 1), B);
for b = 1:B
    Z(:, b) = T(:, prefixes(b, end), u) + O(:, src(b));
end
end
