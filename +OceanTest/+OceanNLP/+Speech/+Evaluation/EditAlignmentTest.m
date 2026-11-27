classdef EditAlignmentTest < OceanTest.BaseTestCase
    % EDITALIGNMENTTEST  Levenshtein alignment: optimality, consistency, jiwer/rapidfuzz split.

    properties
        Fixture
    end

    methods (TestClassSetup)
        function loadFixture(testCase)
            f = fullfile(fileparts(mfilename('fullpath')), 'fixtures', 'jiwer_cases.json');
            testCase.Fixture = jsondecode(fileread(f));
        end
    end

    methods (Test)
        function testDistanceMatchesIndependentDP(testCase)
            % Independent reference: OceanDL.Recurrent.CTCDecoder.editDistance (plain DP).
            rng0 = RandStream('mt19937ar', 'Seed', 11);
            for trial = 1:400
                a = randi(rng0, 4, 1, randi(rng0, [0 10]));
                b = randi(rng0, 4, 1, randi(rng0, [0 10]));
                r = OceanNLP.Speech.Evaluation.EditAlignment.align(a, b);
                testCase.verifyEqual(r.Distance, OceanDL.Recurrent.CTCDecoder.editDistance(a, b));
                testCase.verifyEqual(r.Substitutions + r.Deletions + r.Insertions, r.Distance);
                testCase.verifyEqual(r.Hits + r.Substitutions + r.Deletions, numel(a));
                testCase.verifyEqual(r.Hits + r.Substitutions + r.Insertions, numel(b));
            end
        end

        function testOpsReplayReferenceIntoHypothesis(testCase)
            % Walking the ops over ref must reproduce hyp exactly.
            rng0 = RandStream('mt19937ar', 'Seed', 5);
            for trial = 1:200
                a = randi(rng0, 3, 1, randi(rng0, [0 9]));
                b = randi(rng0, 3, 1, randi(rng0, [0 9]));
                r = OceanNLP.Speech.Evaluation.EditAlignment.align(a, b);
                out = [];
                for k = 1:numel(r.Ops)
                    switch r.Ops(k)
                        case "C"
                            testCase.verifyEqual(a(r.RefIndex(k)), b(r.HypIndex(k)));
                            out(end+1) = a(r.RefIndex(k)); %#ok<AGROW>
                        case "S"
                            testCase.verifyNotEqual(a(r.RefIndex(k)), b(r.HypIndex(k)));
                            out(end+1) = b(r.HypIndex(k)); %#ok<AGROW>
                        case "I"
                            testCase.verifyEqual(r.RefIndex(k), 0);
                            out(end+1) = b(r.HypIndex(k)); %#ok<AGROW>
                        case "D"
                            testCase.verifyEqual(r.HypIndex(k), 0);
                    end
                end
                testCase.verifyEqual(reshape(out, 1, []), reshape(double(b), 1, []));
                testCase.verifyEqual(reshape(r.RefIndex(r.RefIndex > 0), 1, []), 1:numel(a));
                testCase.verifyEqual(reshape(r.HypIndex(r.HypIndex > 0), 1, []), 1:numel(b));
            end
        end

        function testSplitMatchesJiwerWordAndCharacter(testCase)
            % S/D/I/H per pair equal jiwer 4 (rapidfuzz opcodes), word and char level.
            C = testCase.Fixture.cases;
            for k = 1:numel(C)
                ref = string(C(k).ref); hyp = string(C(k).hyp);
                w = OceanNLP.Speech.Evaluation.EditAlignment.align( ...
                    OceanNLP.Speech.Evaluation.WordErrorRate.tokenize(ref), ...
                    OceanNLP.Speech.Evaluation.WordErrorRate.tokenize(hyp));
                c = OceanNLP.Speech.Evaluation.EditAlignment.align( ...
                    OceanNLP.Speech.Evaluation.CharacterErrorRate.tokenize(ref), ...
                    OceanNLP.Speech.Evaluation.CharacterErrorRate.tokenize(hyp));
                msg = sprintf('case %d "%s" vs "%s"', k, ref, hyp);
                testCase.verifyEqual([w.Substitutions w.Deletions w.Insertions w.Hits], ...
                    [C(k).word.S C(k).word.D C(k).word.I C(k).word.H], msg);
                testCase.verifyEqual([c.Substitutions c.Deletions c.Insertions c.Hits], ...
                    [C(k).char.S C(k).char.D C(k).char.I C(k).char.H], msg);
            end
        end

        function testStringAndNumericAndEmpty(testCase)
            A = @OceanNLP.Speech.Evaluation.EditAlignment.align;
            r = A(["one" "two"], strings(1, 0));
            testCase.verifyEqual(r.Ops, ["D" "D"]);
            r = A([], [3 4 5]);
            testCase.verifyEqual(r.Ops, ["I" "I" "I"]);
            testCase.verifyEqual(r.HypIndex, 1:3);
            r = A(strings(1, 0), []);
            testCase.verifyEmpty(r.Ops);
            testCase.verifyEqual(r.Distance, 0);
            testCase.verifyError(@() A(["a" "b"], [1 2]), 'OceanNLP:Speech:Evaluation:EditAlignment:TokenType');
            testCase.verifyError(@() A([1 NaN], 1), 'OceanNLP:Speech:Evaluation:EditAlignment:Nonfinite');
        end

        function testDeterministicTieBreakOnSwap(testCase)
            % "a b" vs "b a": two optimal paths of cost 2 (S,S or D...I); the
            % documented rule yields the rapidfuzz/jiwer choice for the fixture case.
            r1 = OceanNLP.Speech.Evaluation.EditAlignment.align(["aaaa" "bbbb"], ["bbbb" "aaaa"]);
            r2 = OceanNLP.Speech.Evaluation.EditAlignment.align(["aaaa" "bbbb"], ["bbbb" "aaaa"]);
            testCase.verifyEqual(r1, r2);
            testCase.verifyEqual(r1.Distance, 2);
        end
    end
end
