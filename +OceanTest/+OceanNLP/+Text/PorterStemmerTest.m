classdef PorterStemmerTest < OceanTest.BaseTestCase
    % PORTERSTEMMERTEST  Expected stems come from NLTK's
    % PorterStemmer(mode="ORIGINAL_ALGORITHM") (external ground truth, see
    % porter_reference.py) — never from calling the code under test. The full
    % 7,562-word real-vocabulary comparison is compare_text_in_matlab.m.

    methods (Test)
        function testPaperStepExamples(testCase)
            st = OceanNLP.Text.Stem.PorterStemmer();
            words = ["caresses" "ponies" "ties" "feed" "agreed" "plastered" "bled" "motoring" "sing" ...
                     "conflated" "troubled" "sized" "hopping" "tanned" "falling" "hissing" "fizzed" ...
                     "failing" "filing" "happy" "sky"];
            want  = ["caress" "poni" "ti" "feed" "agre" "plaster" "bled" "motor" "sing" ...
                     "conflat" "troubl" "size" "hop" "tan" "fall" "hiss" "fizz" ...
                     "fail" "file" "happi" "sky"];
            for i = 1:numel(words)
                testCase.verifyEqual(st.stem(words(i)), want(i), "stem(" + words(i) + ")");
            end
        end

        function testMultiStepWords(testCase)
            st = OceanNLP.Text.Stem.PorterStemmer();
            words = ["relational" "conditional" "rational" "digitizer" "vietnamization" "predication" "operator" ...
                     "feudalism" "decisiveness" "hopefulness" "triplicate" "formative" "electrical" "generalizations" "oscillators"];
            want  = ["relat" "condit" "ration" "digit" "vietnam" "predic" "oper" ...
                     "feudal" "decis" "hope" "triplic" "form" "electr" "gener" "oscil"];
            for i = 1:numel(words)
                testCase.verifyEqual(st.stem(words(i)), want(i), "stem(" + words(i) + ")");
            end
        end

        function testOriginalAlgorithmSpecificsNotLaterVariants(testCase)
            st = OceanNLP.Text.Stem.PorterStemmer();
            % step 5b / 5a and the 'abli'-vs-'bli', no short-word shortcut behaviour of the ORIGINAL algorithm
            testCase.verifyEqual(st.stem("controll"), "control");
            testCase.verifyEqual(st.stem("roll"), "roll");
            testCase.verifyEqual(st.stem("cease"), "ceas");
            testCase.verifyEqual(st.stem("probate"), "probat");
            testCase.verifyEqual(st.stem("as"), "a");          % short words ARE processed (NLTK ORIGINAL)
            testCase.verifyEqual(st.stem("is"), "i");
            testCase.verifyEqual(st.stem("ss"), "ss");
            testCase.verifyEqual(st.stem("y"), "y");
        end

        function testInputIsLowercased(testCase)
            testCase.verifyEqual(OceanNLP.Text.Stem.PorterStemmer().stem("RUNNING"), "run");
        end

        function testIdempotenceOnSampleIsNotClaimed(testCase)
            % Porter is NOT idempotent in general; guard against someone "fixing" that.
            st = OceanNLP.Text.Stem.PorterStemmer();
            testCase.verifyEqual(st.stem("generalizations"), "gener");
            testCase.verifyNotEqual(st.stem(st.stem("agreed")), st.stem("agreed"));   % agre -> agr
        end

        function testStemFilterAppliesStrategyToEveryToken(testCase)
            f = OceanNLP.Text.Filter.StemFilter('Stemmer', OceanNLP.Text.Stem.PorterStemmer());
            testCase.verifyEqual(f.forward(["ponies" "running" "cats"]), ["poni" "run" "cat"]);
            testCase.verifyEqual(f.getPipelineSpec().Kind, "VIRTUAL");
            testCase.verifyTrue(contains(f.getMetadata().ComposedPartReference.StemmerReference, "Porter"));
            % default stemmer is the identity "basic version"
            testCase.verifyEqual(OceanNLP.Text.Filter.StemFilter().forward(["ponies" "running"]), ["ponies" "running"]);
        end
    end
end
