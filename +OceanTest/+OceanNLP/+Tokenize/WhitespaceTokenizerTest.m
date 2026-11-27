classdef WhitespaceTokenizerTest < OceanTest.BaseTestCase
    % WHITESPACETOKENIZERTEST  Unit tests for OceanNLP.Tokenize.WhitespaceTokenizer.

    methods (Test)
        function testDefaultCaseFolding(testCase)
            import OceanNLP.Tokenize.WhitespaceTokenizer
            tok = WhitespaceTokenizer();
            tokens = tok.forward("Hello, world! Isn't MATLAB great?");

            testCase.verifyEqual(tokens, ["hello" "world" "isn't" "matlab" "great"]);
        end

        function testCaseSensitiveOverridePreservesCase(testCase)
            import OceanNLP.Tokenize.WhitespaceTokenizer
            tok = WhitespaceTokenizer('CaseSensitive', true);
            tokens = tok.forward("Hello World");

            testCase.verifyEqual(tokens, ["Hello" "World"]);
        end

        function testEdgePunctuationStripped(testCase)
            import OceanNLP.Tokenize.WhitespaceTokenizer
            tok = WhitespaceTokenizer();

            testCase.verifyEqual(tok.forward("cat,"), "cat");
            testCase.verifyEqual(tok.forward("(great)"), "great");
            testCase.verifyEqual(tok.forward("world!"), "world");
        end

        function testInternalPunctuationPreserved(testCase)
            import OceanNLP.Tokenize.WhitespaceTokenizer
            tok = WhitespaceTokenizer();

            testCase.verifyEqual(tok.forward("don't"), "don't");
            testCase.verifyEqual(tok.forward("well-known"), "well-known");
            testCase.verifyEqual(tok.forward("3.14"), "3.14");
        end

        function testPurePunctuationChunkIsDropped(testCase)
            import OceanNLP.Tokenize.WhitespaceTokenizer
            tok = WhitespaceTokenizer();
            tokens = tok.forward("well-known 3.14 (great) cat, --");

            testCase.verifyEqual(tokens, ["well-known" "3.14" "great" "cat"]);
        end

        function testEmptyOrWhitespaceOnlyInputReturns1x0(testCase)
            import OceanNLP.Tokenize.WhitespaceTokenizer
            tok = WhitespaceTokenizer();

            emptyTok = tok.forward("   ");
            testCase.verifySize(emptyTok, [1 0]);
            testCase.verifyClass(emptyTok, 'string');

            punctOnlyTok = tok.forward("!!! ... ???");
            testCase.verifySize(punctOnlyTok, [1 0]);
        end

        function testCharAndStringInputBothAccepted(testCase)
            import OceanNLP.Tokenize.WhitespaceTokenizer
            tok = WhitespaceTokenizer();

            fromChar   = tok.forward('Hello World');
            fromString = tok.forward("Hello World");

            testCase.verifyEqual(fromChar, fromString);
            testCase.verifyClass(fromChar, 'string');
        end
    end
end
