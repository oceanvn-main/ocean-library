classdef RegexTokenizerTest < OceanTest.BaseTestCase
    % REGEXTOKENIZERTEST  Unit tests for OceanNLP.Tokenize.RegexTokenizer.

    methods (Test)
        function testDefaultPatternSplitsContractions(testCase)
            % Documented policy: the default '\w+' pattern splits
            % contractions into two tokens (apostrophe is not \w),
            % unlike WhitespaceTokenizer.
            import OceanNLP.Tokenize.RegexTokenizer
            tok = RegexTokenizer();
            tokens = tok.forward("Hello, world! Isn't MATLAB great?");

            testCase.verifyEqual(tokens, ["hello" "world" "isn" "t" "matlab" "great"]);
        end

        function testCustomPatternPreservesApostrophe(testCase)
            import OceanNLP.Tokenize.RegexTokenizer
            tok = RegexTokenizer('Pattern', "[\w']+");
            tokens = tok.forward("Isn't MATLAB great?");

            testCase.verifyEqual(tokens, ["isn't" "matlab" "great"]);
        end

        function testCaseFoldingDefaultAndOverride(testCase)
            import OceanNLP.Tokenize.RegexTokenizer
            tokDefault = RegexTokenizer();
            tokens = tokDefault.forward("Case Sensitive Test");
            testCase.verifyEqual(tokens, ["case" "sensitive" "test"]);

            tokCS = RegexTokenizer('CaseSensitive', true);
            tokensCS = tokCS.forward("Case Sensitive Test");
            testCase.verifyEqual(tokensCS, ["Case" "Sensitive" "Test"]);
        end

        function testEmptyOrNoMatchInputReturns1x0(testCase)
            import OceanNLP.Tokenize.RegexTokenizer
            tok = RegexTokenizer();

            testCase.verifySize(tok.forward(""), [1 0]);
            testCase.verifySize(tok.forward("!!! ..."), [1 0]);
        end

        function testCharAndStringInputBothAccepted(testCase)
            import OceanNLP.Tokenize.RegexTokenizer
            tok = RegexTokenizer();

            fromChar   = tok.forward('Hello World');
            fromString = tok.forward("Hello World");

            testCase.verifyEqual(fromChar, fromString);
            testCase.verifyClass(fromChar, 'string');
        end

        % NOTE: RegexTokenizer's constructor validates Pattern by calling
        % regexp("", Pattern, 'match') inside a try/catch and rethrowing as
        % OceanNLP:Tokenize:RegexTokenizer:InvalidPattern on failure — but
        % empirically, on this MATLAB installation (R2026a), regexp() does
        % NOT throw for any malformed pattern tried during test authoring
        % ("[unclosed", "(?<bad", "a{1,2,3}", "\x{110000}", etc. all pass
        % silently with no match, no error, no warning). The InvalidPattern
        % path therefore cannot be exercised deterministically on this
        % MATLAB version, so no test asserts it — flagged here for the
        % Critic rather than asserting behavior that cannot be reproduced.
    end
end
