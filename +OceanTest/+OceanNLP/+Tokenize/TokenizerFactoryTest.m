classdef TokenizerFactoryTest < OceanTest.BaseTestCase
    % TOKENIZERFACTORYTEST  Unit tests for OceanNLP.Tokenize.TokenizerFactory
    % and the OceanNLP.Tokenize.ITokenizer abstract contract.

    methods (Test)
        function testCreateWhitespace(testCase)
            import OceanNLP.Tokenize.TokenizerFactory
            tok = TokenizerFactory.create('whitespace');

            testCase.verifyClass(tok, 'OceanNLP.Tokenize.WhitespaceTokenizer');
            testCase.verifyEqual(tok.name(), "whitespace");
        end

        function testCreateRegexWithPatternPassthrough(testCase)
            import OceanNLP.Tokenize.TokenizerFactory
            tok = TokenizerFactory.create('regex', 'Pattern', "[\w']+");

            testCase.verifyClass(tok, 'OceanNLP.Tokenize.RegexTokenizer');
            testCase.verifyEqual(tok.Pattern, "[\w']+");
        end

        function testUnknownNameErrors(testCase)
            import OceanNLP.Tokenize.TokenizerFactory
            testCase.verifyError(@() TokenizerFactory.create('bogus'), ...
                'OceanNLP:Tokenize:TokenizerFactory:UnknownTokenizer');
        end

        function testValidNames(testCase)
            import OceanNLP.Tokenize.TokenizerFactory
            names = TokenizerFactory.validNames();

            testCase.verifyClass(names, 'cell');
            testCase.verifyTrue(any(strcmp(names, 'whitespace')));
            testCase.verifyTrue(any(strcmp(names, 'regex')));
        end

        function testAbstractInterfaceCannotBeInstantiated(testCase)
            testCase.verifyError(@() OceanNLP.Tokenize.ITokenizer(), ...
                'MATLAB:class:abstract');
        end
    end
end
