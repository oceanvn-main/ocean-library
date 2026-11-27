classdef StrategyResolverTest < OceanTest.BaseTestCase
    % STRATEGYRESOLVERTEST  Convention-based discovery, explicit registration
    % and error behaviour. Demonstrates the extension rule: a strategy in
    % ANOTHER package becomes available with one register() call, no edits.

    methods (TestMethodSetup)
        function clean(testCase)
            OceanNLP.Text.StrategyResolver.unregisterAll();
            testCase.addTeardown(@() OceanNLP.Text.StrategyResolver.unregisterAll());
        end
    end

    methods (Test)
        function testResolvesByConventionCaseAndSeparatorInsensitive(testCase)
            r = @(n) func2str(OceanNLP.Text.StrategyResolver.resolve("OceanNLP.Text.Filter", "Filter", n));
            testCase.verifyEqual(r("stopword"), 'OceanNLP.Text.Filter.StopWordFilter');
            testCase.verifyEqual(r("Stop_Word"), 'OceanNLP.Text.Filter.StopWordFilter');
            testCase.verifyEqual(r("STOP-WORD"), 'OceanNLP.Text.Filter.StopWordFilter');
        end

        function testNamesExcludeAbstractInterfacesAndAreSorted(testCase)
            n = OceanNLP.Text.StrategyResolver.names("OceanNLP.Text.Stem", "Stemmer");
            testCase.verifyEqual(n, {'none', 'porter'});
            t = OceanNLP.Text.StrategyResolver.names("OceanNLP.Tokenize", "Tokenizer");
            testCase.verifyFalse(any(strcmp(t, 'i')), 'abstract ITokenizer must not appear');
        end

        function testUnknownNameErrorsWithValidList(testCase)
            testCase.verifyError(@() OceanNLP.Text.Factory.create("stemmer", "snowball9"), ...
                'OceanNLP:Text:StrategyResolver:Unknown');
            testCase.verifyError(@() OceanNLP.Text.Factory.create("nokind", "x"), 'OceanNLP:Text:Factory:UnknownKind');
        end

        function testRegisterMakesForeignClassAvailableWithoutEditingSource(testCase)
            OceanNLP.Text.StrategyResolver.register("OceanNLP.Text.Stem", "myplugin", @OceanNLP.Text.Stem.NoneStemmer);
            testCase.verifyTrue(any(strcmp(OceanNLP.Text.Factory.validNames("stemmer"), 'myplugin')));
            s = OceanNLP.Text.Factory.create("stemmer", "MyPlugin");
            testCase.verifyClass(s, 'OceanNLP.Text.Stem.NoneStemmer');
        end

        function testTokenizerFactoryKeepsLegacyBehaviour(testCase)
            import OceanNLP.Tokenize.TokenizerFactory
            testCase.verifyClass(TokenizerFactory.create('whitespace'), 'OceanNLP.Tokenize.WhitespaceTokenizer');
            testCase.verifyClass(TokenizerFactory.create('REGEX', 'Pattern', "[a-z]+"), 'OceanNLP.Tokenize.RegexTokenizer');
            testCase.verifyError(@() TokenizerFactory.create('bogus'), 'OceanNLP:Tokenize:TokenizerFactory:UnknownTokenizer');
        end
    end
end
