classdef TextStrategyContractTest < OceanTest.BaseTestCase
    % TEXTSTRATEGYCONTRACTTEST  Interface contract run over EVERY strategy the
    % factories can discover, so a newly added strategy file is checked
    % automatically without anyone editing this test (extension rule).
    % Kinds: charfilter, filter, stemmer, pretokenizer, tokenizer.

    properties (TestParameter)
        kind = {"charfilter", "filter", "stemmer", "pretokenizer", "subword"}
    end

    methods (Test)
        function testEveryDiscoveredStrategyHonoursThreePillars(testCase, kind)
            names = OceanNLP.Text.Factory.validNames(kind);
            testCase.verifyNotEmpty(names, "no strategies discovered for kind " + kind);
            for i = 1:numel(names)
                s = OceanNLP.Text.Factory.create(kind, names{i});
                label = kind + "/" + names{i};
                testCase.verifyTrue(isa(s, 'OceanNLP.Text.TextStrategy'), label);
                testCase.verifyTrue(isstring(s.name()) && isscalar(s.name()) && strlength(s.name()) > 0, label);
                m = s.getMetadata();
                for f = ["SymbolicLatex" "Symbols" "Assumptions" "IsSymbolic" "Reference"]
                    testCase.verifyTrue(isfield(m, f), label + " metadata lacks " + f);
                end
                testCase.verifyFalse(m.IsSymbolic, label + ": text strategies are honestly non-symbolic (R7)");
                testCase.verifyGreaterThan(strlength(m.Reference), 0, label);
                p = s.getPipelineSpec();
                for f = ["Inputs" "Outputs" "Kind" "DiagramSymbol" "ComposesFrom"]
                    testCase.verifyTrue(isfield(p, f), label + " spec lacks " + f);
                end
                testCase.verifyTrue(any(p.Kind == ["NONVIRTUAL" "VIRTUAL"]), label);
                for c = p.ComposesFrom
                    testCase.verifyTrue(exist(c{1}, 'class') == 8, label + ": ComposesFrom names a class that does not exist");
                end
            end
        end

        function testTokenizersAreDiscoveredAndSatisfyInterface(testCase)
            names = OceanNLP.Text.Factory.validNames("tokenizer");
            testCase.verifyTrue(all(ismember({'regex', 'whitespace'}, names)));
            for i = 1:numel(names)
                t = OceanNLP.Text.Factory.create("tokenizer", names{i});
                testCase.verifyTrue(isa(t, 'OceanNLP.Tokenize.ITokenizer'));
                out = t.forward("Hello, world 42");
                testCase.verifyClass(out, 'string');
                testCase.verifyTrue(isrow(out) || isempty(out));
            end
        end

        function testFiltersAndStemmersHandleEmptyInputs(testCase)
            for n = OceanNLP.Text.Factory.validNames("filter")
                f = OceanNLP.Text.Factory.create("filter", n{1});
                testCase.verifyEmpty(f.forward(strings(1, 0)), "filter " + n{1});
            end
            for n = OceanNLP.Text.Factory.validNames("stemmer")
                s = OceanNLP.Text.Factory.create("stemmer", n{1});
                testCase.verifyEqual(s.stem(""), "", "stemmer " + n{1});
            end
            for n = OceanNLP.Text.Factory.validNames("subword")
                a = OceanNLP.Text.Factory.create("subword", n{1});
                testCase.verifyEqual(a.detokenize(strings(1, 0)), "", "subword " + n{1});
                testCase.verifyClass(a.unkToken(), 'string');
            end
            for n = OceanNLP.Text.Factory.validNames("charfilter")
                c = OceanNLP.Text.Factory.create("charfilter", n{1});
                testCase.verifyEqual(c.forward(""), "", "charfilter " + n{1});
            end
        end
    end
end
