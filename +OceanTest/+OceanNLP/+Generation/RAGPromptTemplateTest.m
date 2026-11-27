classdef RAGPromptTemplateTest < OceanTest.BaseTestCase
    methods (Test)
        function testNamedSlots(testCase)
            t=OceanNLP.Generation.RAGPromptTemplate("{instructions}|{context}|{question}");
            testCase.verifyEqual(t.render("ctx","q","inst"),"inst|ctx|q");
        end
        function testPromptBuilderBudget(testCase)
            b = OceanNLP.Generation.PromptBuilder('MaxChunks', 2, 'MaxCharacters', 5);
            p = b.build(["abcdef", "ghij"], "q");
            expectedMaxLen = 5 + strlength(replace(b.Template, ["{context}", "{question}"], ["", "q"]));
            testCase.verifyLessThanOrEqual(strlength(p), expectedMaxLen);
        end
    end
end
