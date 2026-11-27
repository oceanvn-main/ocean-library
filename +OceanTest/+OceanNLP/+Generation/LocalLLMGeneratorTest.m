classdef LocalLLMGeneratorTest < OceanTest.BaseTestCase
    methods (Test)
        function testConstructorDefaults(testCase)
            gen = OceanNLP.Generation.LocalLLMGenerator();
            testCase.verifyEqual(gen.Endpoint, "http://localhost:1234/v1/chat/completions");
            testCase.verifyEqual(gen.Model, "gemma-4-12b");
            testCase.verifyEqual(gen.Temperature, 0);
        end

        function testCustomOptions(testCase)
            gen = OceanNLP.Generation.LocalLLMGenerator(...
                'Endpoint', "http://localhost:11434/v1/chat/completions", ...
                'Model', "gemma-4-12b", ...
                'Temperature', 0.5);
            testCase.verifyEqual(gen.Endpoint, "http://localhost:11434/v1/chat/completions");
            testCase.verifyEqual(gen.Model, "gemma-4-12b");
            testCase.verifyEqual(gen.Temperature, 0.5);
        end
    end
end
