classdef SlidingWindowChunkerTest < OceanTest.BaseTestCase
    methods (Test)
        function testOverlapAndOrder(testCase)
            c=OceanNLP.Chunking.SlidingWindowChunker('WindowSize',3,'Overlap',1).chunk("a b c d e");
            testCase.verifyEqual(c,["a b c","c d e"]);
        end
    end
end
