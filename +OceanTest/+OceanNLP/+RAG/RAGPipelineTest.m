classdef RAGPipelineTest < OceanTest.BaseTestCase
    methods (Test)
        function testIndexQueryAndEditableWorkflow(testCase)
            ranker=OceanTest.RAGTestRanker(); generator=OceanNLP.Generation.FunctionGenerator(@(p) "generated:"+extractBetween(p,"Question: ","\nAnswer:"));
            pipe=OceanNLP.RAG.RAGPipeline('Chunker',OceanNLP.Chunking.FixedTokenChunker('ChunkSize',3),'Ranker',ranker,'Generator',generator,'TopK',1);
            pipe.index("alpha beta gamma delta"); result=pipe.query("alpha");
            testCase.verifyClass(result,'OceanNLP.RAG.RAGResult'); testCase.verifyEqual(result.ChunkIds,1); testCase.verifyEqual(result.Answer,"generated:alpha");
            testCase.verifyTrue(pipe.getMetadata().Indexed);
        end
    end
end
