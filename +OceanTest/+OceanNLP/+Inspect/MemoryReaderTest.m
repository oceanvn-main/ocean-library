classdef MemoryReaderTest < OceanTest.BaseTestCase
    methods (Test)
        function testLookupAndMissingIds(testCase)
            pipeline = OceanNLP.Pipeline.ComposablePipeline();
            pipeline.add(OceanNLP.Pipeline.Stage.fromFunction("trim", @(value) strtrim(value)));
            [actual, reader] = OceanNLP.Inspect.PipelineAdapter.run(pipeline, "  real text  ");
            testCase.verifyEqual(actual, "real text");
            steps = reader.listSteps();
            testCase.verifyNumElements(steps, 1);
            testCase.verifyEqual(reader.getStep(steps(1).Id), steps(1));
            testCase.verifyEqual(reader.getArtifact(steps(1).OutputIds).Value, actual);
            testCase.verifyEqual(reader.listStages(), "trim");
            lineage = reader.getLineage(steps(1).InputIds);
            testCase.verifyEmpty(lineage.ProducerIds);
            testCase.verifyEqual(lineage.ConsumerIds, steps(1).Id);
            testCase.verifyEqual(reader.listSteps('Query', "TRIM", 'Limit', 1), steps);
            testCase.verifyEmpty(reader.listSteps('Offset', 2));
            testCase.verifyError(@() reader.getStep("missing"), ...
                'OceanNLP:Inspect:UnknownStep');
            testCase.verifyError(@() reader.getArtifact("missing"), ...
                'OceanNLP:Inspect:UnknownArtifact');
        end
    end
end
