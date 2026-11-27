classdef PipelineAdapterTest < OceanTest.BaseTestCase
    methods (Test)
        function testRealPipelineOutputAndLineage(testCase)
            pipeline = OceanNLP.Pipeline.ComposablePipeline();
            pipeline.add(OceanNLP.Pipeline.Stage.fromFunction("trim", @(value) strtrim(value)));
            pipeline.add(OceanNLP.Pipeline.Stage.fromFunction("upper", @(value) upper(value)));
            pipeline.add(OceanNLP.Pipeline.Stage.fromFunction("unused", @(value) reverse(value)));
            pipeline.enable("unused", false);

            input = "  Ocean NLP  ";
            expected = pipeline.run(input);
            [actual, reader] = OceanNLP.Inspect.PipelineAdapter.run(pipeline, input);
            steps = reader.listSteps();

            testCase.verifyEqual(actual, expected);
            testCase.verifyEqual([steps.Name], ["trim", "upper"]);
            testCase.verifyEqual(steps(1).OutputIds, steps(2).InputIds);
            testCase.verifyEqual(reader.getArtifact(steps(1).InputIds).Value, input);
            testCase.verifyEqual(reader.getArtifact(steps(2).OutputIds).Value, expected);
            lineage = reader.getLineage(steps(1).OutputIds);
            testCase.verifyEqual(lineage.ProducerIds, steps(1).Id);
            testCase.verifyEqual(lineage.ConsumerIds, steps(2).Id);
        end

        function testBudgetOmitsLargeValuesWithoutChangingOutput(testCase)
            pipeline = OceanNLP.Pipeline.ComposablePipeline();
            pipeline.add(OceanNLP.Pipeline.Stage.fromFunction("double", @(value) 2 * value));
            input = ones(100, 100);
            [output, reader] = OceanNLP.Inspect.PipelineAdapter.run( ...
                pipeline, input, 'MaxArtifactBytes', 1024);
            testCase.verifyEqual(output, 2 * input);
            step = reader.getStep("step-1");
            artifact = reader.getArtifact(step.OutputIds);
            testCase.verifyEqual(artifact.Status, "omitted");
            testCase.verifyEmpty(artifact.Value);
            testCase.verifyGreaterThan(artifact.ByteCount, 1024);
        end

        function testFailureIsCapturedAndRethrown(testCase)
            pipeline = OceanNLP.Pipeline.ComposablePipeline();
            pipeline.add(OceanNLP.Pipeline.Stage.fromFunction("fail", @(value) throwingStage(value)));
            [~, reader, failure] = OceanNLP.Inspect.PipelineAdapter.capture(pipeline, "input");
            step = reader.getStep("step-1");
            testCase.verifyEqual(step.Status, "failed");
            testCase.verifyEmpty(step.OutputIds);
            testCase.verifyEqual(reader.getArtifact(step.InputIds).Value, "input");
            testCase.verifyEqual(failure.identifier, 'OceanNLP:Inspect:ExpectedFailure');
            testCase.verifyError(@() OceanNLP.Inspect.PipelineAdapter.run(pipeline, "input"), ...
                'OceanNLP:Inspect:ExpectedFailure');
        end
    end
end

function out = throwingStage(value)
    % An anonymous @(v) error(...) has no output; R2026a then raises
    % MATLAB:maxlhs instead of the intended error.
    out = value;
    error('OceanNLP:Inspect:ExpectedFailure', 'Cannot process %s', value);
end
