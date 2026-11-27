classdef PipelineAdapter
    %PIPELINEADAPTER Execute a real ComposablePipeline with step capture.
    % The adapter validates the pipeline exactly as ComposablePipeline.run
    % does, then invokes each enabled stage through Stage.apply, the same
    % call path run uses. Stage errors are captured unwrapped.
    methods (Static)
        function [output, reader] = run(pipeline, input, options)
            arguments
                pipeline (1,1) OceanNLP.Pipeline.ComposablePipeline
                input
                options.MaxArtifactBytes (1,1) double {mustBeNonnegative} = 8e6
                options.MaxTotalBytes (1,1) double {mustBeNonnegative} = 64e6
            end
            [output, reader, failure] = OceanNLP.Inspect.PipelineAdapter.capture( ...
                pipeline, input, 'MaxArtifactBytes', options.MaxArtifactBytes, ...
                'MaxTotalBytes', options.MaxTotalBytes);
            if ~isempty(failure)
                rethrow(failure);
            end
        end

        function [output, reader, failure] = capture(pipeline, input, options)
            %CAPTURE Return a failed step and the original error on failure.
            arguments
                pipeline (1,1) OceanNLP.Pipeline.ComposablePipeline
                input
                options.MaxArtifactBytes (1,1) double {mustBeNonnegative} = 8e6
                options.MaxTotalBytes (1,1) double {mustBeNonnegative} = 64e6
            end

            pipeline.assertValid(input);
            stages = pipeline.Stages;
            enabled = [stages.Enabled];
            count = nnz(enabled);
            steps = repmat(struct('Id', "", 'Name', "", 'Stage', "", ...
                'InputIds', strings(1,0), 'OutputIds', strings(1,0), ...
                'Summary', "", 'Status', "", 'Sequence', 0), 1, count);
            artifacts = repmat(struct('Id', "", 'Kind', "", 'Name', "", ...
                'Value', [], 'Summary', "", 'Status', "", 'ByteCount', 0), ...
                1, count + 1);

            state = input;
            bytesUsed = 0;
            [artifacts(1), bytesUsed] = OceanNLP.Inspect.PipelineAdapter.artifact_( ...
                1, "Input", state, options.MaxArtifactBytes, ...
                options.MaxTotalBytes, bytesUsed);
            previousId = artifacts(1).Id;
            sequence = 0;
            failure = [];
            for i = 1:numel(stages)
                stage = stages(i);
                if ~stage.Enabled
                    continue
                end
                sequence = sequence + 1;
                try
                    nextState = stage.apply(state);
                catch ME
                    failure = ME;
                    steps(sequence) = struct('Id', "step-" + sequence, ...
                        'Name', string(stage.Name), 'Stage', string(stage.Name), ...
                        'InputIds', previousId, 'OutputIds', strings(1,0), ...
                        'Summary', string(ME.message), 'Status', "failed", ...
                        'Sequence', sequence);
                    steps = steps(1:sequence);
                    artifacts = artifacts(1:sequence);
                    output = state;
                    reader = OceanNLP.Inspect.MemoryReader(steps, artifacts);
                    return
                end
                state = nextState;
                [artifacts(sequence + 1), bytesUsed] = ...
                    OceanNLP.Inspect.PipelineAdapter.artifact_(sequence + 1, ...
                    string(stage.Name) + " output", state, ...
                    options.MaxArtifactBytes, options.MaxTotalBytes, bytesUsed);
                currentId = artifacts(sequence + 1).Id;
                steps(sequence) = struct('Id', "step-" + sequence, ...
                    'Name', string(stage.Name), 'Stage', string(stage.Name), ...
                    'InputIds', previousId, 'OutputIds', currentId, ...
                    'Summary', "Produced " + artifacts(sequence + 1).Summary, ...
                    'Status', "completed", 'Sequence', sequence);
                previousId = currentId;
            end
            output = state;
            reader = OceanNLP.Inspect.MemoryReader(steps, artifacts);
        end
    end

    methods (Static, Access = private)
        function [artifact, bytesUsed] = artifact_(index, name, value, maxArtifactBytes, maxTotalBytes, bytesUsed)
            kind = string(class(value));
            dimensions = strjoin(string(size(value)), "×");
            summary = kind + " · " + dimensions;
            information = whos('value');
            byteCount = information.bytes;
            if byteCount > maxArtifactBytes || byteCount > maxTotalBytes - bytesUsed
                storedValue = [];
                status = "omitted";
                summary = summary + " · value omitted by memory budget";
            else
                storedValue = value;
                status = "recorded";
                bytesUsed = bytesUsed + byteCount;
            end
            artifact = struct('Id', "artifact-" + index, ...
                'Kind', kind, 'Name', name, 'Value', storedValue, ...
                'Summary', summary, 'Status', status, 'ByteCount', byteCount);
        end
    end
end
