classdef MemoryReader < handle
    %MEMORYREADER In-memory, read-only view of captured process steps.
    % Values within the capture budget are held in full. Omitted values
    % retain type, size, byte count, and lineage, but cannot be reloaded.
    properties (Access = private)
        Steps struct
        Artifacts struct
    end

    methods
        function obj = MemoryReader(steps, artifacts)
            arguments
                steps struct = struct('Id', {}, 'Name', {}, 'Stage', {}, ...
                    'InputIds', {}, 'OutputIds', {}, 'Summary', {}, ...
                    'Status', {}, 'Sequence', {})
                artifacts struct = struct('Id', {}, 'Kind', {}, 'Name', {}, ...
                    'Value', {}, 'Summary', {}, 'Status', {}, 'ByteCount', {})
            end
            obj.Steps = steps;
            obj.Artifacts = artifacts;
        end

        function steps = listSteps(obj, options)
            arguments
                obj
                options.Query (1,1) string = ""
                options.Offset (1,1) double {mustBeInteger, mustBePositive} = 1
                options.Limit (1,1) double {mustBePositive} = inf
            end
            steps = obj.Steps;
            if options.Query ~= "" && ~isempty(steps)
                names = [steps.Name];
                summaries = [steps.Summary];
                matches = contains(names, options.Query, 'IgnoreCase', true) | ...
                    contains(summaries, options.Query, 'IgnoreCase', true);
                steps = steps(matches);
            end
            first = min(options.Offset, numel(steps) + 1);
            last = min(numel(steps), first + options.Limit - 1);
            steps = steps(first:last);
        end

        function step = getStep(obj, id)
            arguments
                obj
                id (1,1) string
            end
            index = find([obj.Steps.Id] == id, 1);
            if isempty(index)
                error('OceanNLP:Inspect:UnknownStep', 'Unknown step ID: %s', id);
            end
            step = obj.Steps(index);
        end

        function artifact = getArtifact(obj, id)
            arguments
                obj
                id (1,1) string
            end
            index = find([obj.Artifacts.Id] == id, 1);
            if isempty(index)
                error('OceanNLP:Inspect:UnknownArtifact', 'Unknown artifact ID: %s', id);
            end
            artifact = obj.Artifacts(index);
        end

        function stages = listStages(obj)
            arguments
                obj
            end
            if isempty(obj.Steps)
                stages = strings(1,0);
            else
                stages = unique([obj.Steps.Stage], 'stable');
            end
        end

        function lineage = getLineage(obj, artifactId)
            arguments
                obj
                artifactId (1,1) string
            end
            obj.getArtifact(artifactId); % Validate the ID.
            producer = strings(1,0);
            consumers = strings(1,0);
            for i = 1:numel(obj.Steps)
                step = obj.Steps(i);
                if any(step.OutputIds == artifactId)
                    producer(end+1) = step.Id;
                end
                if any(step.InputIds == artifactId)
                    consumers(end+1) = step.Id;
                end
            end
            lineage = struct('ArtifactId', artifactId, ...
                'ProducerIds', producer, 'ConsumerIds', consumers);
        end
    end
end
