classdef CountingLearner < OceanAgents.Core.ILearner
    % COUNTINGLEARNER  Test fixture learner: stores every Transition it is given and, at its declared boundary,
    % reports one "parameter update" per stored-but-unapplied transition. Logs "consume:<id>" / "update:<b>".
    %   'Gamma', g          reported in describe() (use [] to report no Gamma)
    %   'Boundary', b       "step" | "episode"
    %   'ThrowOnConsume', k the k-th consume throws Fixture:Injected
    properties
        Log = []
        Payload = {}
    end
    properties (SetAccess = private)
        Gamma = 1
        Boundary string = "step"
        ThrowOnConsume double = 0
        Consumed cell = {}
        Pending double = 0
        UpdateCalls string = strings(1, 0)
        Applied double = 0
    end
    methods
        function obj = CountingLearner(options)
            arguments
                options.Gamma = 1
                options.Boundary (1,1) string = "step"
                options.ThrowOnConsume (1,1) double = 0
                options.Log = []
            end
            obj.Gamma = options.Gamma;
            obj.Boundary = options.Boundary;
            obj.ThrowOnConsume = options.ThrowOnConsume;
            obj.Log = options.Log;
        end

        function d = describe(obj)
            d = struct('Name', "CountingLearner", 'Version', "1", 'Gamma', obj.Gamma, 'Boundary', obj.Boundary);
        end

        function consume(obj, transition)
            if ~isempty(obj.Log), obj.Log.add("consume:" + transition.AgentID); end
            if numel(obj.Consumed) + 1 == obj.ThrowOnConsume
                error('Fixture:Injected', 'Injected consume fault on call %d.', obj.ThrowOnConsume);
            end
            obj.Consumed{end + 1} = transition;
            obj.Pending = obj.Pending + 1;
        end

        function n = update(obj, boundary)
            obj.UpdateCalls(end + 1) = string(boundary);
            if ~isempty(obj.Log), obj.Log.add("update:" + boundary); end
            n = 0;
            if string(boundary) == obj.Boundary
                n = obj.Pending;
                obj.Applied = obj.Applied + n;
                obj.Pending = 0;
            end
        end

        function s = exportState(obj)
            s = struct('Applied', obj.Applied);
        end

        function importState(obj, s)
            obj.Applied = s.Applied;
        end
    end
end
