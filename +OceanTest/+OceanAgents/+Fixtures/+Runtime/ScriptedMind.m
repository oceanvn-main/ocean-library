classdef ScriptedMind < OceanAgents.Core.IDecisionProcess
    % SCRIPTEDMIND  Test fixture mind: plays a prescribed action sequence (cycled), consumes no randomness, counts
    % every call, and can inject faults:
    %   'ThrowOnDecide', k    the k-th decide throws Fixture:Injected
    %   'IllegalOnDecide', k  the k-th decide returns a label that is not legal (host must reject it)
    %   'ThrowOnObserve', k   the k-th observe throws Fixture:Injected
    % Payload is a free slot for isolation tests (put a handle in it to create sharing on purpose).
    properties
        Log = []
        Payload = {}
    end
    properties (SetAccess = private)
        Actions double = 1
        ThrowOnDecide double = 0
        IllegalOnDecide double = 0
        ThrowOnObserve double = 0
        DecideCalls double = 0
        ObserveCalls double = 0
        BeginCalls double = 0
        EndCalls double = 0
        Feedbacks cell = {}
    end
    methods
        function obj = ScriptedMind(actions, options)
            arguments
                actions (1,:) double = 1
                options.ThrowOnDecide (1,1) double = 0
                options.IllegalOnDecide (1,1) double = 0
                options.ThrowOnObserve (1,1) double = 0
                options.Log = []
            end
            obj.Actions = actions;
            obj.ThrowOnDecide = options.ThrowOnDecide;
            obj.IllegalOnDecide = options.IllegalOnDecide;
            obj.ThrowOnObserve = options.ThrowOnObserve;
            obj.Log = options.Log;
        end

        function d = describe(~)
            d = struct('Profile', "scripted", 'Components', {{}});
        end

        function beginEpisode(obj, ~)
            obj.BeginCalls = obj.BeginCalls + 1;
        end

        function dec = decide(obj, view, ~)
            obj.DecideCalls = obj.DecideCalls + 1;
            if ~isempty(obj.Log), obj.Log.add("decide:" + view.AgentID); end
            if obj.DecideCalls == obj.ThrowOnDecide
                error('Fixture:Injected', 'Injected decide fault on call %d.', obj.DecideCalls);
            end
            action = obj.Actions(mod(obj.DecideCalls - 1, numel(obj.Actions)) + 1);
            if obj.DecideCalls == obj.IllegalOnDecide
                action = max(view.ActionLabels) + 1;          % not an action label of the view
            end
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', action);
        end

        function observe(obj, feedback)
            obj.ObserveCalls = obj.ObserveCalls + 1;
            if ~isempty(obj.Log), obj.Log.add("observe:" + feedback.AgentID); end
            if obj.ObserveCalls == obj.ThrowOnObserve
                error('Fixture:Injected', 'Injected observe fault on call %d.', obj.ObserveCalls);
            end
            obj.Feedbacks{end + 1} = feedback;
        end

        function endEpisode(obj)
            obj.EndCalls = obj.EndCalls + 1;
        end

        function s = exportState(obj)
            s = struct('DecideCalls', obj.DecideCalls);
        end

        function importState(obj, s)
            obj.DecideCalls = s.DecideCalls;
        end
    end
end
