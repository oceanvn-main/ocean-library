classdef FixedProbabilityMind < OceanAgents.Core.IDecisionProcess
    % FIXEDPROBABILITYMIND  Test fixture mind: observation-independent mixed strategy p over the view's columns.
    % One draw u = rand(stream) per decision from the stream the host passes in; the action is the first column
    % whose cumulative probability reaches u (inverse CDF). Mass on illegal columns is removed and p renormalised.
    % Draws records every u, so tests can compare random-stream usage across runs. Payload: free slot for
    % isolation tests.
    properties
        Log = []
        Payload = {}
    end
    properties (SetAccess = private)
        P double = [0.5 0.5]
        Draws double = zeros(1, 0)
    end
    methods
        function obj = FixedProbabilityMind(p, options)
            arguments
                p (1,:) double {mustBeNonnegative}
                options.Log = []
            end
            obj.P = p / sum(p);
            obj.Log = options.Log;
        end

        function d = describe(~)
            d = struct('Profile', "fixed-probability", 'Components', {{}});
        end

        function dec = decide(obj, view, stream)
            if ~isempty(obj.Log), obj.Log.add("decide:" + view.AgentID); end
            p = obj.P .* view.ActionMask;
            p = p / sum(p);
            u = rand(stream);
            obj.Draws(end + 1) = u;
            k = find(u <= cumsum(p), 1);
            if isempty(k), k = find(p > 0, 1, 'last'); end   % guard against round-off at u ~ 1
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', view.ActionLabels(k), 'Probabilities', p, 'Likelihood', p(k));
        end

        function observe(obj, feedback)
            if ~isempty(obj.Log), obj.Log.add("observe:" + feedback.AgentID); end
        end
    end
end
