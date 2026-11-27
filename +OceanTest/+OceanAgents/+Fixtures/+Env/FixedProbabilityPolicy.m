classdef FixedProbabilityPolicy < OceanAgents.Core.IPolicy
    % FIXEDPROBABILITYPOLICY  Test fixture: an observation-independent policy with fixed probabilities over the view's
    % columns. Each decision consumes exactly ONE rand(stream) draw and picks the column by inverse CDF, so a test can
    % reproduce the choice sequence from the stream alone (A0 section 9.4).
    %
    %   pol = OceanTest.OceanAgents.Fixtures.Env.FixedProbabilityPolicy([0.7 0.3]);
    %   d   = pol.act(view, RandStream('mrg32k3a', 'Seed', 1));   % label 1 w.p. 0.7, label 2 w.p. 0.3
    %
    % Probability mass on a column that is illegal in the view is an error (no silent renormalisation).

    properties (SetAccess = private)
        P double = 1
    end

    methods
        function obj = FixedProbabilityPolicy(p)
            arguments
                p (1,:) double {mustBeFinite, mustBeNonnegative}
            end
            if abs(sum(p) - 1) > 1e-12
                error('OceanTest:Fixtures:BadProbabilities', 'Probabilities must sum to 1.');
            end
            obj.P = p;
        end

        function d = describe(obj)
            d = struct('Name', "FixedProbabilityPolicy", 'Version', "1", 'NumActions', numel(obj.P), ...
                'ProvidesLikelihood', true);
        end

        function p = probabilities(obj, view)
            if numel(view.ActionLabels) ~= numel(obj.P) || any(obj.P(~view.ActionMask) > 0)
                error('OceanAgents:Policies:IllegalTableAction', 'Fixed probabilities put mass on illegal columns.');
            end
            p = obj.P;
        end

        function dec = act(obj, view, stream)
            p = obj.probabilities(view);
            k = find(rand(stream) <= cumsum(p), 1);
            if isempty(k), k = find(p > 0, 1, 'last'); end     % guards cumsum rounding just below 1
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', view.ActionLabels(k), 'Probabilities', p, 'Likelihood', p(k));
        end
    end
end
