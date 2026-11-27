classdef Transition
    % TRANSITION  A learner's private experience record for one decision interval, built by the runtime from the
    % agent's own decision and the feedback that followed it. Learners consume these; they never touch a world.
    %
    %   Reward is the discount-aggregated reward over the interval (the single reward when Duration = 1);
    %   Discount = Gamma^Duration with the owning agent's Gamma. NextMask is the successor legality
    %   (for a truncation: the declared successor mask, all false when unknown).
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.4.

    properties (SetAccess = immutable)
        AgentID string = ""
        EpisodeID double = 0
        Observation double = []
        Action double = 0
        ActionLabels double = zeros(1, 0)
        Reward double = 0
        NextObservation double = []
        NextActionLabels double = zeros(1, 0)
        NextMask logical = false(1, 0)
        Terminated logical = false
        Truncated logical = false
        Duration double = 1
        Discount double = 1
        PolicyVersion double = 0
        Likelihood double = []
        NextSensed logical = true   % false: no next view was sensed (NextObservation is the previous one)
    end

    methods
        function obj = Transition(options)
            arguments
                options.AgentID
                options.EpisodeID
                options.Observation
                options.Action
                options.ActionLabels
                options.Reward
                options.NextObservation
                options.NextActionLabels
                options.NextMask
                options.Terminated = false
                options.Truncated = false
                options.Duration = 1
                options.Discount
                options.PolicyVersion = 0
                options.Likelihood = []
                options.NextSensed = true
            end
            requireFields_(options, {'AgentID', 'EpisodeID', 'Observation', 'Action', 'ActionLabels', 'Reward', ...
                'NextObservation', 'NextActionLabels', 'NextMask', 'Discount'}, 'Transition');
            obj.AgentID = asString_(options.AgentID, 'AgentID');
            obj.EpisodeID = asId_(options.EpisodeID, 'EpisodeID');
            obj.Observation = asFinite_(options.Observation, 'Observation');
            obj.ActionLabels = reshape(asFinite_(options.ActionLabels, 'ActionLabels'), 1, []);
            a = asFinite_(options.Action, 'Action');
            if ~isscalar(a) || ~any(obj.ActionLabels == a)
                fail_('InvalidRecord', 'Action must be one of ActionLabels.');
            end
            obj.Action = a;
            r = asFinite_(options.Reward, 'Reward');
            if ~isscalar(r), fail_('InvalidRecord', 'Reward must be a scalar.'); end
            obj.Reward = r;
            obj.NextObservation = asFinite_(options.NextObservation, 'NextObservation');
            obj.NextActionLabels = reshape(asFinite_(options.NextActionLabels, 'NextActionLabels'), 1, []);
            obj.NextMask = asRowFlags_(options.NextMask, numel(obj.NextActionLabels), 'NextMask');
            obj.Terminated = asFlag_(options.Terminated, 'Terminated');
            obj.Truncated = asFlag_(options.Truncated, 'Truncated');
            if obj.Terminated && obj.Truncated
                fail_('InvalidRecord', 'Terminated and Truncated cannot both be true.');
            end
            d = asId_(options.Duration, 'Duration');
            if d < 1, fail_('InvalidRecord', 'Duration must be a positive integer.'); end
            obj.Duration = d;
            g = asFinite_(options.Discount, 'Discount');
            if ~isscalar(g) || g < 0 || g > 1, fail_('InvalidRecord', 'Discount must be a scalar in [0, 1].'); end
            obj.Discount = g;
            obj.PolicyVersion = asId_(options.PolicyVersion, 'PolicyVersion');
            obj.NextSensed = asFlag_(options.NextSensed, 'NextSensed');
            lik = asFinite_(options.Likelihood, 'Likelihood');
            if ~isempty(lik) && (~isscalar(lik) || lik <= 0 || lik > 1)
                fail_('InvalidRecord', 'Likelihood must be a scalar in (0, 1].');
            end
            obj.Likelihood = lik;
        end
    end
end
