classdef (Abstract) IExplorationAgent < handle
    % IEXPLORATIONAGENT  Shared shell of the model-based exploration agents (R-max, E³): a tabular count model,
    % the admissible-action mask seen so far, and an environment loop.
    %
    %   a  = agent.act(s, actionMask)               choose an action in state s (actionMask: 1 x A logical)
    %   agent.observe(s, a, r, s2, terminated)      learn from one real transition
    %   tr = agent.run(env, 'Steps', 10000, 'Seed', 1)
    %        env: OceanStats.RL.Env.IEnvironment with a state-index observation (tabular environments).
    %        Runs exactly Steps real transitions; a terminated or truncated episode is reset and continues.
    %        tr: Rewards (1 x Steps), States, Actions, Modes (agent-specific labels), Episodes.
    properties (SetAccess = protected)
        NumStates (1,1) double
        NumActions (1,1) double
        DiscountFactor (1,1) double
        MaxReward (1,1) double
        Counts                      % OceanStats.RL.PAC.TransitionCounts
        Mask logical                % S x A admissible actions as last reported (true until seen)
        LastMode (1,1) string = ""
    end
    methods (Abstract)
        a = act(obj, s, actionMask)
        observe(obj, s, a, r, s2, terminated)
    end
    methods
        function tr = run(obj, env, opts)
            arguments
                obj
                env (1,1) OceanStats.RL.Env.IEnvironment
                opts.Steps (1,1) double {mustBeInteger, mustBePositive} = 10000
                opts.Seed double {mustBeScalarOrEmpty} = []
            end
            n = opts.Steps;
            rewards = zeros(1, n); states = zeros(1, n); acts = zeros(1, n); modes = strings(1, n);
            [s, info] = env.reset('Seed', opts.Seed); episodes = 1; obj.episodeStart_();
            for t = 1:n
                a = obj.act(s, info.action_mask);
                [s2, r, terminated, truncated, info] = env.step(a);
                obj.observe(s, a, r, s2, terminated);
                rewards(t) = r; states(t) = s; acts(t) = a; modes(t) = obj.LastMode;
                s = s2;
                if terminated || truncated
                    [s, info] = env.reset(); episodes = episodes + 1; obj.episodeStart_();
                end
            end
            tr = struct('Rewards', rewards, 'States', states, 'Actions', acts, 'Modes', modes, 'Episodes', episodes);
        end
    end
    methods (Access = protected)
        function episodeStart_(~)
            % Hook called after every environment reset (default: nothing).
        end
        function init_(obj, numStates, numActions, gamma, rmax, cap)
            obj.NumStates = numStates; obj.NumActions = numActions;
            obj.DiscountFactor = gamma; obj.MaxReward = rmax;
            obj.Counts = OceanStats.RL.PAC.TransitionCounts(numStates, numActions, 'Cap', cap);
            obj.Mask = true(numStates, numActions);
        end
        function noteMask_(obj, s, actionMask)
            m = logical(actionMask(:)');
            if numel(m) ~= obj.NumActions
                error('OceanStats:RL:PAC:BadMask', 'The action mask must have %d entries.', obj.NumActions);
            end
            if ~any(m)
                error('OceanStats:RL:PAC:NoAction', 'State %d admits no action.', s);
            end
            obj.Mask(s, :) = m;
        end
        function checkReward_(obj, r, rmin)
            if r > obj.MaxReward + 1e-12 || r < rmin - 1e-12
                error('OceanStats:RL:PAC:RewardOutOfRange', ...
                    'Reward %g is outside the declared range [%g, %g]; the guarantees need a valid bound.', r, rmin, obj.MaxReward);
            end
        end
    end
end
