classdef GymTabularEnv < OceanStats.RL.Env.External.GymEnv & OceanStats.RL.Env.IModelEnvironment
    % GYMTABULARENV  A Gymnasium toy-text environment with its exact model (env.unwrapped.P) exposed as
    % IModelEnvironment, so anything that needs exact dynamics or a state table can run on it.
    %   env = OceanStats.RL.Env.External.GymTabularEnv("FrozenLake-v1");   [next, prob, reward] = env.get_transitions(1, 2);
    % A state is terminal when some transition into it is flagged done in P. Duplicate successors in P are merged
    % (probabilities added, probability-weighted mean reward).

    properties (SetAccess = protected)
        NumStates double = 0
        TerminalMask (:,1) logical
        ValidMask logical
    end
    properties (Access = private)
        Next
        Prob
        Reward
    end

    methods
        function obj = GymTabularEnv(name, options)
            arguments
                name (1,1) string
                options.Strict (1,1) logical = true
            end
            obj@OceanStats.RL.Env.External.GymEnv(name, 'Strict', options.Strict);
            m = cell(obj.Py.model());
            obj.Next = double(m{1}) + 1;  obj.Prob = double(m{2});  obj.Reward = double(m{3});
            done = double(m{4}) > 0 & obj.Prob > 0;
            obj.NumStates = size(obj.Next, 1);
            obj.TerminalMask = false(obj.NumStates, 1);
            obj.TerminalMask(unique(obj.Next(done))) = true;
            obj.ValidMask = double(m{5}) > 0 & ~obj.TerminalMask;
        end

        function states = get_all_states(obj), states = (1:obj.NumStates)'; end
        function actions = get_all_actions(obj), actions = (1:obj.NumActions)'; end
        function idx = state_to_index(~, state), idx = state(1); end
        function k = action_to_index(~, action), k = action; end
        function actions = get_valid_actions(obj, state), actions = find(obj.ValidMask(state(1), :)); end
        function tf = is_terminal(obj, state), tf = obj.TerminalMask(state(1)); end

        function [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
            n = squeeze(obj.Next(state(1), action, :));  p = squeeze(obj.Prob(state(1), action, :));
            r = squeeze(obj.Reward(state(1), action, :));
            keep = p > 0;  n = n(keep);  p = p(keep);  r = r(keep);
            [next_states, ~, g] = unique(n);
            probabilities = accumarray(g, p);
            immediate_rewards = accumarray(g, p .* r) ./ probabilities;
        end
    end
end
