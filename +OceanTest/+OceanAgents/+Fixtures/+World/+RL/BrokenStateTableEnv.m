classdef BrokenStateTableEnv < OceanStats.RL.Env.Classic.RandomWalkEnv
    % BROKENSTATETABLEENV  Violates exactly RL-8 (Enumerable): the declared state table does not cover the reachable
    % non-terminal state 3 (state_to_index returns 0 for it).
    methods
        function idx = state_to_index(~, state)
            idx = state(1);
            if idx == 3, idx = 0; end
        end
    end
end
