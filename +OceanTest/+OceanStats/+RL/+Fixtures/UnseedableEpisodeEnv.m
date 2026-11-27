classdef UnseedableEpisodeEnv < OceanStats.RL.MonteCarlo.IEpisodeEnvironment
    % UNSEEDABLEEPISODEENV  TEST FIXTURE: a minimal legacy-interface environment WITHOUT setSeed (its reward noise comes
    % from the global stream). States 1, 2, 3 (3 terminal), one action; the reward on the last step is random.
    properties (Access = private)
        S = 1
    end
    methods
        function s = reset(obj), obj.S = 1; s = 1; end
        function [s, r, done] = step(obj, ~)
            obj.S = obj.S + 1;  s = obj.S;  r = rand();  done = obj.S == 3;
        end
        function a = get_valid_actions(obj, s), a = ones(1, ~obj.is_terminal(s)); end
        function s = get_all_states(~), s = (1:3)'; end
        function i = state_to_index(~, s), i = s(1); end
        function i = action_to_index(~, a), i = a; end
        function n = get_num_actions(~), n = 1; end
        function tf = is_terminal(~, s), tf = s(1) == 3; end
    end
end
