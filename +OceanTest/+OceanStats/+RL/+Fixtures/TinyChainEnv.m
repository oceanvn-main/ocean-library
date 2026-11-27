classdef TinyChainEnv < OceanStats.RL.Env.TabularEnvironment
    % TINYCHAINENV  TEST FIXTURE: 4-state chain used to test the +Env core without any learner.
    %   1 --a1--> 2 (r 0)        1 --a2--> 1 (r -1)
    %   2 --a1--> 3 (p .8) | 2 (p .2)  (r 0)          (action 2 not admissible in state 2)
    %   3 --a1--> 4 (r 1)        4 terminal
    methods
        function obj = TinyChainEnv()
            model = cell(4, 2);
            model{1, 1} = struct('Next', 2, 'Prob', 1, 'Reward', 0);
            model{1, 2} = struct('Next', 1, 'Prob', 1, 'Reward', -1);
            model{2, 1} = struct('Next', [3; 2], 'Prob', [0.8; 0.2], 'Reward', [0; 0]);
            model{3, 1} = struct('Next', 4, 'Prob', 1, 'Reward', 1);
            obj.setModel_(model, [false; false; false; true], [1; 0; 0; 0]);
        end
        function setBadModel(obj, model, terminalMask, startProbs)
            obj.setModel_(model, terminalMask, startProbs);
        end
    end
end
