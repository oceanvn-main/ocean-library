classdef ExplorationPolicy < OceanStats.RL.TD.Policies.IBehaviourPolicy
    % EXPLORATIONPOLICY  Behaviour = an exploration rule (epsilon-greedy, softmax, ...) applied to the value function's
    % behaviour estimates. Used for control.
    properties (Constant)
        Latex = ""                                     % no closed form: the rule is whatever Exploration strategy is supplied
        Reference = "exploration rule applied to the action values (see the Exploration strategies)"
    end
    properties (SetAccess = private)
        Exploration
    end
    methods
        function obj = ExplorationPolicy(exploration)
            arguments
                exploration (1,1) OceanStats.RL.TD.Exploration.IExploration
            end
            obj.Exploration = exploration;
        end
        function s = name(~), s = "exploration"; end
        function p = probabilities(obj, values, si, valid, progress)
            p = obj.Exploration.probabilities(values.behaviourEstimates(si), valid, progress);
        end
    end
end
