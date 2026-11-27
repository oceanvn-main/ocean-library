classdef (Abstract) IBehaviourPolicy < OceanStats.RL.TD.ITDComponent
    % IBEHAVIOURPOLICY  How actions are chosen while learning: the probability of each action at a state, given the value
    % function (an exploration rule over Q, or a fixed policy that is being evaluated).
    %
    %   p = policy.probabilities(values, si, valid, progress)     1 x A, zero on inadmissible actions
    %   a = policy.sample(p, stream)                              draw an action index from p with the given RandStream
    %   a = policy.mostLikelyAction(values, si, valid)            the action index reported as pi(s) (default: argmax of p)
    methods (Abstract)
        p = probabilities(obj, values, si, valid, progress)
    end
    methods (Access = protected)
        function a = assumptions(~)
            a = {'chooses actions while learning: probabilities over the admissible actions of a state'};
        end
    end
    methods
        function a = mostLikelyAction(obj, values, si, valid)
            [~, a] = max(obj.probabilities(values, si, valid, 1));
        end
        function a = sample(~, p, stream)
            c = cumsum(p);
            a = find(rand(stream) * c(end) <= c, 1);
        end
    end
end
