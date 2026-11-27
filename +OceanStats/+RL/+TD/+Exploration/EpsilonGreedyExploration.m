classdef EpsilonGreedyExploration < OceanStats.RL.TD.Exploration.IExploration
    % EPSILONGREEDY  Fixed epsilon; ties for the greedy action are split evenly.
    properties (SetAccess = private)
        Epsilon (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Epsilon, 1)} = 0.1
    end
    properties (Constant)
        Latex = "\pi(a \mid s) = \dfrac{\varepsilon}{|\mathcal{A}(s)|} + (1-\varepsilon)\,\dfrac{\mathbb{1}[a \in \arg\max_b q(b)]}{|\arg\max_b q(b)|}"
        Reference = "Sutton & Barto (2018) sec. 2.2 / 5.4"
    end
    methods
        function obj = EpsilonGreedyExploration(epsilon)
            arguments
                epsilon (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(epsilon, 1)} = 0.1
            end
            obj.Epsilon = epsilon;
        end
        function p = probabilities(obj, q, valid, ~)
            p = OceanStats.RL.TD.Functional.epsilonGreedyProbs(q, valid, obj.Epsilon);
        end
        function s = name(~), s = "epsilongreedy"; end
    end
end
