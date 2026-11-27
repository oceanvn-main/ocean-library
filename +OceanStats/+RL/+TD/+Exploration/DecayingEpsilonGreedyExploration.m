classdef DecayingEpsilonGreedyExploration < OceanStats.RL.TD.Exploration.IExploration
    % DECAYINGEPSILONGREEDY  epsilon(u) = Final + (Initial - Final) * (1 - u), linear in training progress
    % u in [0,1] (Stable-Baselines3 style exploration schedule; u is supplied by the learner).
    properties (SetAccess = private)
        Initial (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Initial, 1)} = 1
        Final (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Final, 1)} = 0.05
    end
    properties (Constant)
        Latex = "\varepsilon(u) = \varepsilon_{f} + (\varepsilon_{0} - \varepsilon_{f})(1-u), \; u \in [0,1]"
        Reference = "linear exploration schedule (cf. Stable-Baselines3 DQN exploration_fraction)"
    end
    methods
        function obj = DecayingEpsilonGreedyExploration(options)
            arguments
                options.Initial (1,1) double = 1
                options.Final (1,1) double = 0.05
            end
            obj.Initial = options.Initial;
            obj.Final = options.Final;
        end
        function p = probabilities(obj, q, valid, progress)
            eps = obj.Final + (obj.Initial - obj.Final) * (1 - min(max(progress, 0), 1));
            p = OceanStats.RL.TD.Functional.epsilonGreedyProbs(q, valid, eps);
        end
        function s = name(~), s = "decayingepsilongreedy"; end
    end
end
