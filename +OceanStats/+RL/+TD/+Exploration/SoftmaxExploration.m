classdef SoftmaxExploration < OceanStats.RL.TD.Exploration.IExploration
    % SOFTMAXEXPLORATION  Boltzmann exploration with temperature Tau.
    properties (SetAccess = private)
        Tau (1,1) double {mustBePositive} = 1
    end
    properties (Constant)
        Latex = "\pi(a \mid s) = \dfrac{e^{q(a)/\tau}}{\sum_b e^{q(b)/\tau}}"
        Reference = "Sutton & Barto (2018) eq. 2.11 (soft-max action preferences)"
    end
    methods
        function obj = SoftmaxExploration(tau)
            arguments
                tau (1,1) double {mustBePositive} = 1
            end
            obj.Tau = tau;
        end
        function p = probabilities(obj, q, valid, ~)
            p = OceanStats.RL.TD.Functional.softmaxProbs(q, valid, obj.Tau);
        end
        function s = name(~), s = "softmax"; end
    end
end
