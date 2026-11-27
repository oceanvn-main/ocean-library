classdef QSigmaTarget < OceanStats.RL.TD.Targets.ITarget
    % QSIGMATARGET  Target for unifying n-step Q(sigma) algorithm.
    %   Sutton & Barto (2018) Sec 7.6, Eq. 7.17: Blends Sarsa (sigma=1) and Tree Backup (sigma=0).
    properties (Constant)
        Latex = "G_{t:t+n} = R_{t+1} + \gamma [\sigma_{t+1} A_{t+1} + (1-\sigma_{t+1})\sum_a \pi(a \mid S_{t+1}) Q(S_{t+1},a)]"
        Reference = "Sutton & Barto (2018) Sec 7.6, Q(sigma)"
        IsOffPolicy = true
    end
    properties
        Sigma (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(Sigma, 1)} = 0.5
    end
    methods
        function obj = QSigmaTarget(sigma)
            if nargin >= 1, obj.Sigma = sigma; end
        end
        function y = compute(obj, r, qNext, aNext, terminated, gamma, probsNext, valid)
            if terminated
                b = 0;
            else
                probs = probsNext(valid);
                qVals = qNext(valid);
                expVal = sum(probs .* qVals);
                sampVal = qNext(aNext);
                b = obj.Sigma * sampVal + (1 - obj.Sigma) * expVal;
            end
            y = OceanStats.RL.TD.Functional.oneStepTarget(r, b, terminated, gamma);
        end
        function s = name(~), s = "qsigma"; end
    end
end
