classdef HarmonicStepSize < OceanStats.RL.TD.StepSize.IStepSize
    % HARMONICSTEPSIZE  alpha_n = min(1, Scale / n^Power). Power=1, Scale=1 is the sample average; Power in
    % (0.5, 1] satisfies the Robbins-Monro conditions (sum alpha = inf, sum alpha^2 < inf) needed for convergence.
    properties (SetAccess = private)
        Power (1,1) double {mustBePositive} = 1
        Scale (1,1) double {mustBePositive} = 1
    end
    properties (Constant)
        Latex = "\alpha_n = \min\!\left(1, \dfrac{c}{n^{p}}\right)"
        Reference = "Robbins & Monro (1951); Sutton & Barto (2018) sec. 2.5"
    end
    methods
        function obj = HarmonicStepSize(options)
            arguments
                options.Power (1,1) double {mustBePositive} = 1
                options.Scale (1,1) double {mustBePositive} = 1
            end
            obj.Power = options.Power;
            obj.Scale = options.Scale;
        end
        function a = alpha(obj, n), a = min(1, obj.Scale ./ (max(n, 1) .^ obj.Power)); end
        function s = name(~), s = "harmonic"; end
    end
end
