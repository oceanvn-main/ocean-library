classdef HeunODE < OceanDL.Generative.Diffusion.Samplers.ISampler
    % HEUNODE  Second-order (trapezoidal predictor-corrector) integration of
    %   the probability-flow ODE (Karras et al. 2022, Algorithm 1):
    %     x' = x_t + h f(x_t, t)                         (Euler predictor)
    %     x_s = x_t + h (f(x_t, t) + f(x', s)) / 2       (corrector), h = s - t
    %   The corrector is skipped when sigma(s) = 0 (f is singular there), as in
    %   the paper. 2 NFE per step; global error O(1 / Steps^2).

    methods
        function obj = HeunODE(opts)
            arguments
                opts.Steps (1,1) double = 18
            end
            obj.Steps = opts.Steps;
        end

        function [x, nfe] = step(~, predict, schedule, x, t, s, ~)
            drift = @OceanDL.Generative.Diffusion.Samplers.EulerODE.drift;
            h = s - t;
            f1 = drift(predict, schedule, x, t);
            xe = x + h * f1;
            nfe = 1;
            if schedule.sigma(s) == 0
                x = xe;
                return;
            end
            f2 = drift(predict, schedule, xe, s);
            x = x + h * (f1 + f2) / 2;
            nfe = 2;
        end
    end
end
