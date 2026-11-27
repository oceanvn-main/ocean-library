classdef EulerODE < OceanDL.Generative.Diffusion.Samplers.ISampler
    % EULERODE  First-order integration of the probability-flow ODE
    %     dx/dt = a'(t) x0hat(x, t) + s'(t) epshat(x, t)
    %   (for rectified flow this is the predicted velocity eps - x0):
    %     x_s = x_t + (s - t) f(x_t, t).     Global error O(1 / Steps).

    methods
        function obj = EulerODE(opts)
            arguments
                opts.Steps (1,1) double = 50
            end
            obj.Steps = opts.Steps;
        end

        function [x, nfe] = step(~, predict, schedule, x, t, s, ~)
            x = x + (s - t) * OceanDL.Generative.Diffusion.Samplers.EulerODE.drift(predict, schedule, x, t);
            nfe = 1;
        end
    end

    methods (Static)
        function f = drift(predict, schedule, x, t)
            [x0, e] = predict(x, t);
            f = schedule.dalpha(t) * x0 + schedule.dsigma(t) * e;
        end
    end
end
