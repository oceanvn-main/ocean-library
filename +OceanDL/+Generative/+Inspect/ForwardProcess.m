classdef ForwardProcess
    % FORWARDPROCESS  The forward (noising) process made inspectable, in the
    %   notation of Lai et al., "The Principles of Diffusion Models", Sec. 2.2.1:
    %
    %     x_i = alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps,   eps ~ N(0, I)      (Eq. 2.2.1)
    %
    %   (general schedules: x_t = alpha(t) x_0 + sigma(t) eps). No model is
    %   involved and any step is ONE formula -- no need to run the chain --
    %   so a GUI slider over i = 0..L costs O(1) per frame.
    %
    %   [x_t, info] = ForwardProcess.at(x0, schedule, t, eps)
    %   info        = ForwardProcess.describe(schedule, t)     % the numbers at level t
    %   F           = ForwardProcess.trajectory(x0, schedule, ts, eps)
    %   T           = ForwardProcess.scheduleTable(schedule, n)

    methods (Static)
        function [xt, info] = at(x0, schedule, t, eps)
            info = OceanDL.Generative.Inspect.ForwardProcess.describe(schedule, t);
            xt = info.AlphaBar * x0 + info.Sigma * eps;
        end

        function info = describe(schedule, t)
            % Schedule quantities at level t (and the discrete step i for LinearVP).
            a = schedule.alpha(t); s = schedule.sigma(t);
            info = struct('T', t, 'AlphaBar', a, 'Sigma', s, 'SigmaSq', s ^ 2, ...
                'SNR', (a / s) ^ 2, 'LogSNR', 2 * (log(a) - log(s)), ...
                'Step', NaN, 'L', NaN, 'BetaSq', NaN, 'Beta', NaN, 'AlphaStep', NaN);
            if isa(schedule, 'OceanDL.Generative.Diffusion.Schedules.LinearVP')
                i = round(t * schedule.L);
                info.Step = i; info.L = schedule.L;
                if i >= 1
                    info.BetaSq = schedule.betaSq(i); info.Beta = schedule.beta(i); info.AlphaStep = schedule.alphaStep(i);
                end
            end
        end

        function F = trajectory(x0, schedule, ts, eps)
            % x_t at every level in ts with the SAME noise eps.
            F = struct('T', num2cell(ts), 'X', [], 'Info', []);
            for k = 1:numel(ts)
                [F(k).X, F(k).Info] = OceanDL.Generative.Inspect.ForwardProcess.at(x0, schedule, ts(k), eps);
            end
        end

        function T = scheduleTable(schedule, n)
            % alphaBar, sigma, SNR over n + 1 levels (LinearVP: every step i = 0..L).
            if nargin < 2
                if isa(schedule, 'OceanDL.Generative.Diffusion.Schedules.LinearVP'), n = schedule.L; else, n = 200; end
            end
            t = linspace(0, 1, n + 1).';
            a = schedule.alpha(t); s = schedule.sigma(t);
            T = table(t, a, s, (a ./ s) .^ 2, 'VariableNames', {'T', 'AlphaBar', 'Sigma', 'SNR'});
        end
    end
end
