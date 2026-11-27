classdef (Abstract) ISampler
    % ISAMPLER  Reverse-time integration from noise to data (Strategy pattern).
    %
    %   X = sampler.sample(predict, schedule, shape)
    %   X = sampler.sample(predict, schedule, shape, 'Stream', s, 'Like', single(0))
    %
    %   predict: [x0hat, epshat] = predict(x, t)  -- any denoiser (a trained
    %   DiffusionModel, classifier-free guided or not, or an Analytic one).
    %   Loop (shared by all samplers):
    %     t = schedule.grid(Steps);  x = priorScale * eps
    %     for consecutive (t_i, t_{i+1}):  x = step(...)
    %     if sigma(t_end) > 0:  x = E[x0 | x]        (final denoising)
    %   NFE (network evaluations) is counted in the last 'Info' output.
    %   'TStart' t0 < 1 starts the reverse process part-way (grid scaled to
    %   [t0 .. 0]); pass the partly noised image as 'X1' -- a "round trip"
    %   (noise an image to level t0, then denoise it; SDEdit, Meng et al. 2022).
    %   'ClipX0' [lo hi] clips the clean estimate x_phi to the data range at
    %   every step and re-derives the noise estimate from it (Ho et al.'s
    %   reference sampler, clip_denoised; data in [-1, 1] -> [-1 1]).

    properties (SetAccess = protected)
        Steps (1,1) double {mustBePositive, mustBeInteger} = 50
    end

    methods (Abstract)
        [x, nfe] = step(obj, predict, schedule, x, t, s, stream)
    end

    methods
        function [X, info] = sample(obj, predict, schedule, shape, opts)
            arguments
                obj
                predict function_handle
                schedule OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
                shape (1,:) double
                opts.Stream = RandStream.getGlobalStream()
                opts.Like = 0
                opts.X1 = []                    % optional starting sample (else priorScale * eps)
                opts.TStart (1,1) double {mustBeInRange(opts.TStart, 0, 1, 'exclude-lower')} = 1
                opts.ClipX0 (1,:) double = []
            end
            if ~isempty(opts.ClipX0)
                predict = OceanDL.Generative.Diffusion.Samplers.ISampler.clipped(predict, schedule, opts.ClipX0);
            end
            grid = opts.TStart * schedule.grid(obj.Steps);
            if isempty(opts.X1)
                X = schedule.priorScale() * cast(randn(opts.Stream, shape), 'like', opts.Like);
            else
                X = opts.X1;
            end
            nfe = 0;
            for i = 1:numel(grid) - 1
                [X, n] = obj.step(predict, schedule, X, grid(i), grid(i + 1), opts.Stream);
                nfe = nfe + n;
            end
            if schedule.sigma(grid(end)) > 0
                X = predict(X, grid(end));
                nfe = nfe + 1;
            end
            info = struct('NFE', nfe, 'Grid', grid);
        end
    end

    methods (Static)
        function fn = clipped(predict, schedule, range)
            % Predictor whose clean estimate is clipped to [range(1), range(2)];
            % the noise estimate is recomputed so x_t = alphaBar x_phi + sigma eps_phi still holds.
            fn = @(x, t) OceanDL.Generative.Diffusion.Samplers.ISampler.clipCall(predict, schedule, range, x, t);
        end

        function [x0, eps] = clipCall(predict, schedule, range, x, t)
            x0 = predict(x, t);
            x0 = min(range(2), max(range(1), x0));
            eps = (x - schedule.alpha(t) * x0) / schedule.sigma(t);
        end
    end
end
