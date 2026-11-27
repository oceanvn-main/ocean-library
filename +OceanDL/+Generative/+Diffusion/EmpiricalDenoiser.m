classdef EmpiricalDenoiser < handle
    % EMPIRICALDENOISER  The exact (Bayes-optimal) denoiser when the data
    %   distribution IS a finite set of images {mu_1 .. mu_K} -- the ideal
    %   diffusion model of a training set, with no network and no training.
    %
    %   Under x_t = a x0 + s eps with x0 drawn uniformly from the set (each
    %   image optionally blurred into N(mu_k, SmoothStd^2 I), a kernel
    %   density estimate):
    %     v    = a^2 SmoothStd^2 + s^2
    %     r_k  = softmax_k( -||x_t - a mu_k||^2 / (2 v) )          responsibilities
    %     E[x0 | x_t]  = sum_k r_k ( mu_k + a SmoothStd^2 / v (x_t - a mu_k) )
    %     E[eps | x_t] = (x_t - a E[x0 | x_t]) / s
    %   r_k says how strongly the model believes the clean image is training
    %   image k. Diagnostics of the last call (per sample):
    %     LastEffectiveN  exp(entropy of r) -- how many training images are
    %                     still "in the running" (K at pure noise -> 1)
    %     LastTopIndex / LastTopWeight   the most likely training image
    %
    %   With SmoothStd = 0 the model can only ever produce training images:
    %   exact, but pure memorisation -- why real models learn a network that
    %   generalises instead.
    %
    %   den = EmpiricalDenoiser(Xtrain, schedule);          Xtrain: H x W x C x K (or D x K)
    %   [x0hat, epshat] = den.predict(x, t);                x: H x W x C x N (same layout)

    properties (SetAccess = private)
        Data                % D x K (double)
        SqNorm              % 1 x K
        Schedule
        SmoothStd (1,1) double
        ImageSize = []      % [H W C] when constructed from images
        LastEffectiveN = []
        LastTopIndex = []
        LastTopWeight = []
    end

    methods
        function obj = EmpiricalDenoiser(X, schedule, opts)
            arguments
                X
                schedule OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
                opts.SmoothStd (1,1) double {mustBeNonnegative} = 0
            end
            if ndims(X) == 4 || (size(X, 3) > 1 && ~ismatrix(X))
                obj.ImageSize = size(X, 1:3);
                X = reshape(X, [], size(X, 4));
            end
            obj.Data = double(X);
            obj.SqNorm = sum(obj.Data .^ 2, 1);
            obj.Schedule = schedule;
            obj.SmoothStd = opts.SmoothStd;
        end

        function [x0, eps] = predict(obj, x, t)
            sz = size(x);
            xs = double(reshape(x, size(obj.Data, 1), []));                % D x N
            a = obj.Schedule.alpha(t); s = obj.Schedule.sigma(t);
            v = a ^ 2 * obj.SmoothStd ^ 2 + s ^ 2;
            d2 = sum(xs .^ 2, 1) - 2 * a * (obj.Data.' * xs) + a ^ 2 * obj.SqNorm.';   % K x N  ||x - a mu_k||^2
            logits = -d2 / (2 * v);
            r = exp(logits - max(logits, [], 1)); r = r ./ sum(r, 1);
            m = obj.Data * r;                                               % sum_k r_k mu_k
            x0 = m + (a * obj.SmoothStd ^ 2 / v) * (xs - a * m);
            eps = (xs - a * x0) / s;
            nz = r; nz(nz <= 0) = 1;
            obj.LastEffectiveN = exp(-sum(r .* log(nz), 1));
            [obj.LastTopWeight, obj.LastTopIndex] = max(r, [], 1);
            x0 = cast(reshape(x0, sz), 'like', x);
            eps = cast(reshape(eps, sz), 'like', x);
        end

        function fn = handle(obj)
            % Predictor handle for ISampler.sample.
            fn = @(x, t) obj.predict(x, t);
        end
    end
end
