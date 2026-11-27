classdef DiffusionModel < OceanDL.nn.Module
    % DIFFUSIONMODEL  Training objective + sampling interface for the whole
    %   diffusion family around ANY denoiser network  net.forward(x, t, y).
    %
    %   TRAINING  forward(X, y) -> loss (Tensor):
    %     t ~ TimeSampler,  eps ~ N(0, I),  x_t = alpha(t) X + sigma(t) eps
    %     p = net(x_t, netTime(t), y~)          y~ = y, or the NULL label with
    %                                           prob. CondDropout (classifier-free
    %                                           guidance training, Ho & Salimans 2022)
    %     loss = mean_n  w(t_n) * mean( (p_n - target_n).^2 )
    %   'Target':  "eps" (DDPM) | "x0" | "v" | "velocity" (flow matching) |
    %              "edm" (Karras 2022 preconditioning, EDM schedule only):
    %                 D(x) = c_skip x + c_out F(c_in x, log(sigma)/4)
    %                 c_in = 1/sqrt(s^2 + sd^2), c_skip = sd^2/(s^2 + sd^2),
    %                 c_out = s sd / sqrt(s^2 + sd^2); F is trained on
    %                 (x0 - c_skip x)/c_out with unit weight (= EDM's lambda).
    %   'Weighting': "simple" (w = 1) | "minsnr" (Hang et al. 2023): the loss
    %              in x0 units is weighted by min(SNR, Gamma), converted to the
    %              target's units:  w = min(SNR, Gamma) * (dx0/dp)^2.
    %
    %   SAMPLING  predict = predictor('Labels', y, 'Guidance', w) gives
    %     [x0hat, epshat] = predict(x, t) for any ISampler; with guidance
    %     p = p_null + w (p_cond - p_null)  (w = 1: conditional, w = 0: unconditional).
    %   generate(n, shape, sampler, ...) = sampler.sample(predictor(...), ...).

    properties (SetAccess = protected)
        Net
        Schedule
        Target (1,1) string
        Times
        Weighting (1,1) string
        Gamma (1,1) double
        CondDropout (1,1) double
        NumClasses (1,1) double
        Layout (1,1) string
    end

    properties (SetAccess = private)
        Stream
        LastStats = struct()
    end

    methods
        function obj = DiffusionModel(net, schedule, opts)
            arguments
                net OceanDL.nn.Module
                schedule OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
                opts.Target (1,1) string {mustBeMember(opts.Target, ["eps", "x0", "v", "velocity", "edm"])} = "eps"
                opts.TimeSampler = []
                opts.Weighting (1,1) string {mustBeMember(opts.Weighting, ["simple", "minsnr"])} = "simple"
                opts.Gamma (1,1) double {mustBePositive} = 5
                opts.CondDropout (1,1) double {mustBeInRange(opts.CondDropout, 0, 1)} = 0.1
                opts.NumClasses (1,1) double = 0
                opts.Layout (1,1) string {mustBeMember(opts.Layout, ["SSCB", "CB"])} = "SSCB"
                opts.Seed = []
            end
            isEdm = isa(schedule, 'OceanDL.Generative.Diffusion.Schedules.EDM');
            if opts.Target == "edm" && ~isEdm
                error('OceanDL:Generative:Diffusion:DiffusionModel:Target', 'Target "edm" needs the EDM schedule.');
            end
            if opts.Target == "eps" && schedule.alpha(1) == 0
                error('OceanDL:Generative:Diffusion:DiffusionModel:Target', ...
                    'eps-prediction is singular where alpha = 0 (rectified flow at t = 1); use "velocity", "v" or "x0".');
            end
            obj.Net = net; obj.Schedule = schedule; obj.Target = opts.Target;
            obj.Weighting = opts.Weighting; obj.Gamma = opts.Gamma;
            obj.CondDropout = opts.CondDropout; obj.NumClasses = opts.NumClasses; obj.Layout = opts.Layout;
            if isempty(opts.TimeSampler)
                if isEdm, obj.Times = OceanDL.Generative.Diffusion.TimeSampler("lognormal");
                elseif isa(schedule, 'OceanDL.Generative.Diffusion.Schedules.RectifiedFlow')
                    obj.Times = OceanDL.Generative.Diffusion.TimeSampler("logitnormal");
                else, obj.Times = OceanDL.Generative.Diffusion.TimeSampler("uniform"); end
            else
                obj.Times = opts.TimeSampler;
            end
            if isempty(opts.Seed), obj.Stream = RandStream.getGlobalStream();
            else, obj.Stream = RandStream('mt19937ar', 'Seed', opts.Seed); end
        end

        function L = forward(obj, X, labels)
            if nargin < 3, labels = []; end
            N = OceanDL.Generative.Latent.batchSize(X, obj.Layout);
            sch = obj.Schedule;
            t = obj.Times.sample(N, obj.Stream, sch);
            [a, s, da, ds] = obj.coefficients(t, X);
            e = cast(randn(obj.Stream, size(X)), 'like', X);
            %Forward diffusion closed-form equation

            xt = a .* X + s .* e;

            % Applies Conditioning Label Dropout for Classifier-Free Guidance (CFG) (Ho & Salimans 2022).
            labels = obj.dropLabels(labels, N);
            if obj.Target == "edm"
                [cin, cskip, cout] = obj.edmScalings(s);
                p = obj.Net.forward(cin .* xt, sch.netTime(t), labels);
                target = (X - cskip .* xt) ./ cout;
                w = ones(1, N);
            else
                p = obj.Net.forward(xt, sch.netTime(t), labels);
                target = OceanDL.Generative.Diffusion.Targets.fromX0Eps(obj.Target, X, e, a, s, da, ds);
                w = obj.weights(t);
            end
            bd = obj.batchDim();
            others = setdiff(1:max(bd, ndims(X)), bd);
            perSample = reshape(mean((p - target) .^ 2, others), 1, N);          % mean over non-batch dims
            L = sum(perSample .* cast(w, 'like', X), 'all') / N;
            obj.LastStats = struct('Loss', double(gather(L.Data)), 'T', t, 'PerSample', double(gather(perSample.Data)));
        end

        function predict = predictor(obj, opts)
            arguments
                obj
                opts.Labels = []
                opts.Guidance (1,1) double = 1
            end
            predict = @(x, t) obj.predictAt(x, t, opts.Labels, opts.Guidance);
        end

        function [X, info] = generate(obj, n, shape, sampler, opts)
            arguments
                obj
                n (1,1) double
                shape (1,:) double                   % per-sample shape, e.g. [16 16 1]
                sampler OceanDL.Generative.Diffusion.Samplers.ISampler
                opts.Labels = []
                opts.Guidance (1,1) double = 1
                opts.Stream = RandStream.getGlobalStream()
            end
            P = obj.Net.parameters();
            was = obj.Net.Training; obj.Net.eval();
            cleanup = onCleanup(@() obj.Net.train(was)); %Fallback to trainning mode if the code crash
            [X, info] = sampler.sample(obj.predictor('Labels', opts.Labels, 'Guidance', opts.Guidance), ...
                obj.Schedule, [shape n], 'Stream', opts.Stream, 'Like', P{1}.Data([]));
        end

        function [x0, e] = predictAt(obj, x, t, labels, guidance)
            N = OceanDL.Generative.Latent.batchSize(x, obj.Layout);
            sch = obj.Schedule;
            tt = repmat(sch.netTime(t), 1, N);
            a = sch.alpha(t); s = sch.sigma(t); da = sch.dalpha(t); ds = sch.dsigma(t);
            xin = x;
            if obj.Target == "edm", xin = x / sqrt(s ^ 2 + sch.SigmaData ^ 2); end
            %Define
            run = @(xx, ttt, yy) OceanDL.Autograd.Tensor.raw(OceanDL.Autograd.GradMode.noGrad(@() obj.Net.forward(xx, ttt, yy)));
            
            if isempty(labels) || guidance == 1 || obj.NumClasses == 0
                p = run(xin, tt, labels);
                if ~isempty(labels) && guidance ~= 1 && obj.NumClasses == 0
                    error('OceanDL:Generative:Diffusion:DiffusionModel:Guidance', 'Guidance needs a class-conditional model (NumClasses > 0).');
                end
            else
                null = repmat(obj.NumClasses + 1, N, 1);
                d = obj.batchDim();
                both = run(cat(d, xin, xin), [tt tt], [labels(:); null]);          % one batched pass
                [pc, pu] = obj.splitBatch(both, N);
                p = pu + guidance * (pc - pu);
            end
            if obj.Target == "edm"
                [~, cskip, cout] = obj.edmScalings(s);
                x0 = cskip * x + cout * p;
                e = (x - x0) / s;
            else
                [x0, e] = OceanDL.Generative.Diffusion.Targets.toX0Eps(obj.Target, p, x, a, s, da, ds);
            end
        end
    end

    methods (Access = private)
        function d = batchDim(obj)
            if obj.Layout == "SSCB", d = 4; else, d = 2; end
        end

        function [a, s, da, ds] = coefficients(obj, t, X)
            % Per-sample schedule values, shaped to broadcast along the batch axis.
            sch = obj.Schedule;
            if obj.Layout == "SSCB", shp = [1 1 1 numel(t)]; else, shp = [1 numel(t)]; end
            c = @(v) cast(reshape(v, shp), 'like', X);
            a = c(sch.alpha(t)); s = c(sch.sigma(t)); da = c(sch.dalpha(t)); ds = c(sch.dsigma(t));
        end

        function w = weights(obj, t)
            sch = obj.Schedule;
            if obj.Weighting == "simple"
                w = ones(1, numel(t));
                return;
            end
            f = OceanDL.Generative.Diffusion.Targets.x0ErrorFactor(obj.Target, sch.alpha(t), sch.sigma(t), sch.dalpha(t), sch.dsigma(t));
            w = min(sch.snr(t), obj.Gamma) .* f .^ 2;
        end

        function [cin, cskip, cout] = edmScalings(obj, s)
            sd = obj.Schedule.SigmaData;
            cin = 1 ./ sqrt(s .^ 2 + sd ^ 2);
            cskip = sd ^ 2 ./ (s .^ 2 + sd ^ 2);
            cout = s * sd ./ sqrt(s .^ 2 + sd ^ 2);
        end

        function y = dropLabels(obj, y, N)
            if obj.NumClasses == 0 || isempty(y)
                y = [];
                return;
            end
            y = reshape(double(y), N, 1);
            if obj.Training && obj.CondDropout > 0
                y(rand(obj.Stream, N, 1) < obj.CondDropout) = obj.NumClasses + 1;
            end
        end

        function [a, b] = splitBatch(obj, both, N)
            if obj.Layout == "SSCB"
                a = both(:, :, :, 1:N); b = both(:, :, :, N+1:end);
            else
                a = both(:, 1:N); b = both(:, N+1:end);
            end
        end
    end
end
