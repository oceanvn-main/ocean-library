classdef ReverseProcess < handle
    % REVERSEPROCESS  The reverse (denoising) process run STEP BY STEP with
    %   every intermediate recorded -- for GUIs (progress, cancel, scrubbing)
    %   and for study. Notation: Lai et al., "The Principles of Diffusion
    %   Models", Sec. 2.2.2-2.2.6.
    %
    %   At each recorded step i (level t_i -> t_{i-1}) a frame holds:
    %     X        x_i, the sampler's current noisy sample (INPUT to the model)
    %     X0       x_phi(x_i, i), the model's estimate of the clean sample (OUTPUT)
    %     Eps      eps_phi(x_i, i), its estimate of the noise (OUTPUT)
    %     Mu, Var  mean and variance sigma^2(i) of the reverse conditional of
    %              Lemma 2.2.2 / Eq. (2.2.4) with x -> x_phi: where DDPM's next
    %              sample is centred and how much fresh noise it adds
    %   Works with any predictor (trained DiffusionModel, EmpiricalDenoiser,
    %   Analytic), any schedule and any ISampler; the final sample equals
    %   sampler.sample(...) for the same start and random stream (tested).
    %
    %   'OnStep', @(frame) ...  (optional) is called after EVERY step with the same
    %   frame struct that Frames would store -- used by recorders (OceanReplay);
    %   unset, the process is unchanged.
    %
    %   rp = ReverseProcess(predict, schedule, sampler, [H W C N], 'Stride', 10);
    %   while ~rp.Done, rp.advance(25); <update progress / check cancel>; end
    %   X = rp.X;  frames = rp.Frames;

    properties (SetAccess = private)
        Sampler
        Schedule
        Grid (1,:) double
        X                       % current sample (the result once Done)
        Frames = struct('Step', {}, 'T', {}, 'TNext', {}, 'X', {}, 'X0', {}, 'Eps', {}, 'Mu', {}, 'Var', {})
        Index (1,1) double = 0  % reverse steps completed
        NFE (1,1) double = 0
        Done (1,1) logical = false
        Stride (1,1) double = 1
        OnStep = []             % optional @(frame) called after every step
    end

    properties (Access = private)
        Predict
        Stream
        Seen (1,1) logical = false
        LastX0 = []
        LastEps = []
    end

    methods
        function obj = ReverseProcess(predict, schedule, sampler, shape, opts)
            arguments
                predict function_handle
                schedule OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
                sampler OceanDL.Generative.Diffusion.Samplers.ISampler
                shape (1,:) double
                opts.X1 = []
                opts.TStart (1,1) double = 1
                opts.Stream = RandStream.getGlobalStream()
                opts.Stride (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.ClipX0 (1,:) double = []
                opts.Like = 0
                opts.OnStep = []
            end
            if ~isempty(opts.ClipX0)
                predict = OceanDL.Generative.Diffusion.Samplers.ISampler.clipped(predict, schedule, opts.ClipX0);
            end
            obj.Predict = predict; obj.Schedule = schedule; obj.Sampler = sampler;
            obj.Stream = opts.Stream; obj.Stride = opts.Stride; obj.OnStep = opts.OnStep;
            obj.Grid = opts.TStart * schedule.grid(sampler.Steps);
            if isempty(opts.X1)
                obj.X = schedule.priorScale() * cast(randn(opts.Stream, shape), 'like', opts.Like);
            else
                obj.X = opts.X1;
            end
        end

        function n = numSteps(obj)
            n = numel(obj.Grid) - 1;
        end

        function done = advance(obj, nSteps)
            % Run up to nSteps reverse steps (Inf = to the end).
            if nargin < 2, nSteps = Inf; end
            k = 0;
            while ~obj.Done && k < nSteps
                i = obj.Index + 1;
                if i > obj.numSteps()
                    obj.finish();
                    break;
                end
                t = obj.Grid(i); s = obj.Grid(i + 1);
                obj.Seen = false;
                [xNext, nfe] = obj.Sampler.step(@(x, tt) obj.recording(x, tt), obj.Schedule, obj.X, t, s, obj.Stream);
                keep = mod(i - 1, obj.Stride) == 0 || i == obj.numSteps();
                if keep || ~isempty(obj.OnStep)
                    [mu, v] = OceanDL.Generative.Inspect.ReverseProcess.posterior(obj.Schedule, t, s, obj.X, obj.LastX0);
                    frame = struct('Step', i, 'T', t, 'TNext', s, 'X', obj.X, 'X0', obj.LastX0, ...
                        'Eps', obj.LastEps, 'Mu', mu, 'Var', v);
                    if keep, obj.Frames(end+1) = frame; end
                    if ~isempty(obj.OnStep), obj.OnStep(frame); end
                end
                obj.X = xNext; obj.NFE = obj.NFE + nfe; obj.Index = i; k = k + 1;
                if i == obj.numSteps(), obj.finish(); end
            end
            done = obj.Done;
        end

        function f = progress(obj)
            f = obj.Index / max(1, obj.numSteps());
        end
    end

    methods (Access = private)
        function [x0, eps] = recording(obj, x, t)
            [x0, eps] = obj.Predict(x, t);
            if ~obj.Seen                                   % first call of the step (Heun makes two)
                obj.LastX0 = x0; obj.LastEps = eps; obj.Seen = true;
            end
        end

        function finish(obj)
            if obj.Done, return; end
            if obj.Schedule.sigma(obj.Grid(end)) > 0         % same final denoising as ISampler.sample
                obj.X = obj.Predict(obj.X, obj.Grid(end));
                obj.NFE = obj.NFE + 1;
            end
            obj.Done = true;
        end
    end

    methods (Static)
        function lines = equationText(s)
            % The numbers of Eq. (2.2.4)-(2.2.5) for one step, as text lines. Shared by the
            % Model Lab (live run) and Replay Studio (recording). s: struct with Step,
            % NumSteps, T, TNext, I, INext (NaN if the schedule has no L), AlphaBar,
            % AlphaBarNext, Sigma, Var, Residual.
            head = sprintf('Step %d of %d:  t = %.3f -> %.3f', s.Step, s.NumSteps, s.T, s.TNext);
            if ~isnan(s.I), head = sprintf('%s  (i = %d -> %d)', head, s.I, s.INext); end
            lines = {head, '', ...
                'x_{i-1} ~ N(mu, sigma^2(i) I)                    (Eq. 2.2.5)', ...
                'mu = alphaBar_{i-1} beta_i^2/(1-alphaBar_i^2) x_phi', ...
                '   + (1-alphaBar_{i-1}^2) alpha_i/(1-alphaBar_i^2) x_i   (Eq. 2.2.4)', '', ...
                sprintf('alphaBar_i = %.4f   alphaBar_{i-1} = %.4f', s.AlphaBar, s.AlphaBarNext), ...
                sprintf('sqrt(1 - alphaBar_i^2) = %.4f', s.Sigma), ...
                sprintf('sigma^2(i) = tildeBeta_i^2 = %.3g   (fresh-noise variance)', s.Var), ...
                sprintf('|| x_i - (alphaBar x_phi + sigma eps_phi) || = %.1e', s.Residual)};
        end

        function [mu, v] = posterior(schedule, t_i, t_im1, x_i, x_phi)
            % Reverse conditional of Lemma 2.2.2 / Eq. (2.2.4) with the clean
            % sample replaced by x_phi (general schedule / step jump):
            %   mu = alphaBar_{i-1} beta_i^2 / sigma_i^2 x_phi + sigma_{i-1}^2 alpha_i / sigma_i^2 x_i
            %   v  = sigma_{i-1}^2 / sigma_i^2 beta_i^2   (= tildeBeta_i^2)
            alphaBar_i = schedule.alpha(t_i); alphaBar_im1 = schedule.alpha(t_im1);
            sigmaSq_i = schedule.sigma(t_i) ^ 2; sigmaSq_im1 = schedule.sigma(t_im1) ^ 2;
            if sigmaSq_im1 == 0
                mu = x_phi; v = 0;
                return;
            end
            alpha_i = alphaBar_i / alphaBar_im1;
            betaSq_i = sigmaSq_i - alpha_i ^ 2 * sigmaSq_im1;
            mu = (alphaBar_im1 * betaSq_i / sigmaSq_i) * x_phi + (sigmaSq_im1 * alpha_i / sigmaSq_i) * x_i;
            v = sigmaSq_im1 / sigmaSq_i * betaSq_i;
        end
    end
end
