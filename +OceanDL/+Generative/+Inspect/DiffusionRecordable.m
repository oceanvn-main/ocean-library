classdef DiffusionRecordable < OceanReplay.Record.IRecordable
    % DIFFUSIONRECORDABLE  Records the reverse (denoising) process for replay
    %   (OceanReplay; Planning/System/Replay). Notation: Lai et al., "The
    %   Principles of Diffusion Models", Sec. 2.2.2-2.2.6.
    %
    %   Axis "step": the book's step index i of x_i -- round(t_i L) for a
    %   schedule with L steps (LinearVP: 1000, 999, ..., 1 for DDPM; 1000, 980,
    %   ... for DDIM with 50 steps), otherwise the reverse-step count (numSteps..1).
    %
    %   Channels (at each step)                          Kind
    %     x        x_i, the current sample (network INPUT)   image
    %     eps      eps_phi(x_i, i), predicted noise          image
    %     x0hat    x_phi(x_i, i), predicted clean sample     image
    %     mu       mean of p_phi(x_{i-1} | x_i), Eq. (2.2.4) image
    %     tLevel, tNext, alphaBar, alphaBarNext, sigma, varNext (= sigma^2(i)),
    %     epsRms, identityResidual (|| x_i - (alphaBar x_phi + sigma eps_phi) ||)   scalars
    %     result   the final sample x_0 (no axis)            image
    %   View (forward passes only, works for any predictor):
    %     occlusion  Zeiler & Fergus occlusion applied to the denoiser: how much
    %                eps_phi changes when a patch of x_i is replaced (NOT in Lai et al.)
    %
    %   The run itself is ReverseProcess with 'OnStep'; with a NullRecorder it
    %   gives exactly ReverseProcess's (and the sampler's) result.
    %
    %   Example:
    %     sch = OceanDL.Generative.Diffusion.Schedules.LinearVP();
    %     d = OceanDL.Generative.Inspect.DiffusionRecordable(predict, sch, ...
    %           OceanDL.Generative.Diffusion.Samplers.DDPMAncestral('Steps', 1000), [16 16 1], 'Seed', 7);
    %     OceanReplay.Record.Recorder.run(d, spec, "ddpm.h5");

    properties (SetAccess = private)
        Predict                 % @(x, t) -> [x0, eps] (clipped when ClipX0 is set)
        Schedule
        Sampler
        ImageShape (1,3) double
        Seed (1,1) double
        X1 = []
        TStart (1,1) double
        DataRange (1,2) double
        ModelName (1,1) string
        Like = 0
        Grid (1,:) double       % t of each reverse step (numSteps + 1 levels)
        StepValues (1,:) double % axis "step" value of each reverse step
        StepMeaning (1,1) string
        Stream = []             % optional caller stream (e.g. the Model Lab's), used once by run()
    end

    methods
        function obj = DiffusionRecordable(predict, schedule, sampler, shape, opts)
            arguments
                predict (1,1) function_handle
                schedule OceanDL.Generative.Diffusion.Schedules.INoiseSchedule
                sampler OceanDL.Generative.Diffusion.Samplers.ISampler
                shape (1,:) double
                opts.Seed (1,1) double = 1
                opts.X1 = []
                opts.TStart (1,1) double = 1
                opts.ClipX0 (1,:) double = []
                opts.DataRange (1,2) double = [-1 1]
                opts.ModelName (1,1) string = ""
                opts.Like = 0
                opts.Stream = []            % RandStream to use instead of one seeded with Seed
            end
            if numel(shape) == 4 && shape(4) ~= 1
                error('OceanDL:Generative:Inspect:DiffusionRecordable:batch', 'Record one sample at a time (shape [H W C] or [H W C 1]).');
            end
            shape = [shape(1:min(3, end)) ones(1, 3 - min(3, numel(shape)))];
            if ~isempty(opts.ClipX0)
                predict = OceanDL.Generative.Diffusion.Samplers.ISampler.clipped(predict, schedule, opts.ClipX0);
            end
            obj.Predict = predict; obj.Schedule = schedule; obj.Sampler = sampler;
            obj.ImageShape = shape; obj.Seed = opts.Seed; obj.X1 = opts.X1; obj.TStart = opts.TStart;
            obj.DataRange = opts.DataRange; obj.ModelName = opts.ModelName; obj.Like = opts.Like;
            obj.Stream = opts.Stream;
            obj.Grid = opts.TStart * schedule.grid(sampler.Steps);                 % same grid as ReverseProcess
            n = numel(obj.Grid) - 1;
            if isprop(schedule, 'L')
                v = round(obj.Grid(1:n) * schedule.L);
                if numel(unique(v)) == n
                    obj.StepValues = v;
                    obj.StepMeaning = sprintf("book step index i = round(t_i L), L = %d", schedule.L);
                    return;
                end
            end
            obj.StepValues = n:-1:1;
            obj.StepMeaning = "reverse-step count (numSteps down to 1)";
        end

        function axes = axesInfo(obj)
            axes = OceanReplay.Core.Axis("step", obj.StepValues, 'Symbol', "i", ...
                'Description', "reverse-process step: " + obj.StepMeaning + " (Lai et al. Sec. 2.2)");
        end

        function [c, v] = catalog(obj)
            C = @OceanReplay.Core.ChannelInfo; S = obj.ImageShape; R = obj.DataRange;
            c = [C("x", "image", 'Axes', "step", 'ValueShape', S, 'DisplayRange', R, 'Symbol', "x_i", 'SymbolLatex', '$x_i$', ...
                   'Description', "the sampler's current sample -- the INPUT of the network at step i"), ...
                 C("eps", "image", 'Axes', "step", 'ValueShape', S, 'DisplayRange', [-2 2], 'Symbol', "eps_phi(x_i, i)", 'SymbolLatex', '$\epsilon_\phi(x_i, i)$', ...
                   'Description', "the network's estimate of the noise in x_i (OUTPUT); grey = 0, white/black = +-2"), ...
                 C("x0hat", "image", 'Axes', "step", 'ValueShape', S, 'DisplayRange', R, 'Symbol', "x_phi(x_i, i)", 'SymbolLatex', '$x_\phi(x_i, i)$', ...
                   'Description', "the implied clean-sample estimate (x_i - sigma_i eps_phi) / alphaBar_i (OUTPUT)"), ...
                 C("mu", "image", 'Axes', "step", 'ValueShape', S, 'DisplayRange', R, 'Symbol', "mu", 'SymbolLatex', '$\mu$', ...
                   'Description', "mean of p_phi(x_{i-1} | x_i), Eq. (2.2.4) with x -> x_phi: where the next sample is centred"), ...
                 C("tLevel", "scalar", 'Axes', "step", 'Symbol', "t_i", 'SymbolLatex', '$t_i$', 'Description', "continuous noise level of step i (t = i / L)"), ...
                 C("tNext", "scalar", 'Axes', "step", 'Symbol', "t_{i-1}", 'SymbolLatex', '$t_{i-1}$', 'Description', "noise level after the step"), ...
                 C("alphaBar", "scalar", 'Axes', "step", 'Symbol', "alphaBar_i", 'SymbolLatex', '$\bar{\alpha}_i$', 'Description', "signal scale of x_i (book alphaBar_i)"), ...
                 C("alphaBarNext", "scalar", 'Axes', "step", 'Symbol', "alphaBar_{i-1}", 'SymbolLatex', '$\bar{\alpha}_{i-1}$', 'Description', "signal scale after the step"), ...
                 C("sigma", "scalar", 'Axes', "step", 'Symbol', "sqrt(1 - alphaBar_i^2)", 'SymbolLatex', '$\sqrt{1-\bar{\alpha}_i^2}$', 'Description', "noise scale of x_i"), ...
                 C("varNext", "scalar", 'Axes', "step", 'Symbol', "sigma^2(i)", 'SymbolLatex', '$\sigma^2(i)$', ...
                   'Description', "variance of the fresh noise added by DDPM, tildeBeta_i^2 (Eq. 2.2.5)"), ...
                 C("epsRms", "scalar", 'Axes', "step", 'Symbol', "rms(eps_phi)", 'SymbolLatex', '$\mathrm{rms}(\epsilon_\phi)$', 'Description', "root-mean-square of the predicted noise"), ...
                 C("identityResidual", "scalar", 'Axes', "step", 'Symbol', "||x_i - (alphaBar x_phi + sigma eps_phi)||", 'SymbolLatex', '$\|x_i - (\bar{\alpha}_i x_\phi + \sigma_i \epsilon_\phi)\|$', ...
                   'Description', "numerical check: x_i = alphaBar_i x_phi + sigma_i eps_phi must hold (about 0)"), ...
                 C("result", "image", 'ValueShape', S, 'DisplayRange', R, 'Symbol', "x_0", 'SymbolLatex', '$x_0$', ...
                   'Description', "the final sample (after the last step and the final denoising)")];
            v = OceanReplay.Core.ViewInfo("occlusion", "image", 'Axes', "step", 'ValueShape', [S(1) S(2) 1], ...
                'DisplayRange', [0 1], 'Symbol', "occlusion(eps_phi)", 'SymbolLatex', '$\mathrm{occlusion}(\epsilon_\phi)$', ...
                'Compute', @(s, p, r) obj.occlusion(s, p), ...
                'HowMade', "Occlusion (Zeiler & Fergus): a Size x Size patch of x_i (moved by Stride) is replaced by Fill; " + ...
                    "each pixel gets the average rms change of eps_phi(x_i, i) over the patches covering it; the map is divided " + ...
                    "by its maximum (1 = the most influential region). Forward passes only; no gradients.", ...
                'Reference', "Zeiler & Fergus 2014; Szeliski 2nd ed. Sec. 5.4 -- applied here to a diffusion denoiser", ...
                'InTexts', false);
        end

        function out = run(obj, rec)
            stream = obj.Stream;
            if isempty(stream), stream = RandStream('mt19937ar', 'Seed', obj.Seed); end
            n = numel(obj.Grid) - 1;
            rp = OceanDL.Generative.Inspect.ReverseProcess(obj.Predict, obj.Schedule, obj.Sampler, [obj.ImageShape 1], ...
                'X1', obj.X1, 'TStart', obj.TStart, 'Stream', stream, 'Stride', max(1, n), 'Like', obj.Like, ...
                'OnStep', @(f) obj.onStep(rec, f));
            rp.advance(Inf);
            x = gather(rp.X);
            rec.point(struct(), struct('result', @() x));
            out = struct('X', x, 'NFE', rp.NFE);
        end

        function d = describe(obj)
            m = struct('name', obj.ModelName, 'schedule', string(class(obj.Schedule)), ...
                'sampler', string(class(obj.Sampler)), 'steps', obj.Sampler.Steps, 'tStart', obj.TStart, ...
                'imageShape', obj.ImageShape, 'dataRange', obj.DataRange);
            if isprop(obj.Schedule, 'L'), m.L = obj.Schedule.L; end
            d = struct('model', m, ...
                'references', {{'Lai et al., The Principles of Diffusion Models, Sec. 2.2.2-2.2.6, Eq. (2.2.4)-(2.2.5)'}}, ...
                'notation', struct('alphaBar', "schedule.alpha(t) = alphaBar_i", ...
                                   'sigma', "schedule.sigma(t) = sqrt(1 - alphaBar_i^2)", 'step', obj.StepMeaning), ...
                'layout', "diffusion");
        end
    end

    methods (Access = private)
        function onStep(obj, rec, f)
            k = f.Step; sch = obj.Schedule;
            img = @(v) reshape(gather(double(v)), obj.ImageShape);
            state = struct( ...
                'x', @() img(f.X), 'eps', @() img(f.Eps), 'x0hat', @() img(f.X0), 'mu', @() img(f.Mu), ...
                'tLevel', @() f.T, 'tNext', @() f.TNext, ...
                'alphaBar', @() sch.alpha(f.T), 'alphaBarNext', @() sch.alpha(f.TNext), 'sigma', @() sch.sigma(f.T), ...
                'varNext', @() gather(double(f.Var)), 'epsRms', @() gather(double(rms(f.Eps(:)))), ...
                'identityResidual', @() gather(double(norm(f.X(:) - (sch.alpha(f.T) * f.X0(:) + sch.sigma(f.T) * f.Eps(:))))), ...
                'xRaw', @() f.X, 'tRaw', @() f.T);
            rec.point(struct('step', obj.StepValues(k)), state);
        end

        function map = occlusion(obj, s, p)
            % Occlusion map of eps_phi at the current step (see the catalog text).
            size_ = 4; stride = 2; fill = mean(obj.DataRange);
            if isfield(p, 'size'), size_ = p.size; end
            if isfield(p, 'stride'), stride = p.stride; end
            if isfield(p, 'fill'), fill = p.fill; end
            x = s.xRaw(); t = s.tRaw();
            [~, e0] = obj.Predict(x, t);
            H = obj.ImageShape(1); W = obj.ImageShape(2);
            acc = zeros(H, W); cnt = zeros(H, W);
            rows = unique([1:stride:max(1, H - size_ + 1), max(1, H - size_ + 1)]);
            cols = unique([1:stride:max(1, W - size_ + 1), max(1, W - size_ + 1)]);
            for r = rows
                for c = cols
                    xo = x;
                    ri = r:min(H, r + size_ - 1); ci = c:min(W, c + size_ - 1);
                    xo(ri, ci, :, :) = fill;
                    [~, e1] = obj.Predict(xo, t);
                    d = gather(double(rms(e1(:) - e0(:))));
                    acc(ri, ci) = acc(ri, ci) + d; cnt(ri, ci) = cnt(ri, ci) + 1;
                end
            end
            map = acc ./ max(cnt, 1);
            map = map / max(max(map(:)), eps);
        end
    end
end
