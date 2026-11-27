classdef ReverseProcessPanel < OceanViz.App.ModelLab.Panels.PanelBase
    % REVERSEPROCESSPANEL  Noise -> image (Lai et al. Sec. 2.2.2-2.2.6).
    %   Run the reverse process from pure noise x_L ~ N(0, I) -- or from the
    %   current image noised to level t0 (a "round trip") -- then scrub
    %   through the recorded steps. For step i the views show
    %     x_i              the sampler's current sample (INPUT to the model)
    %     eps_phi(x_i, i)  the model's noise estimate (OUTPUT)
    %     x_phi(x_i, i)    the model's clean-image estimate (OUTPUT)
    %     mu               the mean of p_phi(x_{i-1} | x_i), Eq. (2.2.4)
    %   and the numbers of Eq. (2.2.4)-(2.2.5). Long runs (DDPM: L steps) run
    %   in chunks with a progress bar and Cancel (JobRunner).

    properties (SetAccess = private)
        StartDropdown
        T0Field
        SamplerDropdown
        ClassDropdown = []
        GuidanceField = []
        SeedField
        ClipBox
        FrameSlider
        Views = struct()
        StripView
        Info
        Process = []            % OceanDL.Generative.Inspect.ReverseProcess of the last run
    end

    methods
        function t = title(~), t = "Reverse process"; end
        function c = requires(~), c = "diffusion"; end
        function r = reference(~), r = "Lai et al. Sec. 2.2.2-2.2.6, Eq. (2.2.4)-(2.2.5); DDIM Eq. (2.2.15); guidance Sec. 8.3"; end

        function build(obj, parent, ctx)
            obj.layout(parent, ctx, 3, 3);
            obj.Main.RowHeight = {'1x', '1x', '0.8x'};
            m = ctx.Model; s = obj.Side;
            s.heading('Reverse process (learned decoder)');
            obj.StartDropdown = s.dropdown('Start from:', {'Pure noise  x_L ~ N(0, I)', 'Current image noised to t0'});
            obj.T0Field = s.number('t0 for the round trip (0..1):', 0.6, [0.01 1]);
            obj.SamplerDropdown = s.dropdown('Sampler:', obj.samplerItems());
            if m.has("conditional")
                obj.ClassDropdown = s.dropdown('Class (what to generate):', cellstr(obj.classItems()));
                if ~m.isIdeal(), obj.GuidanceField = s.number('Guidance w (Sec. 8.3; 1 = none):', 1, [0 20]); end
            end
            obj.SeedField = s.number('Seed:', 1, [0 Inf]);
            obj.ClipBox = s.checkbox('Clip x_phi to the data range (Ho et al.)', true);
            s.button('Run reverse process', @(~, ~) obj.run(), true);
            s.button('Save recording...', @(~, ~) obj.saveRecordingDialog());
            obj.FrameSlider = s.slider('Recorded step (scrub):', [1 2], 1, @(~, ~) obj.showFrame());
            obj.FrameSlider.ValueChangingFcn = @(~, e) obj.showFrame(e.Value);
            obj.Views.X = obj.view(1, 1, 'x_i (input)');
            obj.Views.Eps = obj.view(1, 2, 'eps_\phi(x_i, i) (output)');
            obj.Views.X0 = obj.view(1, 3, 'x_\phi(x_i, i) (output)');
            obj.Views.Mu = obj.view(2, 1, '\mu: mean of the next step');
            obj.Views.Final = obj.view(2, 2, 'result x_0');
            obj.Info = obj.textBox(2, 3);
            obj.StripView = obj.view(3, [1 3], 'trajectory: x_i (top) and x_\phi (bottom), left = start');
        end

        function file = saveRecording(obj, file, opts)
            % Record the run these settings define (same model, sampler, seed, start and
            % clipping as "Run reverse process") into a Replay file, then (optionally) open
            % it in Replay Studio. Numbers at every step; images at the same stride as the
            % frames this tab keeps; the final x_0.
            arguments
                obj
                file (1,1) string
                opts.Open (1,1) logical = true
                opts.Overwrite (1,1) logical = true
            end
            m = obj.Ctx.Model;
            [smp, predict, stream, x1, t0] = obj.setup();             % x_phi clipping is inside predict
            name = m.Name; if strlength(name) == 0, name = "Model Lab model"; end
            d = OceanDL.Generative.Inspect.DiffusionRecordable(predict, m.Schedule, smp, m.ImageSize, ...
                'Stream', stream, 'X1', x1, 'TStart', t0, 'DataRange', m.DataRange, 'ModelName', name);
            S = OceanReplay.Core.Sampling;
            stride = max(1, round(smp.Steps / 100));
            spec = OceanReplay.Core.Spec() ...
                .addGroup("numbers", 0, ["tLevel" "tNext" "alphaBar" "alphaBarNext" "sigma" "varNext" "epsRms" "identityResidual"]) ...
                .addGroup("images", 1, ["x" "eps" "x0hat" "mu"], struct('step', S.everyIndex(stride))) ...
                .addGroup("final", 1, "result");
            obj.Ctx.status(sprintf('Recording %d steps to %s ...', smp.Steps, file)); drawnow;
            OceanReplay.Record.Recorder.run(d, spec, file, 'Overwrite', opts.Overwrite, ...
                'Title', name + " -- reverse process (Model Lab)", 'Seeds', struct('sampler', obj.SeedField.Value));
            obj.Ctx.status("Recording saved: " + file);
            if opts.Open, OceanViz.App.Replay.ReplayStudio(file); end
        end

        function run(obj)
            m = obj.Ctx.Model;
            [smp, predict, stream, x1, t0, clip] = obj.setup();
            stride = max(1, round(smp.Steps / 100));
            rp = OceanDL.Generative.Inspect.ReverseProcess(predict, m.Schedule, smp, [m.ImageSize 1], ...
                'X1', x1, 'TStart', t0, 'Stream', stream, 'Stride', stride, 'ClipX0', clip);
            chunk = max(1, round(smp.Steps / 50));
            ok = OceanViz.App.Common.JobRunner.run(obj.Ctx.Figure, 'Reverse process', ...
                @() obj.advance(rp, chunk));
            if ~ok, obj.Ctx.status('Reverse process cancelled.'); return; end
            obj.Process = rp;
            n = numel(rp.Frames);
            obj.FrameSlider.Limits = [1 max(2, n)];
            obj.FrameSlider.Value = 1;
            obj.Views.Final.set(m.toDisplay(rp.X));
            obj.drawStrip();
            obj.showFrame(1);
            obj.Ctx.status(sprintf('Reverse process done: %d steps, %d network evaluations, %d frames recorded.', ...
                rp.numSteps(), rp.NFE, n));
        end

        function [smp, predict, stream, x1, t0, clip] = setup(obj)
            % The run the current settings define (shared by run and saveRecording).
            m = obj.Ctx.Model;
            smp = obj.sampler();
            labels = obj.labels(1);
            w = 1; if ~isempty(obj.GuidanceField), w = obj.GuidanceField.Value; end
            predict = m.predictor(labels, w);
            stream = RandStream('mt19937ar', 'Seed', obj.SeedField.Value);
            x1 = []; t0 = 1;
            if startsWith(obj.StartDropdown.Value, 'Current')
                t0 = obj.T0Field.Value;
                if startsWith(string(obj.SamplerDropdown.Value), "DDPM ancestral, all") && ...
                        isa(m.Schedule, 'OceanDL.Generative.Diffusion.Schedules.LinearVP')
                    % Textbook chain: noise to an INTEGER step k = round(t0 L), then run exactly k
                    % reverse steps k -> k-1 -> ... -> 0 (not L steps squeezed into [0, t0]).
                    k = max(1, round(t0 * m.Schedule.L));
                    t0 = k / m.Schedule.L;
                    smp = OceanDL.Generative.Diffusion.Samplers.DDPMAncestral('Steps', k);
                end
                x0 = m.toModel(obj.Ctx.Image);
                e = randn(stream, size(x0));
                x1 = OceanDL.Generative.Inspect.ForwardProcess.at(x0, m.Schedule, t0, e);
            end
            clip = []; if obj.ClipBox.Value, clip = m.DataRange; end
            if ~isempty(clip)
                predict = OceanDL.Generative.Diffusion.Samplers.ISampler.clipped(predict, m.Schedule, clip);
            end
            clip = [];                                                % clipping now lives inside predict
        end

        function saveRecordingDialog(obj)
            [f, p] = uiputfile({'*.h5', 'Replay recording (*.h5)'}, 'Save recording', 'reverse_process.h5');
            if isequal(f, 0), return; end
            obj.saveRecording(string(fullfile(p, f)));
        end

        function showFrame(obj, value)
            if isempty(obj.Process), return; end
            if nargin < 2, value = obj.FrameSlider.Value; end
            k = min(numel(obj.Process.Frames), max(1, round(value)));
            f = obj.Process.Frames(k);
            m = obj.Ctx.Model;
            obj.Views.X.set(m.toDisplay(f.X));
            obj.Views.Eps.set(double(gather(f.Eps)), "noise");
            obj.Views.X0.set(m.toDisplay(f.X0));
            obj.Views.Mu.set(m.toDisplay(f.Mu));
            L = m.numLevels();
            a = m.Schedule.alpha(f.T); sg = m.Schedule.sigma(f.T);
            obj.Info.Value = OceanDL.Generative.Inspect.ReverseProcess.equationText(struct( ...
                'Step', f.Step, 'NumSteps', obj.Process.numSteps(), 'T', f.T, 'TNext', f.TNext, ...
                'I', round(f.T * L), 'INext', round(f.TNext * L), 'AlphaBar', a, ...
                'AlphaBarNext', m.Schedule.alpha(f.TNext), 'Sigma', sg, 'Var', f.Var, ...
                'Residual', norm(double(gather(f.X(:) - (a * f.X0(:) + sg * f.Eps(:)))))));
        end
    end

    methods (Access = private)
        function [done, fraction, msg] = advance(~, rp, chunk)
            done = rp.advance(chunk);
            fraction = rp.progress();
            msg = sprintf('step %d of %d', rp.Index, rp.numSteps());
        end

        function items = samplerItems(obj)
            m = obj.Ctx.Model;
            items = {};
            if isa(m.Schedule, 'OceanDL.Generative.Diffusion.Schedules.LinearVP')
                items{end+1} = sprintf('DDPM ancestral, all %d steps (Sec. 2.2.6)', m.Schedule.L);
            end
            items = [items, {'DDPM ancestral, 100 steps', 'DDIM deterministic, 50 steps (Sec. 9.2)', ...
                'DDIM eta = 1, 50 steps (Eq. 2.2.15)', 'Heun ODE, 25 steps'}];
        end

        function smp = sampler(obj)
            S = 'OceanDL.Generative.Diffusion.Samplers.';
            v = string(obj.SamplerDropdown.Value);
            if startsWith(v, "DDPM ancestral, all")
                smp = obj.Ctx.Model.defaultSampler();
            elseif startsWith(v, "DDPM ancestral, 100")
                smp = feval([S 'DDPMAncestral'], 'Steps', 100);
            elseif startsWith(v, "DDIM deterministic")
                smp = feval([S 'DDIM'], 'Steps', 50);
            elseif startsWith(v, "DDIM eta")
                smp = feval([S 'DDIM'], 'Steps', 50, 'Eta', 1);
            else
                smp = feval([S 'HeunODE'], 'Steps', 25);
            end
        end

        function y = labels(obj, n)
            y = [];
            if ~isempty(obj.ClassDropdown)
                k = obj.selectedLabel(obj.ClassDropdown);
                if k > 0, y = repmat(k, n, 1); end
            end
        end

        function drawStrip(obj)
            m = obj.Ctx.Model; F = obj.Process.Frames;
            pick = unique(round(linspace(1, numel(F), min(12, numel(F)))));
            imgs = zeros([m.ImageSize 2 * numel(pick)]);
            for j = 1:numel(pick)
                imgs(:, :, :, j) = m.toDisplay(F(pick(j)).X);
                imgs(:, :, :, numel(pick) + j) = m.toDisplay(F(pick(j)).X0);
            end
            obj.StripView.set(OceanViz.App.Common.ImageView.tile(imgs, numel(pick)));
        end
    end
end
