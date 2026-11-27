classdef GeneratePanel < OceanViz.App.ModelLab.Panels.PanelBase
    % GENERATEPANEL  "Generate new sample": draw fresh images from the loaded
    %   model -- each click uses a new seed. Diffusion: choose the class (for
    %   conditional models) and guidance w (Lai et al. Sec. 8.3) and the
    %   sampler; VAE: choose the prior temperature. Works for any model with
    %   the "sample" capability.

    properties (SetAccess = private)
        CountField
        ClassDropdown = []
        GuidanceField = []
        SamplerDropdown = []
        TemperatureField = []
        SeedField
        GridView
        Last = []               % last generated images (display range)
    end

    methods
        function t = title(~), t = "Generate"; end
        function c = requires(~), c = "sample"; end
        function r = reference(~), r = "Lai et al. Sec. 2.2.6 (DDPM sampling), Sec. 8.3 (classifier-free guidance); VAE Sec. 2.1"; end

        function build(obj, parent, ctx)
            obj.layout(parent, ctx, 1, 1);
            m = ctx.Model; s = obj.Side;
            s.heading('Generate new samples');
            obj.CountField = s.number('How many:', 16, [1 64]);
            if m.has("diffusion")
                obj.SamplerDropdown = s.dropdown('Sampler:', {'DDIM, 50 steps (fast)', 'DDPM ancestral, all L steps (textbook)'});
                if m.has("conditional")
                    obj.ClassDropdown = s.dropdown('Class:', cellstr(obj.classItems()));
                    if ~m.isIdeal(), obj.GuidanceField = s.number('Guidance w (1 = none):', 1, [0 20]); end
                end
            else
                obj.TemperatureField = s.number('Temperature (1 = the prior):', 1, [0 3]);
                s.label(['Temperature scales the prior''s spread, z ~ N(0, T^2 I). NOT in Kingma & Welling Ch. 2 ' ...
                         '(which samples z ~ p(z), i.e. T = 1); from later work (e.g. Glow, NVAE).']);
            end
            obj.SeedField = s.number('Seed (auto-increments):', 1, [0 Inf]);
            s.button('Generate new sample', @(~, ~) obj.generate(), true);
            s.button('Save image...', @(~, ~) obj.saveImage());
            obj.GridView = obj.view(1, 1, 'samples');
        end

        function imgs = generate(obj)
            m = obj.Ctx.Model;
            n = obj.CountField.Value;
            stream = RandStream('mt19937ar', 'Seed', obj.SeedField.Value);
            if m.has("diffusion")
                if startsWith(obj.SamplerDropdown.Value, 'DDPM'), smp = m.defaultSampler();
                else, smp = OceanDL.Generative.Diffusion.Samplers.DDIM('Steps', 50); end
                y = [];
                if ~isempty(obj.ClassDropdown)
                    k = obj.selectedLabel(obj.ClassDropdown);
                    if k > 0, y = repmat(k, n, 1); end
                end
                w = 1; if ~isempty(obj.GuidanceField), w = obj.GuidanceField.Value; end
                rp = OceanDL.Generative.Inspect.ReverseProcess(m.predictor(y, w), m.Schedule, smp, [m.ImageSize n], ...
                    'Stream', stream, 'Stride', smp.Steps, 'ClipX0', m.DataRange);
                ok = OceanViz.App.Common.JobRunner.run(obj.Ctx.Figure, 'Generating', @() obj.advance(rp, max(1, round(smp.Steps / 50))));
                if ~ok, imgs = []; obj.Ctx.status('Generation cancelled.'); return; end
                imgs = m.toDisplay(rp.X);
            else
                imgs = m.sample(n, obj.TemperatureField.Value, stream);
            end
            obj.Last = imgs;
            obj.GridView.set(OceanViz.App.Common.ImageView.tile(imgs, ceil(sqrt(n))));
            obj.GridView.setTitle(sprintf('%d samples, seed %d', n, obj.SeedField.Value));
            obj.SeedField.Value = obj.SeedField.Value + 1;               % next click: new samples
        end

        function saveImage(obj)
            if isempty(obj.Last), return; end
            [f, p] = uiputfile({'*.png'}, 'Save samples', 'samples.png');
            if isequal(f, 0), return; end
            imwrite(OceanViz.App.Common.ImageView.tile(obj.Last, ceil(sqrt(size(obj.Last, 4)))), fullfile(p, f));
        end
    end

    methods (Access = private)
        function [done, fraction, msg] = advance(~, rp, chunk)
            done = rp.advance(chunk);
            fraction = rp.progress();
            msg = sprintf('step %d of %d', rp.Index, rp.numSteps());
        end
    end
end
