classdef ForwardProcessPanel < OceanViz.App.ModelLab.Panels.PanelBase
    % FORWARDPROCESSPANEL  Image -> noise (Lai et al. Sec. 2.2.1). Drag step i
    %   (0..L) or press Play; every frame is Eq. (2.2.1) evaluated directly,
    %     x_i = alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps,
    %   with the book's numbers at that step (alphaBar_i, beta_i, alpha_i, SNR)
    %   and a marker on the schedule curves. No model is used: the forward
    %   process is fixed. The input is the app's current image.

    properties (SetAccess = private)
        StepSlider
        SeedField
        FreshNoise
        PlayButton
        XView
        StripView
        CurveAxes
        Marker
        Info
        Playing (1,1) logical = false
    end

    methods
        function t = title(~), t = "Forward process"; end
        function c = requires(~), c = "noise-schedule"; end
        function r = reference(~), r = "Lai et al. Sec. 2.2.1, Eq. (2.2.1): x_i = alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps"; end

        function build(obj, parent, ctx)
            obj.layout(parent, ctx, 2, 2);
            obj.Main.RowHeight = {'2x', '1x'}; obj.Main.ColumnWidth = {'1x', '1x'};
            L = ctx.Model.numLevels();
            s = obj.Side;
            s.heading('Forward process (fixed encoder)');
            s.label('Input: the current image (top bar: Dataset / Load / Random image).');
            obj.StepSlider = s.slider(sprintf('Step i (0 = clean, %d = noise):', L), [0 L], round(0.3 * L), @(~, ~) obj.update());
            obj.StepSlider.ValueChangingFcn = @(~, e) obj.update(e.Value);
            obj.SeedField = s.number('Noise seed:', 1, [0 Inf]);
            obj.SeedField.ValueChangedFcn = @(~, ~) obj.refresh();
            obj.FreshNoise = s.checkbox('Fresh noise at every step', false);
            obj.FreshNoise.ValueChangedFcn = @(~, ~) obj.update();
            obj.PlayButton = s.button('Play  (0 -> L)', @(~, ~) obj.togglePlay(), true);
            s.label(['The same eps is used at every i unless "fresh noise" is ticked: then each frame ' ...
                'is an independent draw from p_i(x_i | x_0).']);
            obj.XView = obj.view(1, 1, 'x_i');
            obj.CurveAxes = obj.plotAxes(1, 2);
            obj.StripView = obj.view(2, 1, 'x_i at i = 0, L/10, ..., L (same noise)');
            obj.Info = obj.textBox(2, 2);
            obj.drawCurves();
            obj.refresh();
        end

        function onImageChanged(obj, ~)
            obj.refresh();
        end

        function refresh(obj)
            obj.drawStrip();
            obj.update();
        end

        function [xi, info] = frame(obj, i)
            % x_i (model range) and the schedule numbers at step i.
            m = obj.Ctx.Model;
            L = m.numLevels();
            x0 = m.toModel(obj.Ctx.Image);
            seed = obj.SeedField.Value;
            if obj.FreshNoise.Value, seed = seed * 100003 + i; end
            eps = randn(RandStream('mt19937ar', 'Seed', seed), size(x0));
            [xi, info] = OceanDL.Generative.Inspect.ForwardProcess.at(x0, m.Schedule, i / L, eps);
        end

        function update(obj, value)
            if nargin < 2, value = obj.StepSlider.Value; end
            i = round(value);
            [xi, info] = obj.frame(i);
            m = obj.Ctx.Model;
            obj.XView.set(m.toDisplay(xi));
            obj.XView.setTitle(sprintf('x_i at step i = %d', i));
            obj.Marker.XData = [i i];
            L = m.numLevels();
            lines = {sprintf('Step i = %d of L = %d   (t = i/L = %.3f)', i, L, i / L), '', ...
                'x_i = alphaBar_i * x_0 + sqrt(1 - alphaBar_i^2) * eps     (Eq. 2.2.1)', '', ...
                sprintf('alphaBar_i            = %.5f   (signal kept)', info.AlphaBar), ...
                sprintf('sqrt(1 - alphaBar_i^2) = %.5f   (noise std)', info.Sigma), ...
                sprintf('SNR = alphaBar^2 / (1 - alphaBar^2) = %.4g   (log %.2f)', info.SNR, info.LogSNR)};
            if ~isnan(info.Beta)
                lines = [lines, {'', sprintf('this step: beta_i = %.5f (beta_i^2 = %.6f), alpha_i = sqrt(1 - beta_i^2) = %.6f', ...
                    info.Beta, info.BetaSq, info.AlphaStep)}];
            end
            obj.Info.Value = lines;
        end

        function togglePlay(obj)
            obj.Playing = ~obj.Playing;
            obj.PlayButton.Text = char(ifelse(obj.Playing, "Stop", "Play  (0 -> L)"));
            L = obj.Ctx.Model.numLevels();
            i = 0;
            while obj.Playing && i <= L && isvalid(obj.StepSlider)
                obj.StepSlider.Value = i; obj.update(i);
                drawnow limitrate;
                i = i + max(1, round(L / 100));
            end
            obj.Playing = false;
            if isvalid(obj.PlayButton), obj.PlayButton.Text = 'Play  (0 -> L)'; end
        end
    end

    methods (Access = private)
        function drawStrip(obj)
            m = obj.Ctx.Model;
            L = m.numLevels();
            idx = round(linspace(0, L, 11));
            frames = zeros([m.ImageSize numel(idx)]);
            fresh = obj.FreshNoise.Value; obj.FreshNoise.Value = false;
            for k = 1:numel(idx), frames(:, :, :, k) = m.toDisplay(obj.frame(idx(k))); end
            obj.FreshNoise.Value = fresh;
            obj.StripView.set(OceanViz.App.Common.ImageView.tile(frames, numel(idx)));
        end

        function drawCurves(obj)
            ax = obj.CurveAxes; c = obj.Ctx.Theme;
            m = obj.Ctx.Model; L = m.numLevels();
            T = OceanDL.Generative.Inspect.ForwardProcess.scheduleTable(m.Schedule, L);
            i = (0:L).';
            hold(ax, 'on');
            plot(ax, i, T.AlphaBar, 'LineWidth', 1.8, 'Color', c.Primary);
            plot(ax, i, T.Sigma, 'LineWidth', 1.8, 'Color', c.Secondary);
            obj.Marker = plot(ax, [0 0], [0 1], '--', 'Color', c.Text);
            xlabel(ax, 'step i'); ylim(ax, [0 1]); xlim(ax, [0 L]);
            legend(ax, {'alphaBar_i (signal)', 'sqrt(1 - alphaBar_i^2) (noise)'}, 'TextColor', c.Text, ...
                'Color', c.Panel, 'Location', 'east');
            title(ax, 'Noise schedule');
        end
    end
end

function v = ifelse(c, a, b)
if c, v = a; else, v = b; end
end
