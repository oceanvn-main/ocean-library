classdef TrainingStepPanel < OceanViz.App.ModelLab.Panels.PanelBase
    % TRAININGSTEPPANEL  One L_simple training example opened up (Lai et al.
    %   Eq. (2.2.10)): pick a step i and a noise draw, form
    %     x_i = alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps           (Eq. 2.2.1)
    %   feed (x_i, i) to the model and compare its eps_phi with the true eps:
    %     loss_i = || eps - eps_phi(x_i, i) ||^2 / D                  (Eq. 2.2.10)
    %   The error map shows WHERE the model is wrong. Training minimises the
    %   average of this loss over images, steps and noise draws.

    properties (SetAccess = private)
        StepSlider
        SeedField
        ClassDropdown = []
        Views = struct()
        Info
    end

    methods
        function t = title(~), t = "Training step"; end
        function c = requires(~), c = "diffusion"; end
        function r = reference(~), r = "Lai et al. Eq. (2.2.10): L_simple = E || eps - eps_phi(alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps, i) ||^2"; end

        function build(obj, parent, ctx)
            obj.layout(parent, ctx, 2, 3);
            m = ctx.Model; s = obj.Side;
            L = m.numLevels();
            s.heading('One training example (L_simple)');
            s.label('x_0 = the current image.');
            obj.StepSlider = s.slider(sprintf('Step i (1..%d):', L), [1 L], round(L / 2), @(~, ~) obj.update());
            obj.SeedField = s.number('Noise seed:', 1, [0 Inf]);
            obj.SeedField.ValueChangedFcn = @(~, ~) obj.update();
            if m.has("conditional")
                obj.ClassDropdown = s.dropdown('Class given to the model:', cellstr(obj.classItems()));
                obj.ClassDropdown.ValueChangedFcn = @(~, ~) obj.update();
            end
            s.button('Random i and noise', @(~, ~) obj.randomize(), true);
            obj.Views.X0 = obj.view(1, 1, 'x_0 (data)');
            obj.Views.Eps = obj.view(1, 2, '\epsilon (true noise)');
            obj.Views.X = obj.view(1, 3, 'x_i (model input)');
            obj.Views.EpsHat = obj.view(2, 1, '\epsilon_\phi(x_i, i) (model output)');
            obj.Views.Err = obj.view(2, 2, '|\epsilon - \epsilon_\phi| (error map)');
            obj.Info = obj.textBox(2, 3);
            obj.update();
        end

        function onImageChanged(obj, ~)
            obj.update();
        end

        function loss = update(obj)
            m = obj.Ctx.Model;
            L = m.numLevels();
            i = round(obj.StepSlider.Value);
            x0 = m.toModel(obj.Ctx.Image);
            e = randn(RandStream('mt19937ar', 'Seed', obj.SeedField.Value), size(x0));
            [xi, info] = OceanDL.Generative.Inspect.ForwardProcess.at(x0, m.Schedule, i / L, e);
            y = [];
            if ~isempty(obj.ClassDropdown)
                k = obj.selectedLabel(obj.ClassDropdown);
                if k > 0, y = k; end
            end
            predict = m.predictor(y, 1);
            [~, epsHat] = predict(xi, i / L);
            epsHat = double(gather(epsHat));
            err = abs(e - epsHat);
            loss = mean((e - epsHat) .^ 2, 'all');
            obj.Views.X0.set(m.toDisplay(x0));
            obj.Views.Eps.set(e, "noise");
            obj.Views.X.set(m.toDisplay(xi));
            obj.Views.EpsHat.set(epsHat, "noise");
            obj.Views.Err.set(err / 2);
            obj.Info.Value = {sprintf('Step i = %d  (t = %.3f)', i, i / L), '', ...
                'x_i = alphaBar_i x_0 + sqrt(1 - alphaBar_i^2) eps   (Eq. 2.2.1)', ...
                sprintf('alphaBar_i = %.4f, sqrt(1 - alphaBar_i^2) = %.4f', info.AlphaBar, info.Sigma), '', ...
                'loss = mean (eps - eps_phi(x_i, i))^2           (Eq. 2.2.10)', ...
                sprintf('     = %.4f', loss), '', ...
                'Reference: predicting eps = 0 gives loss ~ 1.', ...
                'Typically: low i (little noise) -> eps is hard to', ...
                'separate from the image -> larger loss; high i ->', ...
                'x_i is mostly eps -> smaller loss.'};
        end

        function randomize(obj)
            obj.StepSlider.Value = randi(obj.Ctx.Model.numLevels());
            obj.SeedField.Value = randi(1e6);
            obj.update();
        end
    end
end
