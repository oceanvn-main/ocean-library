classdef DiffusionLayout < OceanViz.App.Replay.Layouts.IReplayStudioLayout
    % DIFFUSIONLAYOUT  A reverse-process recording (DiffusionRecordable):
    %   x_i | eps_phi | x_phi | mu | result (+ views), and the numbers of
    %   Eq. (2.2.4)-(2.2.5) READ FROM THE FILE for the current step -- the same
    %   text as the Model Lab's Reverse process tab
    %   (OceanDL.Generative.Inspect.ReverseProcess.equationText).

    properties (Constant)
        Needed = ["x" "eps" "x0hat" "mu" "alphaBar" "alphaBarNext" "sigma" "varNext" "tLevel" "tNext"]
    end

    methods
        function n = name(~), n = "Diffusion"; end

        function n = timeline(~, ~), n = "epsRms"; end

        function tf = appliesTo(obj, rec)
            tf = all(ismember(obj.Needed, rec.names())) && isfield(rec.Meta, 'model');
        end

        function tiles = tiles(~, rec)
            tiles = [OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Current sample (network input)", "x"), OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Predicted noise (output)", "eps"), ...
                     OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Predicted clean image (output)", "x0hat"), OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Mean of the next step", "mu")];
            if any(rec.names() == "result"), tiles(end+1) = OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile("Final result", "result"); end
            for v = rec.Views
                t = v.Name; if v.MethodId == "occlusion", t = "Occlusion: where the predicted noise depends on the input"; end
                tiles(end+1) = OceanViz.App.Replay.Layouts.IReplayStudioLayout.tile(t, v.Name); %#ok<AGROW>
            end
        end

        function lines = notes(~, rec, coords)
            g = @(n) double(rec.get(n, struct('step', coords.step)));
            [~, at] = rec.get("alphaBar", struct('step', coords.step));
            steps = rec.axis("step").Values;
            k = find(steps == at.step, 1);
            L = NaN; if isfield(rec.Meta.model, 'L'), L = rec.Meta.model.L; end
            s = struct('Step', k, 'NumSteps', numel(steps), 'T', g("tLevel"), 'TNext', g("tNext"), ...
                'I', round(g("tLevel") * L), 'INext', round(g("tNext") * L), 'AlphaBar', g("alphaBar"), ...
                'AlphaBarNext', g("alphaBarNext"), 'Sigma', g("sigma"), 'Var', g("varNext"), 'Residual', NaN);
            if any(rec.names() == "identityResidual"), s.Residual = g("identityResidual"); end
            lines = [{sprintf('REVERSE PROCESS -- Lai et al. Sec. 2.2  (numbers read from the recording, step = %g)', at.step), ''}, ...
                OceanDL.Generative.Inspect.ReverseProcess.equationText(s)];
        end
    end
end
