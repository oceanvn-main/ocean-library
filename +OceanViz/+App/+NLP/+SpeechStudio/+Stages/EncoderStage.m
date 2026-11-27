classdef EncoderStage < OceanViz.App.NLP.SpeechStudio.Stages.StageBase
    % ENCODERSTAGE  6 - Encoder: the recorded activations of each encoder block (hidden unit
    %   x output frame), their per-frame L2 norm, and the subsampling frames in -> frames out.

    properties (SetAccess = private)
        LayerDropDown = []
        Layer (1,1) string = ""
    end

    methods
        function t = title(~)
            t = "Encoder";
        end

        function c = requires(~)
            c = strings(1, 0);
        end

        function buildPlots(obj, grid)
            grid.RowHeight = {'fit', '1x'};
            bar = uigridlayout(grid, [1 2], 'ColumnWidth', {'fit', '1x'}, 'Padding', [0 0 0 0]);
            bar.Layout.Row = 1;
            uilabel(bar, 'Text', 'Encoder block:');
            obj.LayerDropDown = uidropdown(bar, 'Items', {'(run first)'}, ...
                'ValueChangedFcn', @(d, ~) obj.selectLayer(d.Value));
            g = obj.addSlots(grid, ["Activations", "Norm"], 2, 1);
            g.Layout.Row = 2;
            g.RowHeight = {'2x', '1x'};
        end

        function [rows, classes] = draw(obj, ctx)
            E = ctx.artifacts("^Encoder");
            if isempty(E), error('No encoder step was recorded.'); end
            names = [E.Name];
            obj.LayerDropDown.Items = cellstr(names);
            if ~any(names == obj.Layer), obj.Layer = names(end); end
            obj.LayerDropDown.Value = char(obj.Layer);
            e = E(names == obj.Layer).Value;
            ax = obj.slotAxes("Activations");
            obj.plot(ax, "SpeechFeatures", e);
            timed = ~isempty(e.FrameTimes);
            if timed, obj.addTimeAxes(ax); end
            nrm = vecnorm(e.Values, 2, 1);
            x = e.FrameTimes; xl = "Time (s), receptive-field centres";
            if ~timed, x = 1:numel(nrm); xl = "Output frame"; end
            ax = obj.slotAxes("Norm");
            obj.plot(ax, "SpeechContour", struct('X', x, 'Series', struct('Name', "||h_t||_2", 'Y', nrm), ...
                'XLabel', xl, 'YLabel', "L2 norm of h_t", 'Title', "Per-frame activation norm: " + obj.Layer));
            if timed, obj.addTimeAxes(ax); end

            F = ctx.artifact("Model features");
            Tin = size(F.Values, 2);
            rows = ["Input features (F x T)", obj.shape(F.Values)];
            for k = 1:numel(E)
                v = E(k).Value;
                rows(end+1, :) = [E(k).Name, obj.shape(v.Values) + " (hidden x frames)"]; %#ok<AGROW>
            end
            Tout = size(E(end).Value.Values, 2);
            rows = [rows; "Frames in -> out", sprintf("%d -> %d (x%.2f subsampling)", Tin, Tout, Tin / max(Tout, 1))];
            m = ctx.Recognizer;
            for k = 1:numel(m.Encoders)
                rows(end+1, :) = [sprintf("Encoders{%d}", k), string(class(m.Encoders{k}))]; %#ok<AGROW>
            end
            classes = ["OceanNLP.Speech.Inspect.SpeechProcessAdapter", string(class(m)), ...
                string(cellfun(@class, m.Encoders, 'UniformOutput', false))];
        end

        function selectLayer(obj, name)
            obj.Layer = string(name);
            if ~isempty(obj.Context), obj.render(obj.Context); end
        end

        function L = notation(~)
            L = ["$[H^{(j)}, T_j] = E_j\big(H^{(j-1)}, T_{j-1}\big),\; H^{(0)} = \tilde{X}$", ...
                "$T_j = \big\lfloor (T_{j-1} + 2p - d(k-1) - 1)/s \big\rfloor + 1$ (conv)", ...
                "$\|h_t\|_2 = \sqrt{\sum_i h_{t,i}^2}$", ...
                "$E_j$: encoder block $j$; $H^{(j)}$: its output ($D \times T_j$)", ...
                "$T_j$: valid frames after block $j$", ...
                "$k, s, p, d$: kernel, stride, padding, dilation"];
        end

        function s = description(~)
            s = ["The encoder turns the normalised features into a sequence of hidden vectors h_t. " + ...
                "Convolutional front layers with stride > 1 subsample time (fewer, longer-context frames), " + ...
                "then recurrent / Conformer blocks mix information across the utterance.", "", ...
                "Each image is the real activation matrix recorded by SpeechProcessAdapter while replaying the " + ...
                "model's forward pass (the test suite asserts the replay equals recognizer.forward).", "", ...
                "Frame times on the x axis are receptive-field centres mapped back through the convolutions; " + ...
                "when that mapping is unknown the x axis is the output-frame index."];
        end
    end
end
