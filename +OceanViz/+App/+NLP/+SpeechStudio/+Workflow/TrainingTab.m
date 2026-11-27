classdef TrainingTab < OceanViz.App.NLP.SpeechStudio.Workflow.IWorkflowTab
    % TRAININGTAB  Training curves from a checkpoint's saved trainer History
    %   (ModelCatalog.history): per-step loss and learning rate, per-epoch train / validation
    %   loss and validation WER / CER, with the selected (best) epoch marked.

    properties (SetAccess = private)
        ModelDropDown = []
        Session = []
        Viz = []
        Axes struct = struct()
        History = []
        EpochTable = []
    end

    methods
        function t = title(~)
            t = "Training curves";
        end

        function build(obj, parent, session, viz)
            obj.Session = session;
            obj.Viz = viz;
            g = uigridlayout(parent, [4 2], 'RowHeight', {'fit', '1x', '1x', 'fit'});
            uilabel(g, 'Text', 'Checkpoint:');
            names = string(session.models().Name);
            obj.ModelDropDown = uidropdown(g, 'Items', cellstr(names), ...
                'ValueChangedFcn', @(d, ~) obj.show(string(d.Value)));
            slots = {"StepLoss", 2, 1; "Epochs", 2, 2; "LR", 3, 1};
            for k = 1:size(slots, 1)
                s = uigridlayout(g, [1 1], 'Padding', [0 0 0 0]);
                s.Layout.Row = slots{k, 2}; s.Layout.Column = slots{k, 3};
                obj.Axes.(slots{k, 1}) = s;
            end
            obj.EpochTable = uitable(g, 'RowName', {}, 'ColumnEditable', false);
            obj.EpochTable.Layout.Row = 3; obj.EpochTable.Layout.Column = 2;
            obj.SourceLabel = uilabel(g, 'Text', '', 'WordWrap', 'on', 'FontAngle', 'italic');
            obj.SourceLabel.Layout.Row = 4; obj.SourceLabel.Layout.Column = [1 2];
            pick = names(1);
            if session.ModelName ~= "" && any(names == session.ModelName), pick = session.ModelName; end
            obj.show(pick);
        end

        function show(obj, name)
            c = obj.Session.catalog();
            h = c.history(name);
            obj.History = h;
            obj.ModelDropDown.Value = char(name);
            T = c.list();
            file = string(T.File(string(T.Name) == name));
            E = h.Epochs;
            steps = 1:numel(h.StepLoss);
            ax = obj.fresh("StepLoss");
            obj.Viz.plot(ax, "SpeechContour", struct('X', steps, 'Series', struct('Name', "Training loss per step", ...
                'Y', h.StepLoss), 'XLabel', "Optimizer step", 'YLabel', "Loss (nats)", ...
                'Title', sprintf("%s: %d steps", name, numel(steps))));
            ax = obj.fresh("LR");
            obj.Viz.plot(ax, "SpeechContour", struct('X', steps, 'Series', struct('Name', "Learning rate", 'Y', h.StepLR), ...
                'XLabel', "Optimizer step", 'YLabel', "Learning rate", 'Title', "Schedule (warmup + cosine)"));
            ax = obj.fresh("Epochs");
            if isempty(E)
                title(ax, "No per-epoch history in this checkpoint", 'Interpreter', 'none');
            else
                ep = [E.Epoch];
                ser = struct('Name', {"Train loss (epoch mean)", "Validation loss", "Validation WER", "Validation CER"}, ...
                    'Y', {[E.TrainLoss], [E.ValLoss], [E.ValWER], [E.ValCER]}, 'Marker', {"o", "o", "s", "s"});
                best = NaN;
                if isfield(T, 'BestEpoch') || ismember('BestEpoch', T.Properties.VariableNames)
                    best = T.BestEpoch(string(T.Name) == name);
                end
                d = struct('X', ep, 'Series', ser, 'XLabel', "Epoch", 'YLabel', "Loss (nats) / error rate", ...
                    'Title', sprintf("Per epoch (best epoch %g by validation WER)", best));
                if isfinite(best), d.Intervals = [best - 0.5, best + 0.5]; d.IntervalName = "Selected checkpoint epoch"; end
                obj.Viz.plot(ax, "SpeechContour", d);
                obj.EpochTable.Data = struct2table(rmfield(E, intersect(fieldnames(E), {'Steps'})));
            end
            obj.SourceLabel.Text = char("Source: TrainerState.History of " + file + ...
                " (OceanNLP.Speech.Inference.ModelCatalog.history; SpeechTrainer records every step and epoch).");
        end
    end

    methods (Access = private)
        function ax = fresh(obj, name)
            s = obj.Axes.(name);
            delete(s.Children);
            ax = uiaxes(s);
        end
    end
end
