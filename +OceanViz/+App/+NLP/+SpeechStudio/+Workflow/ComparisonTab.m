classdef ComparisonTab < OceanViz.App.NLP.SpeechStudio.Workflow.IWorkflowTab
    % COMPARISONTAB  Model comparison: the ModelCatalog table (type, parameters, selected
    %   epoch, validation WER read from each checkpoint) with a bar chart of validation WER,
    %   and the held-out comparison tables of the generated experiment reports.

    properties (SetAccess = private)
        Table = []
        Viewer = []
        Catalog table = table()
        Groups = {}
    end

    methods
        function t = title(~)
            t = "Model comparison";
        end

        function build(obj, parent, session, viz)
            g = uigridlayout(parent, [3 2], 'RowHeight', {'1x', 'fit', '1x'}, 'ColumnWidth', {'1x', '1x'});
            T = session.models();
            obj.Catalog = T;
            keep = intersect(["Name", "Type", "ModelId", "Units", "SampleRate", "NumParams", "BestEpoch", "BestValWER"], ...
                string(T.Properties.VariableNames), 'stable');
            obj.Table = uitable(g, 'Data', T(:, keep), 'RowName', {}, 'ColumnEditable', false);
            obj.Table.Layout.Row = 1; obj.Table.Layout.Column = 1;
            % Validation WERs come from DIFFERENT validation sets: the canonical FSDD split
            % (takes 45-49 of the training speakers) for the supervised models, and each
            % HuBERT run's own fine-tuning validation subset -> two separate axes.
            cg = uigridlayout(g, [1 2], 'Padding', [0 0 0 0], 'ColumnWidth', {'1x', '2x'});
            cg.Layout.Row = 1; cg.Layout.Column = 2;
            isH = startsWith(string(T.Name), "hubert");
            obj.Groups = {T(~isH, :), T(isH, :)};
            titles = ["FSDD canonical validation", "HuBERT fine-tuning (own validation subsets)"];
            for k = 1:2
                Tk = obj.Groups{k};
                ax = uiaxes(cg);
                if isempty(Tk), title(ax, titles(k) + ": none"); continue; end
                viz.plot(ax, "BarGroup", struct('X', 1:height(Tk), 'Y', Tk.BestValWER(:), ...
                    'Names', "Validation WER (selected epoch)"), 'Title', titles(k), 'XLabel', "", ...
                    'YLabel', "Validation WER (selected epoch)", 'ShowLegend', false);
                ax.Title.Interpreter = 'none';
                ax.XTick = 1:height(Tk);
                ax.XTickLabel = cellstr(string(Tk.Name));
                ax.TickLabelInterpreter = 'none';
                ax.XTickLabelRotation = 45;
                ax.YLim = [0, max(1e-3, max(Tk.BestValWER)) * 1.15];
            end
            obj.SourceLabel = uilabel(g, 'Text', char("Source: OceanNLP.Speech.Inference.ModelCatalog('Root', """ + ...
                session.CheckpointDir + """).list() -- validation WER = TrainerState.BestScore(1) of each checkpoint " + ...
                "(supervised models: canonical FSDD validation split; HuBERT runs: their own fine-tuning validation subsets, " + ...
                "so the two charts are not comparable with each other). Report tables below are " + ...
                "copied verbatim from the generated reports."), 'WordWrap', 'on', 'FontAngle', 'italic');
            obj.SourceLabel.Layout.Row = 2; obj.SourceLabel.Layout.Column = [1 2];
            d = dir(fullfile(session.Root, 'Results', 'Speech', '*report*.md'));
            files = string(fullfile({d.folder}, {d.name}));
            p = uipanel(g, 'BorderType', 'none');
            p.Layout.Row = 3; p.Layout.Column = [1 2];
            obj.Viewer = OceanViz.App.NLP.SpeechStudio.Workflow.ReportTableViewer(p, files);
        end
    end
end
