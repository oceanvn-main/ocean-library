classdef CorpusTab < OceanViz.App.NLP.SpeechStudio.Workflow.IWorkflowTab
    % CORPUSTAB  Corpus & split: SpeechManifest.audit() of the canonical FSDD split (per
    %   split x speaker counts and durations) as a table and as grouped bars.

    properties (SetAccess = private)
        Table = []
        Audit table = table()
    end

    methods
        function t = title(~)
            t = "Corpus & split";
        end

        function build(obj, parent, session, viz)
            g = uigridlayout(parent, [3 2], 'RowHeight', {'fit', '1x', 'fit'}, 'ColumnWidth', {'1x', '1x'});
            m = session.manifest();
            obj.Audit = m.audit();
            A = obj.Audit;
            h = uilabel(g, 'Text', sprintf('%s: %d utterances, %.1f min; test = held-out speaker "%s", validation = takes %d-%d of the training speakers.', ...
                m.Name, height(m.Entries), sum(m.Entries.Duration) / 60, m.FSDDTestSpeaker, ...
                min(m.FSDDValidationTakes), max(m.FSDDValidationTakes)), 'WordWrap', 'on', 'FontWeight', 'bold');
            h.Layout.Row = 1; h.Layout.Column = [1 2];
            obj.Table = uitable(g, 'Data', A, 'RowName', {}, 'ColumnEditable', false);
            obj.Table.Layout.Row = 2; obj.Table.Layout.Column = 1;
            rows = A(A.Split ~= "ALL", :);
            ax = uiaxes(g);
            ax.Layout.Row = 2; ax.Layout.Column = 2;
            viz.plot(ax, "BarGroup", struct('X', 1:height(rows), ...
                'Y', [rows.TotalDuration, rows.NumUtterances], 'Names', ["Total duration (s)", "Utterances"]), ...
                'Title', "Per split / speaker", 'XLabel', "Split / speaker", 'YLabel', "Seconds / count");
            ax.XTick = 1:height(rows);
            ax.XTickLabel = cellstr(rows.Split + " / " + rows.Speaker);
            ax.TickLabelInterpreter = 'none';
            ax.XTickLabelRotation = 30;
            obj.SourceLabel = uilabel(g, 'Text', char("Source: OceanNLP.Speech.Data.SpeechManifest.fromFSDD(""" + ...
                m.Root + """).canonicalFSDDSplit().audit() -- read from the WAV headers (audioinfo)."), ...
                'WordWrap', 'on', 'FontAngle', 'italic');
            obj.SourceLabel.Layout.Row = 3; obj.SourceLabel.Layout.Column = [1 2];
        end
    end
end
