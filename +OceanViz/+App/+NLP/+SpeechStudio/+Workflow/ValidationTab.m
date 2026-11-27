classdef ValidationTab < OceanViz.App.NLP.SpeechStudio.Workflow.IWorkflowTab
    % VALIDATIONTAB  Validation & tests: the external-validation reports (features vs
    %   librosa / MATLAB, pitch threshold, WER/CER vs jiwer) shown table by table, and the
    %   speech test classes present in the repository.

    properties (SetAccess = private)
        Viewer = []
        TestList = []
    end

    methods
        function t = title(~)
            t = "Validation & tests";
        end

        function build(obj, parent, session, ~)
            g = uigridlayout(parent, [2 2], 'RowHeight', {'1x', 'fit'}, 'ColumnWidth', {'3x', '1x'});
            d = dir(fullfile(session.Root, 'Results', 'Speech', '*validation*.md'));
            files = string(fullfile({d.folder}, {d.name}));
            p = uipanel(g, 'BorderType', 'none');
            p.Layout.Row = 1; p.Layout.Column = 1;
            obj.Viewer = OceanViz.App.NLP.SpeechStudio.Workflow.ReportTableViewer(p, files);
            t = dir(fullfile(session.Root, '+OceanTest', '**', '*Test.m'));
            folders = string({t.folder});
            keep = contains(folders, "Speech", 'IgnoreCase', true);
            t = t(keep);
            rel = strings(numel(t), 1);
            for k = 1:numel(t)
                f = string(fullfile(t(k).folder, t(k).name));
                rel(k) = replace(extractAfter(f, strlength(session.Root) + 1), "\", "/");
            end
            obj.TestList = uitable(g, 'Data', table(rel, 'VariableNames', "SpeechTestClass"), ...
                'RowName', {}, 'ColumnEditable', false);
            obj.TestList.Layout.Row = 1; obj.TestList.Layout.Column = 2;
            obj.SourceLabel = uilabel(g, 'Text', char(sprintf("Source: Results/Speech/*validation*.md (%d generated reports) and %d speech test classes under +OceanTest (run with runtests).", ...
                numel(files), numel(t))), 'WordWrap', 'on', 'FontAngle', 'italic');
            obj.SourceLabel.Layout.Row = 2; obj.SourceLabel.Layout.Column = [1 2];
        end
    end
end
