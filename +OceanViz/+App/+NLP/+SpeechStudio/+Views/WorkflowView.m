classdef WorkflowView < handle
    % WORKFLOWVIEW  The secondary "Workflow" content: a tab group (Corpus & split, Training
    %   curves, Model comparison, Validation & tests). Tabs are built lazily on first show.

    properties (SetAccess = private)
        TabGroup
        Tabs = {}
        TabObjects = {}
    end

    properties (Access = private)
        Session
        Viz
        OnError
    end

    methods
        function obj = WorkflowView(parent, session, viz, onError)
            arguments
                parent
                session
                viz
                onError = @(err) []
            end
            obj.Session = session;
            obj.Viz = viz;
            obj.OnError = onError;
            W = 'OceanViz.App.NLP.SpeechStudio.Workflow.';
            obj.TabObjects = {feval([W 'CorpusTab']), feval([W 'TrainingTab']), ...
                feval([W 'ComparisonTab']), feval([W 'ValidationTab'])};
            obj.TabGroup = uitabgroup(parent, 'SelectionChangedFcn', @(~, ~) obj.ensureSelected());
            for k = 1:numel(obj.TabObjects)
                obj.Tabs{k} = uitab(obj.TabGroup, 'Title', char(obj.TabObjects{k}.title()));
            end
        end

        function ensureSelected(obj)
            k = find(cellfun(@(t) t == obj.TabGroup.SelectedTab, obj.Tabs), 1);
            obj.buildTab(k);
        end

        function buildTab(obj, k)
            t = obj.TabObjects{k};
            if t.Built, return; end
            try
                t.ensureBuilt(obj.Tabs{k}, obj.Session, obj.Viz);
            catch err
                uilabel(uigridlayout(obj.Tabs{k}, [1 1]), 'Text', char("Could not build this tab: " + err.message), ...
                    'WordWrap', 'on');
                obj.OnError(err);
            end
        end

        function selectTab(obj, k)
            obj.TabGroup.SelectedTab = obj.Tabs{k};
            obj.buildTab(k);
        end
    end
end
