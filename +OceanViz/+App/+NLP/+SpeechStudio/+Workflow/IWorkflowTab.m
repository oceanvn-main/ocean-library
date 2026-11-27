classdef (Abstract) IWorkflowTab < handle
    % IWORKFLOWTAB  One tab of the Workflow view (system-level: corpus, training, comparison,
    %   validation). build() is called lazily the first time the tab is shown; every number
    %   comes from the manifest, the checkpoints (ModelCatalog) or machine-generated reports,
    %   and the tab names its source.

    properties (SetAccess = protected)
        Built (1,1) logical = false
        SourceLabel = []
    end

    methods (Abstract)
        t = title(obj)
        build(obj, parent, session, viz)
    end

    methods
        function ensureBuilt(obj, parent, session, viz)
            if obj.Built, return; end
            obj.build(parent, session, viz);
            obj.Built = true;
        end
    end
end
