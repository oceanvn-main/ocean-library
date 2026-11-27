classdef (Abstract) ITDComponent < handle
    % ITDCOMPONENT  Common base of every swappable TD component (targets, step sizes, exploration
    % rules, eligibility traces). Supplies the 3-pillar contract (.agents/pipeline-architecture-spec.md):
    %   Pillar 1 (symbolic): getMetadata() with a LaTeX formula when the rule has a closed form.
    %   Pillar 3 (diagram):  getPipelineSpec().
    % A concrete component declares the constants below and name(); nothing else is required.
    %
    % EXTENSION RULE: a new target / step size / exploration / trace is ONE new file in the matching
    % package; OceanStats.RL.TD.Factory finds it by class-name convention, no existing file is edited.

    properties (Abstract, Constant)
        Latex (1,1) string          % closed-form rule, "" if none is honest
        Reference (1,1) string      % citation / description
    end

    methods (Abstract)
        s = name(obj)
    end

    methods
        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = obj.Latex;
            if strlength(obj.Latex) == 0, meta.SymbolicLatex = strings(1, 0); end
            meta.Symbols = containers.Map();
            meta.Assumptions = obj.assumptions();
            meta.IsSymbolic = strlength(obj.Latex) > 0;
            meta.Reference = obj.Reference;
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "transition", 'Domain', "RL-Transition", 'Dtype', "double", 'Shape', "record");
            spec.Outputs = struct('Name', "value", 'Domain', "RL-Value", 'Dtype', "double", 'Shape', "scalar");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', obj.name());
            spec.ComposesFrom = {};
        end
    end

    methods (Access = protected)
        function a = assumptions(~)
            a = {'tabular; one scalar per (state, action)'};
        end
    end
end
