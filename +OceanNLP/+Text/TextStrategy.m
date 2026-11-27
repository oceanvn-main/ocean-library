classdef (Abstract) TextStrategy < handle
    % TEXTSTRATEGY  Common base of every swappable text-processing strategy
    % (char filters, token filters, stemmers, pre-tokenizers). It supplies the
    % 3-pillar contract (.agents/pipeline-architecture-spec.md) with the
    % honest-opacity default used by every string-processing operator in this
    % library (R7): these operators are discrete procedures, not closed-form
    % maps, so IsSymbolic=false and no LaTeX is fabricated.
    %
    % A concrete strategy only has to declare:
    %   Reference  (Constant) citation / description of the algorithm
    %   InDomain / OutDomain (Constant) pipeline port domains
    %   name()     short identifier
    % and implement its interface's forward()/stem().
    %
    % EXTENSION RULE: a new algorithm is a NEW FILE in the matching package;
    % factories find it by class-name convention, so no existing source is edited.
    %
    % See also: OceanNLP.Text.StrategyResolver, OceanNLP.Text.Factory

    properties (Abstract, Constant)
        Reference (1,1) string
        InDomain (1,1) string
        OutDomain (1,1) string
    end

    methods (Abstract)
        s = name(obj)
    end

    methods
        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = obj.assumptions();
            meta.IsSymbolic = false;
            meta.Reference = obj.Reference;
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = struct('Name', "in", 'Domain', obj.InDomain, 'Dtype', "string", 'Shape', "varies");
            spec.Outputs = struct('Name', "out", 'Domain', obj.OutDomain, 'Dtype', "string", 'Shape', "varies");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', obj.name());
            spec.ComposesFrom = {};
        end
    end

    methods (Access = protected)
        function a = assumptions(~)
            % Override to list behavioural assumptions; default is a generic honesty note.
            a = {'discrete string procedure: no closed-form formula is claimed (R7)'};
        end
    end
end
