classdef (Abstract) IDynaComponent < handle
    %IDYNACOMPONENT Common explainability contract for Dyna components.

    properties (Abstract, Constant)
        Latex (1,1) string
        Reference (1,1) string
    end

    methods (Abstract)
        s = name(obj)
    end

    methods
        function meta = getMetadata(obj)
            meta = struct( ...
                'SymbolicLatex', obj.Latex, ...
                'Symbols', struct(), ...
                'Assumptions', {obj.assumptions()}, ...
                'IsSymbolic', strlength(obj.Latex) > 0, ...
                'Reference', obj.Reference);
            if ~meta.IsSymbolic
                meta.SymbolicLatex = strings(1, 0);
            end
        end

        function spec = getPipelineSpec(obj)
            spec = struct( ...
                'Inputs', struct('Name', "transition", 'Domain', "RL-Dyna", ...
                    'Dtype', "record", 'Shape', "scalar"), ...
                'Outputs', struct('Name', "result", 'Domain', "RL-Dyna", ...
                    'Dtype', "record", 'Shape', "scalar"), ...
                'Kind', "NONVIRTUAL", ...
                'DiagramSymbol', struct('Type', "default", 'Label', obj.name()), ...
                'ComposesFrom', {{}});
        end
    end

    methods (Access = protected)
        function value = assumptions(~)
            value = {'finite tabular state and action spaces'; 'stationary action legality'};
        end
    end
end
