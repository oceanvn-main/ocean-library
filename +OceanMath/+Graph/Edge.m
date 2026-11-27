classdef Edge
    % EDGE  Immutable value object representing a graph edge.
    %
    % A pure data container: source, target, weight, and an optional label.
    % No domain-specific logic (splines, curves, etc. belong in wrappers).
    %
    % Properties:
    %   Source  - Integer node ID of the source
    %   Target  - Integer node ID of the target
    %   Weight  - Scalar cost/distance
    %   Label   - Human-readable label (e.g. 'North-South Line')
    %
    % Usage:
    %   e = OceanMath.Graph.Edge(1, 2, 5.3);
    %   e = OceanMath.Graph.Edge(1, 2, 3, 'Label', 'NSL');
    %
    % See also: OceanMath.Graph.Node, OceanMath.Graph.DigraphCus

    properties (SetAccess = immutable)
        Source
        Target
        Weight
        Label
    end

    methods
        function obj = Edge(source, target, weight, options)
            arguments
                source  (1,1) double {mustBePositive, mustBeInteger}
                target  (1,1) double {mustBePositive, mustBeInteger}
                weight  (1,1) double {mustBeNonnegative} = 0
                options.Label  (1,:) char = ''
            end
            obj.Source = source;
            obj.Target = target;
            obj.Weight = weight;
            obj.Label  = options.Label;
        end

        function disp(obj)
            if ~isempty(obj.Label)
                lbl = [' "', obj.Label, '"'];
            else
                lbl = '';
            end
            fprintf('  Edge(%d -> %d) w=%.4f%s\n', ...
                obj.Source, obj.Target, obj.Weight, lbl);
        end
    end
end
