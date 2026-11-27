classdef Node
    % NODE  Immutable value object representing a graph vertex (station).
    %
    % Stores the identity, spatial position, and arbitrary metadata of
    % a single node in a NetworkGraph.
    %
    % Properties:
    %   Id       - Unique integer identifier (1-indexed)
    %   Name     - Human-readable label (e.g. 'Orchard')
    %   Position - [1 x d] spatial coordinate for visualization
    %   Metadata - struct of arbitrary attributes (zone, interchange, color)
    %
    % Usage:
    %   n = OceanMath.Graph.Node(1, 'Orchard', [3.5, 7.2]);
    %   n = OceanMath.Graph.Node(2, 'Dhoby Ghaut', [4, 6], ...
    %           struct('zone', 1, 'interchange', true));
    %
    % See also: OceanMath.Graph.Edge, OceanMath.Graph.DigraphCus

    properties (SetAccess = immutable)
        Id
        Name
        Position
        Metadata
    end

    methods
        function obj = Node(id, name, position, metadata)
            arguments
                id        (1,1) double {mustBePositive, mustBeInteger}
                name      (1,:) char
                position  (1,:) double = [0, 0]
                metadata  (1,1) struct = struct()
            end
            obj.Id       = id;
            obj.Name     = name;
            obj.Position = position;
            obj.Metadata = metadata;
        end

        function disp(obj)
            fprintf('  Node(%d) "%s" at [%s]\n', obj.Id, obj.Name, ...
                strjoin(arrayfun(@(x) sprintf('%.2f', x), obj.Position, ...
                'UniformOutput', false), ', '));
        end
    end
end
