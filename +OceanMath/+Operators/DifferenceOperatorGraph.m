classdef DifferenceOperatorGraph < OceanMath.Operators.IDifferenceOperator
    % DIFFERENCEOPERATORGRAPH  Graph incidence difference operator for network nodes.
    %
    %   Builds sparse Graph Incidence Matrix D (E x V) for network nodes:
    %     (D * f)_k = f(v_k) - f(u_k)  for edge k = (u_k, v_k)
    %
    %   Package: OceanMath.Operators
    %   Author: Ocean Library  |  2026-08-05

    properties
        Edges (:, 2) double  % (E x 2) matrix of [src, tgt] node indices
        NumNodes (1,1) double
    end

    methods
        function obj = DifferenceOperatorGraph(edges, options)
            arguments
                edges (:, 2) double
                options.NumNodes (1,1) double = 0
            end
            obj.Edges = edges;
            if options.NumNodes > 0
                obj.NumNodes = options.NumNodes;
            else
                obj.NumNodes = max(edges(:));
            end
        end

        function D = buildMatrix(obj, shape)
            % BUILDMATRIX Build sparse Graph Incidence Matrix D (E x V).
            if nargin >= 2 && ~isempty(shape)
                V = shape(1);
            else
                V = obj.NumNodes;
            end
            
            E = size(obj.Edges, 1);
            edge_idx = (1:E)';
            src_nodes = obj.Edges(:, 1);
            tgt_nodes = obj.Edges(:, 2);

            rows = [edge_idx; edge_idx];
            cols = [src_nodes; tgt_nodes];
            vals = [-ones(E, 1); ones(E, 1)];

            D = sparse(rows, cols, vals, E, V);
        end

        function d = apply(obj, f, shape)
            if nargin < 3
                shape = numel(f);
            end
            D = obj.buildMatrix(shape);
            d = D * f(:);
        end

        function str = getName(~)
            str = "Graph Incidence Difference";
        end
    end
end
