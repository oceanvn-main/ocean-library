classdef DifferenceOperator1D < OceanMath.Operators.IDifferenceOperator
    % DIFFERENCEOPERATOR1D  1D difference operator for time series and signals.
    %
    %   Order = 1: First-difference operator D_1 * f = f_{i+1} - f_i (N-1 x N)
    %   Order = 2: Second-difference operator D_2 * f = f_{i+2} - 2*f_{i+1} + f_i (N-2 x N)
    %
    %   Package: OceanMath.Operators
    %   Author: Ocean Library  |  2026-08-05

    properties
        Order (1,1) double {mustBeMember(Order, [1, 2])} = 1
    end

    methods
        function obj = DifferenceOperator1D(options)
            arguments
                options.Order (1,1) double = 1
            end
            obj.Order = options.Order;
        end

        function D = buildMatrix(obj, shape)
            % BUILDMATRIX Build sparse 1D difference operator D.
            %   shape — scalar N (length of 1D signal)
            N = shape(1);

            if obj.Order == 1
                % (N-1 x N) first difference
                Nh = N - 1;
                edge_h = (1:Nh)';
                src_h = (1:Nh)';
                tgt_h = (2:N)';
                D = sparse([edge_h; edge_h], [src_h; tgt_h], [-ones(Nh,1); ones(Nh,1)], Nh, N);
            else
                % (N-2 x N) second difference
                Nh = N - 2;
                edge_h = (1:Nh)';
                i1 = (1:Nh)';
                i2 = (2:Nh+1)';
                i3 = (3:N)';
                rows = [edge_h; edge_h; edge_h];
                cols = [i1; i2; i3];
                vals = [ones(Nh,1); -2*ones(Nh,1); ones(Nh,1)];
                D = sparse(rows, cols, vals, Nh, N);
            end
        end

        function d = apply(obj, f, shape)
            if nargin < 3
                shape = numel(f);
            end
            D = obj.buildMatrix(shape);
            d = D * f(:);
        end

        function str = getName(obj)
            str = sprintf('1D Difference (Order %d)', obj.Order);
        end
    end
end
