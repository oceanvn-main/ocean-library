classdef VariationalEnergy < handle
    % VARIATIONALENERGY  Domain-agnostic variational regularization energy.
    %
    %   Part of the Unified Regularization Architecture.
    %   Computes the generalized regularization energy for ANY 1D signal, 2D image,
    %   or N-D graph, combining a linear IDifferenceOperator and an IRobustPenalty:
    %
    %       E(f) = (1/2) * ||f - g||^2   +   lambda * sum_k rho( (D * f)_k )
    %                  data fidelity                   regularizer
    %
    %   Representation of D (Sparse Linear Difference Matrix):
    %     - D is a sparse matrix of dimension (K x N), where:
    %         N = total number of elements in state f(:)
    %             (N = length(f) for 1D, N = rows*cols for 2D, N = V for graphs)
    %         K = total number of directional differences / graph edges
    %             (K = N-1 for 1D Order-1, K = N-2 for 1D Order-2,
    %              K = rows*(cols-1) + (rows-1)*cols for 2D images,
    %              K = E for graph edges)
    %
    %   What are the numbers in D?
    %     - Every row k of D contains EXACTLY TWO non-zero numbers: -1 and +1.
    %     - Row k computes: (-1)*f(u) + (+1)*f(v) = f(v) - f(u) for neighbor pair (u,v).
    %     - 99.999% of entries in D are 0 (stored efficiently as MATLAB sparse matrix).
    %
    %   Visual 2x2 Image Example (N=4 pixels, K=4 differences):
    %     Given 2x2 image F = [f11 f12; f21 f22], flattened f = [f11; f21; f12; f22]:
    %
    %        Col 1(f11)  Col 2(f21)  Col 3(f12)  Col 4(f22)
    %       ┌                                               ┐  ┌     ┐    ┌             ┐
    %  Row 1│   -1          0          +1          0        │  │ f11 │    │  f12 - f11  │  (H-1)
    %  Row 2│    0         -1           0         +1        │  │ f21 │    │  f22 - f21  │  (H-2)
    %  Row 3│   -1         +1           0          0        │  │ f12 │ =  │  f21 - f11  │  (V-1)
    %  Row 4│    0          0          -1         +1        │  │ f22 │    │  f22 - f12  │  (V-2)
    %       └                                               ┘  └     ┘    └             ┘
    %
    %   What happens when we multiply d = D * f(:)?
    %     - Matrix-vector product d = D * f(:) maps the flattened state field f(:)
    %       to a column vector d of length K containing discrete directional differences:
    %
    %       • 1D Signal (Order 1):
    %           d_i = f(i+1) - f(i)   --> 1st-order temporal difference
    %
    %       • 1D Signal (Order 2):
    %           d_i = f(i+2) - 2*f(i+1) + f(i)   --> 2nd-order curvature
    %
    %       • 2D Image Grid:
    %           d = [ d_horizontal ; d_vertical ]
    %           where d_h = f(r, c+1) - f(r, c)   (horizontal spatial gradient)
    %                 d_v = f(r+1, c) - f(r, c)   (vertical spatial gradient)
    %
    %       • Graph ML Network (nodes V, edges E):
    %           d_k = f(v_k) - f(u_k)   for edge k = (u_k, v_k)
    %
    %     - Evaluating sum_k rho(d_k) computes the total penalty across all
    %       directional differences according to the injected IRobustPenalty strategy.
    %
    %   IRLS Normal Equations (solved by IRLSSolver):
    %       (I + lambda * D' * W * D) f_{k+1} = g
    %
    %   where W = diag(w(D*f)) and w(d) = rho'(d)/d.
    %
    %   Package: OceanMath.Variational
    %   Author: Ocean Library  |  2026-08-05

    properties
        Operator OceanMath.Operators.IDifferenceOperator = OceanMath.Operators.DifferenceOperator2D()
        Penalty  OceanMath.Regularization.IRobustPenalty  = OceanMath.Regularization.QuadraticPenalty()
        Lambda   (1,1) double {mustBeNonnegative} = 0.1
    end

    methods
        function obj = VariationalEnergy(operator, penalty, options)
            % VARIATIONALENERGY Constructor.
            arguments
                operator OceanMath.Operators.IDifferenceOperator = OceanMath.Operators.DifferenceOperator2D()
                penalty  OceanMath.Regularization.IRobustPenalty  = OceanMath.Regularization.QuadraticPenalty()
                options.Lambda (1,1) double {mustBeNonnegative} = 0.1
            end
            obj.Operator = operator;
            obj.Penalty  = penalty;
            obj.Lambda   = options.Lambda;
        end

        function E = totalEnergy(obj, f, g)
            % TOTALENERGY Compute E(f) = 0.5 * ||f - g||^2 + lambda * E_reg(f).
            E_data = 0.5 * sum((f(:) - g(:)) .^ 2);
            E_reg  = obj.regularizationEnergy(f);
            E      = E_data + obj.Lambda * E_reg;
        end

        function E_reg = regularizationEnergy(obj, f)
            % REGULARIZATIONENERGY Compute E_reg(f) = sum_k rho( (D * f)_k ).
            %   1. d = D * f(:) applies matrix D to compute all directional differences.
            %   2. evaluate(d) applies robust penalty rho(d_k) element-wise.
            shape = size(f);
            d = obj.Operator.apply(f, shape);
            E_reg = sum(obj.Penalty.evaluate(d));
        end

        function D = buildDifferenceMatrix(obj, varargin)
            % BUILDDIFFERENCEMATRIX Retrieve sparse difference operator matrix D (K x N).
            %   D = energy.buildDifferenceMatrix(shape)        % e.g. [50, 50] or [100]
            %   D = energy.buildDifferenceMatrix(rows, cols)   % e.g. 50, 50
            if nargin == 2
                shape = varargin{1};
            elseif nargin >= 3
                shape = [varargin{1}, varargin{2}];
            else
                error('VariationalEnergy:MissingShape', 'Shape argument required.');
            end
            if isscalar(shape)
                shape = [shape, 1];
            end
            D = obj.Operator.buildMatrix(shape);
        end

        function W_diag = buildWeightMatrix(obj, f)
            % BUILDWEIGHTMATRIX Build sparse diagonal IRLS weight matrix W (K x K).
            %   w(d_k) = rho'(d_k) / d_k is computed element-wise on differences d = D*f.
            shape = size(f);
            D = obj.buildDifferenceMatrix(shape);
            dv = D * f(:);
            w = obj.Penalty.weight(dv);
            W_diag = spdiags(w, 0, numel(dv), numel(dv));
        end
    end
end
