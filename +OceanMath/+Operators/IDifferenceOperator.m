classdef (Abstract) IDifferenceOperator < handle
    % IDIFFERENCEOPERATOR  Abstract strategy interface for linear difference matrices.
    %
    %   Part of the Domain-Agnostic Regularization Architecture.
    %   Encapsulates the construction of sparse linear difference operator matrix D
    %   for 1D signals, 2D images, and N-D graphs.
    %
    %   Package: OceanMath.Operators
    %   Author: Ocean Library  |  2026-08-05

    methods (Abstract)
        % BUILDMATRIX Construct sparse difference operator matrix D for state dimensions.
        %
        %   D = obj.buildMatrix(shape)
        %   shape — [N] for 1D, [H, W] for 2D, [V] for graph
        D = buildMatrix(obj, shape)

        % APPLY Apply difference operator D * f(:).
        d = apply(obj, f, shape)

        % GETNAME Human-readable operator description.
        str = getName(obj)
    end
end
