classdef IRobustPenalty < handle
    % IROBUSTPENALTY  Abstract strategy interface for robust penalty functions.
    %
    %   Defines the penalty family rho(d) for 1D time series, 2D images, and N-D graphs
    %   (Szeliski §4.2 "Robust Regularization"). Any function that replaces the
    %   quadratic norm in the regularization energy
    %
    %       E_reg(f) = sum_k rho( (D * f)_k )
    %
    %   must implement this interface. For a 2D image this is
    %
    %       E_IR = sum_{i,j} s_x * rho(f(i+1,j) - f(i,j))
    %                       + s_y * rho(f(i,j+1) - f(i,j))
    %
    %   Subclasses (Strategy Pattern):
    %       QuadraticPenalty  — rho(d) = d^2            (L2 norm, standard Tikhonov)
    %       L1Penalty         — rho(d) = |d|            (Total Variation / L1 Trend)
    %       PNormPenalty      — rho(d) = |d|^p          (p in (0,2], Hyper-Laplacian for p<1)
    %       HuberPenalty      — rho(d) = piecewise      (smooth L2/L1 compromise)
    %       GNCPenalty        — rho(d) = min(d^2, V)    (Blake-Zisserman non-convex)
    %
    %   IRLS Connection:
    %       The weight() method returns the IRLS weighting function:
    %           w(d) = rho'(d) / d
    %       which converts the non-quadratic problem into a sequence of
    %       weighted least-squares sub-problems (see IRLSSolver). With this
    %       weight, a fixed point of (I + lambda * D' W D) f = g satisfies
    %       f - g + lambda * D' rho'(D f) = 0, the stationarity condition of
    %       (1/2)||f - g||^2 + lambda * sum_k rho((D f)_k).
    %
    %   Package: OceanMath.Regularization
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2, Eq. 4.27-4.30

    methods (Abstract)
        % EVALUATE  Compute the penalty value rho(d).
        %
        %   val = obj.evaluate(d)
        %
        %   d   — (M x 1) or scalar array of differences
        %   val — same size as d, element-wise penalty values
        val = evaluate(obj, d)

        % WEIGHT  Compute the IRLS weight w(d) = rho'(d) / d.
        %
        %   w = obj.weight(d)
        %
        %   d — same as evaluate()
        %   w — element-wise IRLS weights (always finite and non-negative)
        w = weight(obj, d)

        % DERIVATIVE  Compute first derivative rho'(d).
        %
        %   fp = obj.derivative(d)
        %
        %   A subgradient is returned where rho is not differentiable.
        fp = derivative(obj, d)

        % GETNAME  Return a short human-readable label (used in plot legends).
        str = getName(obj)

        % ISCONVEX  Return true if rho is a convex function.
        %
        %   Convex penalties: QuadraticPenalty, L1Penalty, HuberPenalty, PNorm(p>=1)
        %   Non-convex:       GNCPenalty, PNorm(p<1)
        %
        %   Used by GNCSolver to gate continuation annealing.
        tf = isConvex(obj)
    end
end
