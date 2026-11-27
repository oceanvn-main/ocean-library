classdef IRobustPenalty < handle
    % IROBUSTPENALTY  Abstract strategy interface for robust penalty functions.
    %
    %   Defines the penalty family rho(d) for 1D time series, 2D images, and N-D graphs.
    %
    %       E_reg(f) = sum_k rho( (D * f)_k )
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
    %           w(d) = rho(d) / d^2
    %       which converts the non-quadratic problem into a sequence of
    %       weighted least-squares sub-problems.
    %
    %   Package: OceanMath.Regularization
    %
    %   Author: Ocean Library  |  2026-08-05
    %   Reference: Szeliski (2021) §4.2, Eq. 4.27-4.30

    methods (Abstract)
        % EVALUATE  Compute the penalty value rho(d).
        val = evaluate(obj, d)

        % WEIGHT  Compute the IRLS weight w(d) = rho(d) / d^2.
        w = weight(obj, d)

        % DERIVATIVE  Compute first derivative rho'(d).
        fp = derivative(obj, d)

        % GETNAME  Return human-readable strategy name.
        str = getName(obj)

        % ISCONVEX  Return true if penalty is convex.
        tf = isConvex(obj)
    end
end
