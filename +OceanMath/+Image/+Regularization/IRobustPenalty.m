classdef IRobustPenalty < handle
    % IROBUSTPENALTY  Abstract strategy interface for robust penalty functions.
    %
    %   Defines the rho(x) family from Szeliski §4.2 "Robust Regularization".
    %   Any function that replaces the quadratic norm in the regularization energy
    %
    %       E_IR = sum_{i,j} s_x * rho(f(i+1,j) - f(i,j))
    %                       + s_y * rho(f(i,j+1) - f(i,j))
    %
    %   must implement this interface.
    %
    %   Subclasses (Strategy Pattern):
    %       QuadraticPenalty  — rho(x) = x^2            (p=2 norm, standard L2)
    %       L1Penalty         — rho(x) = |x|             (Total Variation, p=1)
    %       PNormPenalty      — rho(x) = |x|^p           (hyper-Laplacian for p<1)
    %       HuberPenalty      — rho(x) = piecewise       (smooth L2/L1 blend)
    %       GNCPenalty        — rho(x) = min(x^2, V)     (Blake-Zisserman 1987)
    %
    %   IRLS Connection:
    %       The weight() method returns the IRLS weighting function
    %           w(d) = rho(d) / d^2
    %       which converts the non-quadratic problem into a sequence of
    %       weighted least-squares sub-problems (see IRLSSolver).
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

        % WEIGHT  Compute the IRLS weight w(d) = rho(d) / d^2.
        %
        %   w = obj.weight(d)
        %
        %   The default implementation in the base class handles d→0 via
        %   epsilon stabilisation.  Subclasses MAY override for efficiency
        %   or to exploit a closed-form limit.
        %
        %   d — same as evaluate()
        %   w — element-wise IRLS weights (always finite and non-negative)
        w = weight(obj, d)

        % DERIVATIVE  Compute rho'(d), the first derivative.
        %
        %   g = obj.derivative(d)
        %
        %   Used by the Primal-Dual solver (sub-gradient for non-smooth penalties).
        g = derivative(obj, d)

        % GETNAME  Return a short human-readable label (used in plot legends).
        name = getName(obj)

        % ISCONVEX  Return true if rho is a convex function.
        %
        %   Convex penalties: QuadraticPenalty, L1Penalty, HuberPenalty, PNorm(p>=1)
        %   Non-convex:       GNCPenalty, PNorm(p<1)
        %
        %   Used by GNCSolver to gate continuation annealing.
        tf = isConvex(obj)
    end
end
