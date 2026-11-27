classdef IDistribution < handle
    % IDISTRIBUTION  Abstract base interface for all probability entities.
    %
    % This is the foundational contract for the Information Theoretic
    % architecture. ALL probability objects (discrete, continuous, empirical,
    % parametric) must implement this interface.
    %
    % Design Principle:
    %   Any metric strategy (Shannon Entropy, KL Divergence, etc.) only
    %   ever interacts with an IDistribution — it never cares whether the
    %   distribution is a histogram, a Gaussian PDF, or a kernel estimator.
    %   This is the Dependency Inversion Principle in action.
    %
    % Required Implementations:
    %   p    = evaluate(obj, x)           - Query probability at state(s) x
    %   tf   = isValid(obj)               - Assert the distribution is legal
    %   n    = getSupportSize(obj)         - Number of distinct states (Inf for continuous)
    %   supp = getSupport(obj)             - Return the support domain
    %
    % Optional Override:
    %   h    = entropy(obj)               - Compute own entropy (in bits)
    %
    % See also: OceanMath.Information.Distributions.DiscreteDistribution
    %           OceanMath.Information.Distributions.ContinuousDistribution

    methods (Abstract)
        % EVALUATE  Query the probability (or density) at state(s) x.
        %
        % For a DiscreteDistribution, this returns P(X = x), the PMF value.
        % For a ContinuousDistribution, this returns f(x), the PDF value.
        %
        % INPUTS:
        %   x - scalar, vector, or set of query states/points
        %
        % OUTPUTS:
        %   p - probability (discrete) or density (continuous) values,
        %       same shape as x. Values are always >= 0.
        p = evaluate(obj, x)

        % ISVALID  Assert the distribution is mathematically well-formed.
        %
        % A discrete distribution is valid if: all(p >= 0) and sum(p) == 1.
        % A continuous distribution is valid if: f(x) >= 0 everywhere and
        % the integral equals 1.
        %
        % OUTPUTS:
        %   tf - logical scalar; true if the distribution is valid
        tf = isValid(obj)

        % GETSUPPORTSIZE  Return the cardinality of the support.
        %
        % OUTPUTS:
        %   n - Number of discrete states. Returns Inf for continuous.
        n = getSupportSize(obj)

        % GETSUPPORT  Return the domain of the distribution.
        %
        % OUTPUTS:
        %   supp - For discrete: unique state labels (categorical, numeric, etc.)
        %          For continuous: [lower_bound, upper_bound] interval
        supp = getSupport(obj)
    end

    methods
        % ENTROPY  Compute the Shannon entropy (or differential entropy) of
        % this distribution, in bits. Subclasses may override this for
        % efficiency, but the default dispatches to the metric strategies.
        %
        % OUTPUTS:
        %   h - scalar, entropy in bits (log base 2)
        function h = entropy(obj)
            p = obj.evaluate(obj.getSupport());
            p = p(p > 0);   % Avoid log(0)
            h = -sum(p .* log2(p));
        end

        % SAMPLE Generate random variates from this distribution using a specific sampler.
        %
        % INPUTS:
        %   N       - Number of samples to generate.
        %   sampler - An instance of OceanMath.Random.Samplers.ISampler.
        %
        % OUTPUTS:
        %   samples - Generated random samples.
        function samples = sample(obj, N, sampler)
            arguments
                obj
                N (1,1) double {mustBePositive, mustBeInteger}
                sampler OceanMath.Random.Samplers.ISampler
            end
            samples = sampler.sample(obj, N);
        end
    end
end
