classdef (Abstract) ISampler < handle
    % ISAMPLER Abstract interface for probability sampling strategies.
    %
    % This interface decouples the mathematical definition of a probability
    % distribution (IDistribution) from the algorithms used to generate
    % random variates from it (Inverse Transform, Rejection, Latent Variable).
    %
    % Design Principle (Strategy Pattern):
    %   A distribution only defines the mathematical properties (PDF, PMF, Support).
    %   The ISampler dictates the specific algorithm to realize samples from it.
    %   This ensures the code remains open to new sampling algorithms (e.g. MCMC)
    %   without modifying the underlying distribution classes.
    %
    % See also: OceanMath.Information.Distributions.IDistribution

    methods (Abstract)
        % SAMPLE Generate random variates from the given distribution.
        %
        % INPUTS:
        %   dist - An instance of IDistribution.
        %   N    - Number of samples to generate (integer > 0).
        %
        % OUTPUTS:
        %   samples - Generated samples. For univariate, an (N x 1) vector.
        %             For multivariate, an (N x p) matrix.
        samples = sample(obj, dist, N)
    end
end
