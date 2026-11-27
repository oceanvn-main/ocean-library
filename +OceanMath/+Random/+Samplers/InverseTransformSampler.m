classdef InverseTransformSampler < OceanMath.Random.Samplers.ISampler
    % INVERSETRANSFORMSAMPLER Generates samples using the Inverse Transform method.
    %
    % The Inverse Transform method samples a target distribution by generating
    % uniform random variates U ~ Uniform(0,1) and passing them through the
    % inverse Cumulative Distribution Function (CDF): X = F^{-1}(U).
    %
    % For DiscreteDistribution: computes the exact CDF via cumulative sum,
    % then uses vectorized search to pick the corresponding state in O(N).
    %
    % For ContinuousDistribution: uses numerical integration (integral) and 
    % root finding (fzero). Note: this is a generic fallback and is 
    % computationally expensive for large N.
    %
    % See also: OceanMath.Random.Samplers.ISampler, OceanMath.Information.Distributions.IDistribution

    methods
        function samples = sample(~, dist, N)
            arguments
                ~
                dist OceanMath.Information.Distributions.IDistribution
                N (1,1) double {mustBePositive, mustBeInteger}
            end
            
            U = rand(N, 1);
            
            if isa(dist, 'OceanMath.Information.Distributions.DiscreteDistribution')
                supp = dist.getSupport();
                p = dist.evaluate(supp);
                
                % Vectorized discrete inverse transform
                cdf = cumsum(p(:)');
                % Guarantee final CDF value is 1 (avoid numerical precision issues)
                cdf(end) = 1.0; 
                
                % For each U, find first index where cdf >= U
                % Using broadcasting: (N x 1) >= (1 x K) -> (N x K) logical matrix
                idx = sum(U >= cdf, 2) + 1;
                
                if isnumeric(supp)
                    samples = supp(idx);
                elseif iscell(supp)
                    samples = supp(idx);
                else
                    samples = supp(idx);
                end
                
            elseif isa(dist, 'OceanMath.Information.Distributions.ContinuousDistribution')
                supp = dist.getSupport();
                lowerBound = supp(1);
                upperBound = supp(2);
                
                if isinf(lowerBound) || isinf(upperBound)
                    error('OceanMath:Random:Samplers:InfiniteSupport', ...
                        'Generic numerical inverse transform requires finite support.');
                end
                
                samples = zeros(N, 1);
                pdfFunc = @(x) dist.evaluate(x);
                
                % fzero requires a sign change. The root is in [lowerBound, upperBound]
                % F(lower) - U = -U (negative). F(upper) - U = 1 - U (positive)
                for i = 1:N
                    targetU = U(i);
                    objFunc = @(x) integral(pdfFunc, lowerBound, x) - targetU;
                    samples(i) = fzero(objFunc, [lowerBound, upperBound]);
                end
            else
                error('OceanMath:Random:Samplers:UnsupportedDistribution', ...
                    'Distribution type %s not supported for generic inverse transform.', class(dist));
            end
        end
    end
end
