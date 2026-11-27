classdef RejectionSampler < OceanMath.Random.Samplers.ISampler
    % REJECTIONSAMPLER Generates samples using the Accept-Reject method.
    %
    % This sampler draws from a target distribution f(x) by sampling from an
    % easier proposal distribution g(x), and accepting samples with probability
    % proportional to f(x) / (M * g(x)).
    %
    % By default, if no proposal distribution is provided, it uses a Uniform
    % proposal over the finite support of the target distribution, and
    % empirically estimates the bounding constant M.
    %
    % Usage:
    %   % Auto-proposal mode
    %   sampler = RejectionSampler();
    %   X = dist.sample(1000, sampler);
    %
    %   % Custom proposal mode
    %   sampler = RejectionSampler('ProposalDist', g_dist, ...
    %                              'ProposalSampler', g_sampler, ...
    %                              'M', 2.5);
    %
    % See also: OceanMath.Random.Samplers.ISampler
    
    properties
        ProposalSampler 
        ProposalDist    
        M               (1,1) double = NaN
        AutoM           (1,1) logical = true
    end
    
    methods
        function obj = RejectionSampler(options)
            arguments
                options.ProposalSampler = []
                options.ProposalDist = []
                options.M = []
            end
            if ~isempty(options.ProposalSampler)
                obj.ProposalSampler = options.ProposalSampler;
            end
            if ~isempty(options.ProposalDist)
                obj.ProposalDist = options.ProposalDist;
            end
            if ~isempty(options.M)
                obj.M = options.M;
                obj.AutoM = false;
            end
        end
        
        function samples = sample(obj, dist, N)
            arguments
                obj
                dist OceanMath.Information.Distributions.IDistribution
                N (1,1) double {mustBePositive, mustBeInteger}
            end
            
            % 1. Setup default uniform proposal if none provided
            if isempty(obj.ProposalDist)
                supp = dist.getSupport();
                lowerBound = supp(1);
                upperBound = supp(2);
                
                if isinf(lowerBound) || isinf(upperBound)
                    error('OceanMath:Random:Samplers:InfiniteSupport', ...
                        'Cannot auto-generate a uniform proposal for infinite support.');
                end
                
                pdfFunc = @(x) ones(size(x)) / (upperBound - lowerBound);
                obj.ProposalDist = OceanMath.Information.Distributions.ContinuousDistribution.fromPDF(pdfFunc, supp, 'Validate', false);
                obj.ProposalSampler = OceanMath.Random.Samplers.InverseTransformSampler();
            end
            
            % 2. Estimate bounding constant M if not provided
            if obj.AutoM || isnan(obj.M)
                supp = dist.getSupport();
                testPts = linspace(supp(1), supp(2), 1000);
                targetPdf = dist.evaluate(testPts);
                propPdf = obj.ProposalDist.evaluate(testPts);
                
                % Add 20% margin to the maximum observed ratio
                obj.M = max(targetPdf ./ propPdf) * 1.2;
                if obj.M == 0 || isnan(obj.M)
                    error('OceanMath:Random:Samplers:InvalidM', ...
                        'Failed to estimate bounding constant M. Check PDF evaluations.');
                end
            end
            
            % 3. Generation Loop
            samples = zeros(N, 1);
            accepted = 0;
            
            while accepted < N
                % Vectorized draw chunks (draw slightly more to reduce loop iterations)
                chunkSize = N - accepted;
                
                % Sample Y ~ g(x)
                Y = obj.ProposalSampler.sample(obj.ProposalDist, chunkSize);
                
                % Sample U ~ Uniform(0, 1)
                U = rand(chunkSize, 1);
                
                % Evaluate target and proposal densities
                targetPdf = dist.evaluate(Y);
                propPdf = obj.ProposalDist.evaluate(Y);
                
                % Acceptance condition: U <= f(Y) / (M * g(Y))
                % Force column vectors to prevent matrix broadcasting errors
                targetPdf = targetPdf(:);
                propPdf = propPdf(:);
                acceptIdx = U(:) <= (targetPdf ./ (obj.M * propPdf));
                
                numAccepted = sum(acceptIdx);
                
                if numAccepted > 0
                    % Only take up to the number we still need
                    needed = min(numAccepted, N - accepted);
                    
                    % Extract accepted values
                    acceptedVals = Y(acceptIdx);
                    
                    samples(accepted+1 : accepted+needed) = acceptedVals(1:needed);
                    accepted = accepted + needed;
                end
            end
        end
    end
end
