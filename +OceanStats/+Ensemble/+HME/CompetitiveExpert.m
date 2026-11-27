classdef CompetitiveExpert < OceanStats.Ensemble.HME.IExpert
    % COMPETITIVEEXPERT A "Survival of the Fittest" expert strategy.
    % This class manages a pool of diverse expert candidates (Gaussian, Bernoulli, etc.)
    % and allows the EM algorithm to perform "Automated Hiring" by favoring the 
    % expert type that best fits the local region.
    
    properties (Access = private)
        Experts      % Cell array of IExpert candidates
        K            % Number of candidates
        Prior        % [1 x K] base weights (used if Gating is disabled)
        PoolGamma    % [D x K] weights for localized hiring (Solution A)
        Link         % SoftmaxLink for hiring gate
        
        % Hyperparameters
        Alpha = 1.0  % Sparsity / Dirichlet Factor (Solution C)
        UseLocalGating = false % Toggle for Solution A
        IsInitialized = false
    end
    
    methods
        function obj = CompetitiveExpert(candidates, options)
            arguments
                candidates
                options.Alpha = 1.0
                options.UseLocalGating = false
            end
            obj.Experts = candidates;
            obj.K = length(candidates);
            obj.Prior = ones(1, obj.K) / obj.K; 
            obj.Alpha = options.Alpha;
            obj.UseLocalGating = options.UseLocalGating;
            obj.Link = OceanStats.Link.SoftmaxLink();
            
            % --- STRICT DOMAIN VALIDATION ---
            master_type = obj.Experts{1}.getMeasureType();
            for i = 2:obj.K
                if ~strcmp(obj.Experts{i}.getMeasureType(), master_type)
                    error('HME:CompetitiveExpert:UnitMismatch', 'Mixed MeasureTypes in pool.');
                end
            end
        end
        
        function Y_hat = predict(obj, X)
            if ~obj.IsInitialized, obj.initialize(size(X, 2)); end
            g = obj.computeHiringProbabilities(X); 
            
            % Get first prediction to determine dimensionality (K-classes)
            first_pred = obj.Experts{1}.predict(X);
            K_class = size(first_pred, 2);
            
            Y_hat = zeros(size(X, 1), K_class);
            for i = 1:obj.K
                pred_i = obj.Experts{i}.predict(X);
                for k = 1:K_class
                    Y_hat(:, k) = Y_hat(:, k) + g(:, i) .* pred_i(:, k);
                end
            end
        end
        
        function likelihood = calculateLikelihood(obj, X, Y)
            if ~obj.IsInitialized, obj.initialize(size(X, 2)); end
            g = obj.computeHiringProbabilities(X);
            likelihood = zeros(size(X, 1), 1);
            for i = 1:obj.K
                likelihood = likelihood + g(:, i) .* obj.Experts{i}.calculateLikelihood(X, Y);
            end
        end
        
        function update(obj, X, Y, h_leaf)
            % THE ADVANCED M-STEP
            if ~obj.IsInitialized, obj.initialize(size(X, 2)); end

            % 1. Calculate Pool Sub-Responsibilities (Competition)
            g = obj.computeHiringProbabilities(X);
            L_pool = zeros(size(X, 1), obj.K);
            for i = 1:obj.K
                L_pool(:, i) = obj.Experts{i}.calculateLikelihood(X, Y);
            end
            
            % Marginal Likelihood of the whole pool
            denom = sum(g .* L_pool, 2);
            denom = max(denom, 1e-10);
            h_pool = (g .* L_pool) ./ repmat(denom, 1, obj.K);
            
            % 2. Update Specialist Workers (Cascade)
            for i = 1:obj.K
                h_joint = h_leaf .* h_pool(:, i);
                if sum(h_joint) > 1e-6
                    obj.Experts{i}.update(X, Y, h_joint);
                end
            end
            
            % 3. Update Hiring Gating (Solution A & C Combined)
            if obj.UseLocalGating
                % Solve GLM for local hiring weights
                import OceanStats.Optimization.*
                % Applying Sparsity: We can amplify the winners here or uses a Dirichlet penalty
                % For now, we fit the GLM to the sub-responsibilities.
                recipe = GLMRecipe(X, h_pool, obj.Link, ...
                    'InitialTheta', obj.PoolGamma, 'Weights', h_leaf, 'Name', 'Pool Hiring Gate');
                optimizer = GenericOptimizer('MaxIter', 5, 'Verbose', false);
                [final_params, ~] = optimizer.run(recipe, []);
                obj.PoolGamma = final_params.Theta;
            else
                % Update Global Prior with Sparsity (Solution C)
                % Formula: Prior propto sum(h_joint) + (Alpha - 1)
                % If Alpha > 1, it favors uniform. If Alpha < 1, it favors sparsity (1.0/0.0).
                raw_weights = sum(repmat(h_leaf, 1, obj.K) .* h_pool, 1);
                
                % Applying Dirichlet Sparsity Penalty (Solution C)
                % We nudge the counts slightly to favor the winner
                counts = raw_weights + (obj.Alpha - 1);
                counts = max(counts, 1e-10); % Positivity guard
                obj.Prior = counts / sum(counts);
            end
            
            obj.IsInitialized = true;
        end
        
        function params = getParams(obj)
            params.Prior = obj.Prior;
            params.PoolGamma = obj.PoolGamma;
            params.ChildParams = cell(1, obj.K);
            for i = 1:obj.K, params.ChildParams{i} = obj.Experts{i}.getParams(); end
            params.IsInitialized = obj.IsInitialized;
        end
        
        function setParams(obj, params)
            obj.Prior = params.Prior;
            obj.PoolGamma = params.PoolGamma;
            for i = 1:obj.K, obj.Experts{i}.setParams(params.ChildParams{i}); end
            obj.IsInitialized = params.IsInitialized;
        end
        
        function reset(obj)
            obj.IsInitialized = false;
            obj.Prior = ones(1, obj.K) / obj.K;
            for i = 1:obj.K, obj.Experts{i}.reset(); end
        end
        
        function warmup(obj, X, Y)
            obj.initialize(size(X, 2));
            for i = 1:obj.K
                obj.Experts{i}.warmup(X, Y);
                % Solution B: Structural Diversity via asymmetric warmup if they are identical
                % This is a soft fallback for structural diversity.
                p = obj.Experts{i}.getParams();
                if isfield(p, 'Beta')
                    p.Beta = p.Beta + randn(size(p.Beta)) * 0.1;
                    obj.Experts{i}.setParams(p);
                end
            end
            obj.IsInitialized = true;
        end
        
        function weights = getWorkerWeights(obj, X)
            if nargin < 2 || isempty(X)
                weights = obj.Prior;
            else
                weights = obj.computeHiringProbabilities(X);
            end
        end
        
        function type = getMeasureType(obj)
            type = obj.Experts{1}.getMeasureType();
        end
    end
    
    methods (Access = private)
        function initialize(obj, D)
            if isempty(obj.PoolGamma)
                obj.PoolGamma = zeros(D, obj.K);
            end
            obj.IsInitialized = true;
        end
        
        function g = computeHiringProbabilities(obj, X)
            if obj.UseLocalGating
                g = obj.Link.apply(X * obj.PoolGamma);
            else
                g = repmat(obj.Prior, size(X, 1), 1);
            end
        end
    end
end
