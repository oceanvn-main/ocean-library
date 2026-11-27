classdef InternalNode < OceanStats.Ensemble.HME.BaseNode
    % INTERNALNODE Represents a Gating Network in the HME model.
    
    properties (Access = private)
        Children
        K
        Gamma
        Link
        IsInitialized = false
        GatingProbs
    end
    
    methods
        function obj = InternalNode(children)
            obj.Children = children;
            obj.K = length(children);
            obj.Link = OceanStats.Link.SoftmaxLink();
        end
        
        function gate = getGate(obj)
            gate = struct('predict', @(X) obj.computeGatingProbabilities(X));
        end
        
        function child = getChild(obj, i)
            % GETCHILD Returns the i-th child node (1-indexed).
            if i < 1 || i > obj.K
                error('HME:InternalNode:InvalidIndex', ...
                    'Child index %d is out of range [1, %d].', i, obj.K);
            end
            child = obj.Children{i};
        end
        
        function Y_hat = predict(obj, X)
            g = obj.computeGatingProbabilities(X);
            
            % Detect dimensionality (K-classes) from first child
            first_pred = obj.Children{1}.predict(X);
            K_class = size(first_pred, 2);
            
            Y_hat = zeros(size(X, 1), K_class);
            for i = 1:obj.K
                pred_i = obj.Children{i}.predict(X);
                for k = 1:K_class
                    Y_hat(:, k) = Y_hat(:, k) + g(:, i) .* pred_i(:, k);
                end
            end
        end
        
        function likelihood = computeLikelihood(obj, X, Y, iter)
            if ~obj.IsInitialized, obj.initialize(size(X, 2)); end
            if nargin < 4, iter = []; end
            g = obj.computeGatingProbabilities(X);
            obj.GatingProbs = g;
            
            likelihood = zeros(size(X, 1), 1);
            for i = 1:obj.K
                likelihood = likelihood + g(:, i) .* obj.Children{i}.computeLikelihood(X, Y, iter);
            end
            % Numerical Guard
            likelihood(isnan(likelihood)) = 0;
            likelihood(isinf(likelihood)) = 1e100;
            obj.LastLikelihood = likelihood;
        end
        
        function computeResponsibility(obj, X, Y, prior, totalLikelihood)
            g = obj.GatingProbs;
            obj.LocalResponsibility = prior; 
            denom = max(obj.LastLikelihood, 1e-10);
            for i = 1:obj.K
                L_child = obj.Children{i}.LastLikelihood;
                h_conditional = (g(:, i) .* L_child) ./ denom;
                obj.Children{i}.computeResponsibility(X, Y, prior .* h_conditional, totalLikelihood);
            end
        end
        
        function updateParameters(obj, X, Y, iter)
            h_matrix = zeros(size(X, 1), obj.K);
            denom = max(obj.LastLikelihood, 1e-10);
            for i = 1:obj.K
                h_matrix(:, i) = (obj.GatingProbs(:, i) .* obj.Children{i}.LastLikelihood) ./ denom;
            end
            
            import OceanStats.Optimization.*
            recipe = GLMRecipe(X, h_matrix, obj.Link, ...
                'InitialTheta', obj.Gamma, 'Weights', obj.LocalResponsibility, 'Name', 'Gating Update');
            optimizer = GenericOptimizer('MaxIter', 15, 'Verbose', false);
            [final_params, ~] = optimizer.run(recipe, []);
            
            % Annealing
            if nargin > 3 && ~isempty(iter) && iter <= 10
                annealing = 0.5 + 0.5 * (iter / 10);
                obj.Gamma = final_params.Theta * annealing;
            else
                obj.Gamma = final_params.Theta;
            end
            
            for i = 1:obj.K
                obj.Children{i}.updateParameters(X, Y, iter);
            end
        end
        
        function warmup(obj, X, Y, labels)
            if isempty(X) || isempty(labels), return; end
            child_labels = zeros(size(labels));
            offset = 0;
            for i = 1:obj.K
                child_leaves = obj.Children{i}.getLeaves();
                num_child_leaves = length(child_leaves);
                mask = (labels > offset) & (labels <= offset + num_child_leaves);
                child_labels(mask) = i;
                obj.Children{i}.warmup(X(mask, :), Y(mask), labels(mask) - offset);
                offset = offset + num_child_leaves;
            end
            if any(child_labels > 0)
                targets = zeros(length(child_labels), obj.K);
                for i = 1:obj.K, targets(child_labels == i, i) = 1.0; end
                import OceanStats.Optimization.*
                recipe = GLMRecipe(X, targets, obj.Link, 'InitialTheta', obj.Gamma);
                optimizer = GenericOptimizer('MaxIter', 15, 'Verbose', false);
                [final_params, ~] = optimizer.run(recipe, []);
                obj.Gamma = final_params.Theta;
            end
            obj.IsInitialized = true;
        end
        
        function leaves = getLeaves(obj)
            leaves = {};
            for i = 1:obj.K, leaves = [leaves, obj.Children{i}.getLeaves()]; end %#ok<AGROW>
        end
        
        function h = getResponsibility(obj), h = obj.LocalResponsibility; end
        
        function params = getParams(obj)
            params.Gamma = obj.Gamma;
            params.ChildParams = cell(1, obj.K);
            for i = 1:obj.K, params.ChildParams{i} = obj.Children{i}.getParams(); end
            params.IsInitialized = obj.IsInitialized;
        end
        
        function setParams(obj, params)
            obj.Gamma = params.Gamma;
            for i = 1:obj.K, obj.Children{i}.setParams(params.ChildParams{i}); end
            obj.IsInitialized = params.IsInitialized;
        end
        
        function reset(obj)
            obj.IsInitialized = false;
            for i = 1:obj.K, obj.Children{i}.reset(); end
        end
    end
    
    methods (Access = private)
        function initialize(obj, D)
            obj.Gamma = 0.5 * randn(D, obj.K);
            obj.IsInitialized = true;
        end
        
        function g = computeGatingProbabilities(obj, X)
            g = obj.Link.apply(X * obj.Gamma);
        end
    end
end