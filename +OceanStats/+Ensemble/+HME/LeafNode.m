classdef LeafNode < OceanStats.Ensemble.HME.BaseNode
    % LEAFNODE Terminal node in the HME representing an Expert.
    
    properties (Access = private)
        Expert % Instance of OceanStats.Ensemble.HME.IExpert
    end
    
    methods
        function obj = LeafNode(expert)
            % Constructor accepts an IExpert object (Strategy Pattern)
            obj.Expert = expert;
        end
        
        function expert = getExpert(obj)
            expert = obj.Expert;
        end
        
        function Y_hat = predict(obj, X)
            Y_hat = obj.Expert.predict(X);
        end
        
        function likelihood = computeLikelihood(obj, X, Y, ~)
            likelihood = obj.Expert.calculateLikelihood(X, Y);
            % Numerical Guard
            likelihood(isnan(likelihood)) = 0;
            likelihood(isinf(likelihood)) = 1e100;
            obj.LastLikelihood = likelihood;
        end
        
        function computeResponsibility(obj, ~, ~, prior, ~)
            % In the refactored top-down pass, 'prior' is already the scale-adjusted 
            % posterior h_leaf. We simply store it.
            obj.LocalResponsibility = prior;
        end
        
        function updateParameters(obj, X, Y, ~)
            obj.Expert.update(X, Y, obj.LocalResponsibility);
        end
        
        function warmup(obj, X, Y, ~)
            obj.Expert.warmup(X, Y);
        end
        
        function leaves = getLeaves(obj)
            leaves = {obj};
        end
        
        function h = getResponsibility(obj)
            h = obj.LocalResponsibility;
        end
        
        function params = getParams(obj)
            params = obj.Expert.getParams();
        end
        
        function setParams(obj, params)
            obj.Expert.setParams(params);
        end
        
        function reset(obj)
            obj.Expert.reset();
        end
    end
end
