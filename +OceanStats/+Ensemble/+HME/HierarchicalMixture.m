classdef HierarchicalMixture < handle
    % HIERARCHICALMIXTURE Main facade for the HME model.
    
    properties (Access = private)
        RootNode
        Trainer
    end
    
    properties
        Type = 'Regression' % 'Regression' or 'Classification'
    end
    
    methods
        function obj = HierarchicalMixture(rootNode, options)
            arguments
                rootNode
                options.Type = 'Regression'
            end
            obj.RootNode = rootNode;
            obj.Type = options.Type;
            obj.Trainer = OceanStats.Ensemble.HME.EMTrainer();
        end
        
        function root = getRoot(obj), root = obj.RootNode; end
        
        function [final_ll, params] = fit(obj, data)
            X = data(:, 1:end-1);
            Y = data(:, end);
            [final_ll, params] = obj.Trainer.fit(obj, X, Y);
        end
        
        function setTrainer(obj, trainer)
            obj.Trainer = trainer;
        end
        
        function setInitializer(obj, initializer)
            obj.Trainer.Initializer = initializer;
        end
        
        function Y_hat = predict(obj, X, params)
            if nargin > 2 && ~isempty(params)
                old_params = obj.getParams();
                obj.setParams(params);
                Y_hat = obj.RootNode.predict(X);
                obj.setParams(old_params);
            else
                Y_hat = obj.RootNode.predict(X);
            end
        end
        
        function reset(obj), obj.RootNode.reset(); end
        function params = getParams(obj), params = obj.RootNode.getParams(); end
        function setParams(obj, params), obj.RootNode.setParams(params); end
        function warmup(obj, X, Y, labels), obj.RootNode.warmup(X, Y, labels); end
        function leaves = getLeaves(obj), leaves = obj.RootNode.getLeaves(); end
        
        %%% EM DELEGATES %%%
        function total_likelihood = computeLikelihoodPass(obj, X, Y, iter)
            total_likelihood = obj.RootNode.computeLikelihood(X, Y, iter);
        end
        function distributeResponsibilityPass(obj, X, Y, total_likelihood)
            prior = ones(size(X, 1), 1);
            obj.RootNode.computeResponsibility(X, Y, prior, total_likelihood);
        end
        function updateParameters(obj, X, Y, iter)
            obj.RootNode.updateParameters(X, Y, iter);
        end
    end
end
