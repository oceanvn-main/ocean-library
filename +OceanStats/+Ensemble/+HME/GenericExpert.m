classdef GenericExpert < OceanStats.Ensemble.HME.IExpert
    % GENERICEXPERT Adapter that wraps any library ProbabilisticModel as an HME expert.
    % This follows the "Adapter Pattern" to reuse robust library code.
    
    properties (Access = private)
        Model          % OceanStats.Base.ProbabilisticModel
        Params         % Current fitted parameter struct
        MeasureType    % 'Density' or 'Mass'
        IsInitialized = false
    end
    
    methods
        function obj = GenericExpert(model, measureType)
            if nargin < 2, measureType = 'Density'; end
            obj.Model = model;
            obj.MeasureType = measureType;
        end
        
        function Y_hat = predict(obj, X)
            if ~obj.IsInitialized
                Y_hat = zeros(size(X, 1), 1); % Default fallback
                return;
            end
            Y_hat = obj.Model.predict(X, obj.Params);
        end
        
        function likelihood = calculateLikelihood(obj, X, Y)
            if ~obj.IsInitialized
                likelihood = ones(size(X, 1), 1);
                return;
            end
            
            % Logic: Delegate to model if it has calculateLikelihood
            % Otherwise, implement generic Gaussian or Bernoulli based on MeasureType
            if ismethod(obj.Model, 'calculateLikelihood')
                likelihood = obj.Model.logLikelihood([X, Y], obj.Params);
                likelihood = exp(likelihood);
            elseif strcmp(obj.MeasureType, 'Density')
                % Gaussian Likelihood Fallback
                y_hat = obj.predict(X);
                sigma = obj.Params.sigma;
                precision = 1 / max(sigma^2, 1e-10);
                exponent = -0.5 * (Y - y_hat).^2 * precision;
                constant = 1 / (sqrt(2 * pi * sigma^2) + 1e-10);
                likelihood = constant * exp(exponent);
            else
                % Categorical / Bernoulli Likelihood Fallback
                p = obj.predict(X);
                if size(p, 2) > 1
                    % Multi-class Categorical: P(Y=k) = p_k
                    likelihood = zeros(size(X, 1), 1);
                    for i = 1:size(X, 1)
                        class_idx = round(Y(i));
                        if class_idx > 0 && class_idx <= size(p, 2)
                            likelihood(i) = p(i, class_idx);
                        end
                    end
                else
                    % Binary Bernoulli
                    p = min(max(p, 1e-10), 1 - 1e-10);
                    likelihood = (p.^Y) .* ((1 - p).^(1 - Y));
                end
            end
        end
        
        function update(obj, X, Y, h)
            % M-Step: Leverage the library model's weighted fit
            obj.Params = obj.Model.fit([X, Y], h);
            obj.IsInitialized = true;
        end
        
        function params = getParams(obj)
            params.HMEParams = obj.Params;
            params.IsInitialized = obj.IsInitialized;
        end
        
        function setParams(obj, params)
            obj.Params = params.HMEParams;
            obj.IsInitialized = params.IsInitialized;
        end
        
        function reset(obj)
            obj.IsInitialized = false;
        end
        
        function warmup(obj, X, Y)
            % Data-driven initialization via the library model
            obj.update(X, Y, ones(size(X, 1), 1));
        end
        
        function type = getMeasureType(obj)
            type = obj.MeasureType;
        end
    end
end
