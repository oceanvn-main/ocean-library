classdef PolyReg < OceanStats.Base.IModelStrategy
    % POLYREG Polynomial Regression Strategy.
    %
    % Wraps polyfit/polyval.
    
    properties
        Name string
        Degree double
        SupportsAIC logical = true
    end
    
    methods
        function obj = PolyReg(degree)
            obj.Degree = degree;
            obj.Name = "PolyReg (d=" + degree + ")";
        end
        
        function model = train(obj, X, Y)
            % Simple 1D polyfit for now
            % If X is multi-dim, this will fail or need interactions.
            % For this experiment, we assume 1D X.
            if size(X, 2) > 1
                 error('PolyReg currently supports only 1D input.');
            end
            model = polyfit(X, Y, obj.Degree);
        end
        
        function preds = predict(obj, model, X)
            preds = polyval(model, X);
        end
        
        function err = evaluate(obj, Y_true, Y_pred)
            err = sum((Y_true - Y_pred).^2); % RSS for BIC
        end
        
        function k = getComplexity(obj, model)
            k = obj.Degree + 1;
        end
    end
end
