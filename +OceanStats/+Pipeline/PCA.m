classdef PCA < OceanStats.Base.IModelStrategy
    % PCA Principal Component Analysis Transformer.
    %
    % USAGE:
    %   pipe = Pipeline({PCA(3), LinearReg()});
    
    properties
        Name string = "PCA"
        SupportsAIC logical = false
        NumComponents (1, 1) double = 2
    end
    
    methods
        function obj = PCA(n_comp)
            arguments
                n_comp (1,1) double {mustBePositive, mustBeInteger} = 2
            end
            obj.NumComponents = n_comp;
            obj.Name = "PCA (k=" + obj.NumComponents + ")";
        end
        
        function model = train(obj, X, Y)
            % TRAIN Learns the projection matrix.
            % Returns a 'model' struct containing the transformation logic.
            arguments
                obj
                X (:,:) double
                Y = []
            end
            
            % 1. Centering
            mu = mean(X);
            X_c = X - mu;
            
            % 2. Explicit PCA via SVD
            % X_c = U * S * V'
            % The columns of V are the principal component directions (loadings).
            % We use 'econ' to avoid computing the full U matrix if N > P.
            [~, S, V] = svd(X_c, 'econ');
            
            % Compute eigenvalues (variance explained)
            s_diag = diag(S);
            eigenvalues = (s_diag.^2) / (max(1, size(X, 1) - 1));
            
            % 3. Truncate to requested components
            K = min(obj.NumComponents, size(V, 2));
            coeffs = V(:, 1:K);
            
            % Store state in model
            model = struct('Mu', mu, 'Coeffs', coeffs, 'Eigenvalues', eigenvalues);
        end
        
        function X_new = predict(obj, model, X)
            % PREDICT Applies the transformation.
            % Alias for transform to satisfy Pipeline.
            arguments
                obj
                model struct
                X (:,:) double
            end
            
            % 1. Center
            X_c = X - model.Mu;
            
            % 2. Project
            X_new = X_c * model.Coeffs;
        end
        
        function err = evaluate(~, ~, ~)
            err = NaN; 
        end
        
        function k = getComplexity(obj, model)
            k = 0; % Transformer
        end
    end
end
