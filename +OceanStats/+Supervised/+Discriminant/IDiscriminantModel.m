classdef (Abstract) IDiscriminantModel < handle
    % IDISCRIMINANTMODEL Abstract interface for Discriminant Analysis
    %
    % Description:
    %   Defines the core contract for discriminant models (e.g., LDA, QDA,
    %   FDA, MDA) within the OceanStats library. Ensures that all
    %   discriminant models implement standard training, prediction (with
    %   posteriors), and canonical projection methods.
    %
    % Properties:
    %   Classes    - K x 1 vector of unique class labels
    %   NumClasses - Integer representing K
    %   Priors     - K x 1 vector of estimated or provided prior probabilities
    
    properties (SetAccess = protected)
        Classes
        NumClasses
        Priors
    end
    
    methods (Abstract)
        % FIT Train the discriminant model
        %
        % Inputs:
        %   X - N x p numeric predictor matrix
        %   y - N x 1 numeric class labels
        fit(obj, X, y)
        
        % PREDICT Classify new observations
        %
        % Inputs:
        %   X - M x p numeric predictor matrix
        %
        % Outputs:
        %   y_pred    - M x 1 vector of predicted class labels
        %   posterior - M x K matrix of posterior probabilities (optional)
        [y_pred, posterior] = predict(obj, X)
    end
    
    methods
        function coords = project(obj, X, dims)
            % PROJECT Transform data into canonical discriminant space
            % Optional implementation. Some methods like QDA or independent
            % MDA do not yield a global linear canonical projection space.
            error('OceanStats:Discriminant:NoProjection', ...
                'Projection to canonical space is not natively supported by this specific discriminant method.');
        end
    end
end
