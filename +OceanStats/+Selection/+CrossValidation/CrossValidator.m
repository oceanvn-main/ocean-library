classdef CrossValidator < handle
    % CROSSVALIDATOR A generic worker for performing Cross-Validation.
    %
    % OVERVIEW:
    %   Takes a Model Strategy and Data, partitions the data, and evaluates
    %   the model on each fold (possibly in parallel).
    %
    % EXAMPLES:
    %   cv = CrossValidator('KFold', 10);
    %   results = cv.evaluate(strategy, X, Y);
    %
    % Author: Ocean Library System
    
    properties
        PartitionStrategy (1, 1) string = "KFold" % "KFold", "HoldOut", "LeaveOneOut"
        NumFolds          (1, 1) double = 10      % For KFold
        HoldOutRatio      (1, 1) double = 0.3     % For HoldOut
        UseParallel       (1, 1) logical = true
    end
    
    methods
        function obj = CrossValidator(varargin)
            if nargin > 0
                for i = 1:2:nargin
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
        
        function results = evaluate(obj, strategy, X, Y)
            % EVALUATE Runs the cross-validation.
            %
            % Args:
            %   strategy (IModelStrategy): The model definition.
            %   X (N x P): Features.
            %   Y (N x 1): Target.
            %
            % Returns:
            %   results (struct): containing MeanError, StdError, RawErrors.
            
            % 1. Create Partition
            if strcmpi(obj.PartitionStrategy, "KFold")
                cvPart = cvpartition(Y, 'KFold', obj.NumFolds);
            elseif strcmpi(obj.PartitionStrategy, "HoldOut")
                cvPart = cvpartition(Y, 'HoldOut', obj.HoldOutRatio);
            elseif strcmpi(obj.PartitionStrategy, "LeaveOneOut")
                cvPart = cvpartition(Y, 'LeaveOut');
            else
                error('Unknown PartitionStrategy: %s', obj.PartitionStrategy);
            end
            
            K = cvPart.NumTestSets;
            foldErrors = zeros(K, 1);
            
            % 2. Broadcast Strategy for Parallel Execution
            % We assume strategy is stateless or configured.
            
            if obj.UseParallel
                pool = gcp('nocreate');
                if isempty(pool), pool = parpool(); end
                
                parfor k = 1:K
                    % 2a. Split Data
                    trIdx = training(cvPart, k);
                    teIdx = test(cvPart, k);
                    
                    X_tr = X(trIdx, :);
                    Y_tr = Y(trIdx); %#ok<PFBNS>
                    X_te = X(teIdx, :);
                    Y_te = Y(teIdx);
                    
                    % 2b. Train
                    % Note: 'strategy' variable is broadcast to workers.
                    model = strategy.train(X_tr, Y_tr);
                    
                    % 2c. Predict
                    Y_pred = strategy.predict(model, X_te);
                    
                    % 2d. Evaluate
                    err = strategy.evaluate(Y_te, Y_pred);
                    foldErrors(k) = err;
                end
            else
                for k = 1:K
                    trIdx = training(cvPart, k);
                    teIdx = test(cvPart, k);
                    X_tr = X(trIdx, :);
                    Y_tr = Y(trIdx);
                    X_te = X(teIdx, :);
                    Y_te = Y(teIdx);
                    model = strategy.train(X_tr, Y_tr);
                    Y_pred = strategy.predict(model, X_te);
                    err = strategy.evaluate(Y_te, Y_pred);
                    foldErrors(k) = err;
                end
            end
            
            % 3. Aggregate Results
            results.MeanError = mean(foldErrors);
            results.StdError  = std(foldErrors) / sqrt(K);
            results.FoldErrors = foldErrors;
        end
    end
end
