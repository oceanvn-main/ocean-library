classdef LVQ1 < handle
    % LVQ1 Learning Vector Quantization (Kohonen, 1989).
    %
    % ALGORITHM:
    %   1. Initialize R prototypes per class via K-means within each class.
    %   2. For each training sample (randomly shuffled):
    %      - Find the nearest prototype (Euclidean distance).
    %      - If the prototype's class matches the sample's class:
    %            move prototype TOWARD the sample.
    %      - Otherwise:
    %            move prototype AWAY from the sample.
    %   3. Decay learning rate after each epoch.
    %
    % PREDICTION:
    %   Classify new points by nearest-prototype (1-NN on prototypes).
    %
    % USAGE:
    %   lvq = OceanStats.Supervised.Neighbors.LVQ1('R', 5, 'MaxEpochs', 50);
    %   lvq.fit(X_train, y_train);
    %   y_pred = lvq.predict(X_test);
    %
    % Ref: Kohonen (1989). Self-Organization and Associative Memory.
    %      Hastie et al. (2009), ESL Section 13.2.
    
    properties
        R            (1,1) double {mustBePositive, mustBeInteger} = 5
        LearningRate (1,1) double {mustBePositive} = 0.1
        MaxEpochs    (1,1) double {mustBePositive, mustBeInteger} = 50
        Decay        (1,1) double {mustBeNonnegative} = 0.01
        Verbose      (1,1) logical = false
        
        % Fitted state
        Prototypes      % (R*K x p) prototype locations
        PrototypeLabels % (R*K x 1) prototype class labels
    end
    
    methods
        function obj = LVQ1(opts)
            arguments
                opts.R            (1,1) double = 5
                opts.LearningRate (1,1) double = 0.1
                opts.MaxEpochs    (1,1) double = 50
                opts.Decay        (1,1) double = 0.01
                opts.Verbose      (1,1) logical = false
            end
            obj.R = opts.R;
            obj.LearningRate = opts.LearningRate;
            obj.MaxEpochs = opts.MaxEpochs;
            obj.Decay = opts.Decay;
            obj.Verbose = opts.Verbose;
        end
        
        function fit(obj, X, y, opts)
            % FIT Train prototypes via LVQ1.
            %
            %   lvq.fit(X, y)
            %   lvq.fit(X, y, 'InitPrototypes', P, 'InitLabels', L)
            %
            % Optional: provide initial prototypes (e.g. from K-means).
            arguments
                obj
                X double
                y double
                opts.InitPrototypes double = []
                opts.InitLabels double = []
            end
            
            classes = unique(y);
            K = numel(classes);
            [N, p] = size(X);
            
            % --- Initialize prototypes ---
            if ~isempty(opts.InitPrototypes) && ~isempty(opts.InitLabels)
                obj.Prototypes = opts.InitPrototypes;
                obj.PrototypeLabels = opts.InitLabels;
            else
                % K-means initialization within each class
                obj.Prototypes = zeros(K * obj.R, p);
                obj.PrototypeLabels = zeros(K * obj.R, 1);
                km = OceanStats.Unsupervised.KMeans();
                for k = 1:K
                    X_k = X(y == classes(k), :);
                    km.fit(X_k, obj.R);
                    startIdx = (k-1)*obj.R + 1;
                    endIdx = k * obj.R;
                    obj.Prototypes(startIdx:endIdx, :) = km.Centroids;
                    obj.PrototypeLabels(startIdx:endIdx) = classes(k);
                end
            end
            
            % --- LVQ1 training loop ---
            nProto = size(obj.Prototypes, 1);
            lr = obj.LearningRate;
            
            for epoch = 1:obj.MaxEpochs
                % Shuffle training data
                perm = randperm(N);
                nCorrect = 0;
                
                for i = 1:N
                    xi = X(perm(i), :);
                    yi = y(perm(i));
                    
                    % Find nearest prototype
                    dists = sum((obj.Prototypes - xi).^2, 2);
                    [~, nearest] = min(dists);
                    
                    if obj.PrototypeLabels(nearest) == yi
                        % Correct: pull toward
                        obj.Prototypes(nearest, :) = obj.Prototypes(nearest, :) + ...
                            lr * (xi - obj.Prototypes(nearest, :));
                        nCorrect = nCorrect + 1;
                    else
                        % Wrong: push away
                        obj.Prototypes(nearest, :) = obj.Prototypes(nearest, :) - ...
                            lr * (xi - obj.Prototypes(nearest, :));
                    end
                end
                
                % Decay learning rate
                lr = obj.LearningRate * exp(-obj.Decay * epoch);
                
                if obj.Verbose
                    acc = nCorrect / N;
                    fprintf('[LVQ1] Epoch %3d/%d | LR = %.5f | Train Acc = %.3f\n', ...
                        epoch, obj.MaxEpochs, lr, acc);
                end
            end
        end
        
        function y_pred = predict(obj, X)
            % PREDICT Classify via nearest-prototype (1-NN on prototypes).
            %
            %   y_pred = lvq.predict(X_test)
            arguments
                obj
                X double
            end
            
            N = size(X, 1);
            y_pred = zeros(N, 1);
            for i = 1:N
                dists = sum((obj.Prototypes - X(i,:)).^2, 2);
                [~, nearest] = min(dists);
                y_pred(i) = obj.PrototypeLabels(nearest);
            end
        end
    end
end
