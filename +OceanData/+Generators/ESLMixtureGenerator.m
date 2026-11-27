classdef ESLMixtureGenerator < OceanData.Generators.IDataGenerator
    % ESLMIXTUREGENERATOR Generates K-class mixture data from ESL.
    %
    % Data Generation Process (for each class k = 1..K):
    % 1. Create CentersPerClass means from N(shift_k, I)
    %    where shift_k is auto-spaced on a unit circle.
    % 2. To sample a point for Class k, select one center uniformly
    %    at random, and add noise N(0, I/5).
    %
    % Constructor:
    %   gen = ESLMixtureGenerator('N', 200)                       % 2-class (default, backward compat)
    %   gen = ESLMixtureGenerator('N', 300, 'NumClasses', 3)      % 3-class
    %   gen = ESLMixtureGenerator('NumClasses', 5, 'CentersPerClass', 8)
    %
    % Ref: Hastie, Tibshirani, Friedman (2009). The Elements of Statistical Learning.
    %      Fig 2.1 (K=2), Fig 13.1 (K=3).
    
    properties
        NumClasses     (1,1) double {mustBePositive, mustBeInteger} = 2
        CentersPerClass (1,1) double {mustBePositive, mustBeInteger} = 10
    end

    methods
        function obj = ESLMixtureGenerator(opts)
            arguments
                opts.N (1,1) double = 200
                opts.Seed (1,1) double = 42
                opts.NumClasses (1,1) double = 2
                opts.CentersPerClass (1,1) double = 10
            end
            obj.N = opts.N;
            obj.Seed = opts.Seed;
            obj.NumClasses = opts.NumClasses;
            obj.CentersPerClass = opts.CentersPerClass;
            
            % Generate and cache the fixed centers
            rng(obj.Seed, 'twister');
            
            K = obj.NumClasses;
            R = obj.CentersPerClass;
            
            % Auto-space class mean shifts on a unit circle
            % K=2: angles [0, pi/2] → shifts (1,0) and (0,1) — backward compat
            % K=3: angles [0, 2pi/3, 4pi/3] — 120° apart
            % K>=4: evenly spaced on circle
            if K == 2
                % Preserve original ESL shifts for exact backward compatibility
                shifts = [1, 0; 0, 1];
            else
                angles = 2 * pi * (0:(K-1))' / K;
                shifts = [cos(angles), sin(angles)];
            end
            
            obj.MetaData.ClassCenters = cell(K, 1);
            for k = 1:K
                centers = randn(R, 2);
                centers(:,1) = centers(:,1) + shifts(k, 1);
                centers(:,2) = centers(:,2) + shifts(k, 2);
                obj.MetaData.ClassCenters{k} = centers;
            end
            
            % Backward compat fields for K=2
            if K == 2
                obj.MetaData.Class1Centers = obj.MetaData.ClassCenters{1};
                obj.MetaData.Class2Centers = obj.MetaData.ClassCenters{2};
            end
            
            obj.MetaData.Dimensions = 2;
            obj.MetaData.Classes = K;
            obj.MetaData.Shifts = shifts;
        end
    end
    
    methods
        function tf = hasBayesOracle(~)
            tf = true;
        end

        function [y_pred, err, densities] = bayesPredict(obj, X, y_true)
            % BAYESPREDICT  Bayes optimal classifier using true generative centers.
            %
            %   [y_pred, err] = gen.bayesPredict(X_test, y_test)
            %
            %   Computes per-class mixture densities and picks the argmax.
            arguments
                obj
                X       double
                y_true  = []
            end

            D = obj.bayesDensity(X);   % N x K

            [~, idx] = max(D, [], 2);
            y_pred = idx;              % class labels are 1..K

            if ~isempty(y_true)
                err = mean(y_pred ~= y_true);
            else
                err = NaN;
            end

            if nargout >= 3
                densities = D;
            end
        end

        function D = bayesDensity(obj, X)
            % BAYESDENSITY  Unnormalized class-conditional mixture densities.
            %
            %   D = gen.bayesDensity(X)   % returns N x K matrix
            arguments
                obj
                X double
            end

            K = obj.NumClasses;
            N = size(X, 1);
            sigma2 = 1/5;
            scale  = 1 / (2 * sigma2);     % = 2.5

            D = zeros(N, K);
            for k = 1:K
                centers = obj.MetaData.ClassCenters{k};
                R = size(centers, 1);
                dk = zeros(N, 1);
                for r = 1:R
                    diff = X - centers(r, :);
                    dk = dk + exp(-scale * sum(diff.^2, 2));
                end
                D(:, k) = dk;
            end
        end
    end

    methods (Access = protected)
        function [X, Y] = generateCore(obj)
            K = obj.NumClasses;
            R = obj.CentersPerClass;
            
            % Balanced class distribution
            Y = zeros(obj.N, 1);
            nPerClass = floor(obj.N / K);
            for k = 1:K
                startIdx = (k-1)*nPerClass + 1;
                if k == K
                    endIdx = obj.N;   % last class absorbs remainder
                else
                    endIdx = k * nPerClass;
                end
                Y(startIdx:endIdx) = k;
            end
            
            X = zeros(obj.N, 2);
            for i = 1:obj.N
                k = Y(i);
                centers = obj.MetaData.ClassCenters{k};
                idx = randi(R);
                noise = randn(1, 2) * sqrt(1/5);
                X(i, :) = centers(idx, :) + noise;
            end
        end
    end
end
