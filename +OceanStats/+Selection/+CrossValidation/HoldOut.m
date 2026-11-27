classdef HoldOut < OceanStats.Base.ICrossValidator
    % HOLDOUT Simple Train/Test Split Validation.
    %
    % OVERVIEW:
    %   Splits data once into a training set and a test set based on a ratio.
    %   - Training Set: (1 - TestRatio) * N samples.
    %   - Test Set: TestRatio * N samples.
    %   - NumFolds: Always 1.
    %
    % USAGE:
    %   cv = HoldOut(0.2); % 20% validation set
    %   cv = HoldOut(0.3, 'Seed', 123);
    
    properties
        NumFolds double = 1
        TestRatio (1,1) double {mustBeInRange(TestRatio, 0, 1)} = 0.2
        Seed (1,1) double = -1
        
        % Internal State
        Cache struct
    end
    
    methods
        function obj = HoldOut(ratio, options)
            % Usage: HoldOut(0.2, 'Seed', 42)
            arguments
                ratio (1,1) double = 0.2
                options.Seed (1,1) double = -1
            end
            obj.TestRatio = ratio;
            obj.Seed = options.Seed;
            obj.reset();
        end
        
        function reset(obj)
             % Force-clears the cache to ensure independence for next run.
            obj.Cache = struct('N', 0, 'cv', []);
        end
        
        function [trainIdx, testIdx] = getIndices(obj, N, k)
            % Check cache consistency
            if obj.Cache.N ~= N
                % 1. Manage Randomness
                if obj.Seed ~= -1
                    s = RandStream('dsfmt19937', 'Seed', obj.Seed);
                    RandStream.setGlobalStream(s);
                end
                
                % 2. Create and Cache Partition
                obj.Cache.N = N;
                obj.Cache.cv = cvpartition(N, 'HoldOut', obj.TestRatio);
            end
            
             % 3. Retrieve Indices (HoldOut only has 1 fold, so k is ignored or must be 1)
            c = obj.Cache.cv;
            trainIdx = c.training;
            testIdx  = c.test;
        end
    end
end
