classdef KFold < OceanStats.Base.ICrossValidator
    % KFOLD Robust K-Fold Cross-Validation with Seed Control.
    %
    % Uses randperm-based fold assignment (same algorithm as crossvalind)
    % with full RNG isolation to avoid contaminating the global stream.
    %
    % USAGE:
    %   cv = OceanStats.Selection.CrossValidation.KFold(10, 'Seed', 42);
    %   for k = 1:10
    %       [trainIdx, testIdx] = cv.getIndices(N, k);
    %   end
    
    properties
        NumFolds double
        Seed (1,1) double = -1  % -1 means "Use Global RNG"
        
        % Internal State
        Cache struct
    end
    
    methods
        function obj = KFold(k, options)
            % Usage: KFold(5, 'Seed', 42)
            arguments
                k (1,1) double
                options.Seed (1,1) double = -1
            end
            obj.NumFolds = k;
            obj.Seed = options.Seed;
            obj.reset();
        end
        
        function reset(obj)
            % Force-clears the cache to ensure independence for next run.
            obj.Cache = struct('N', 0, 'foldIdx', []);
        end
        
        function [trainIdx, testIdx] = getIndices(obj, N, k)
            % Check cache consistency
            if obj.Cache.N ~= N
                % 1. Save global RNG state
                if obj.Seed ~= -1
                    oldRng = rng;
                    rng(obj.Seed);
                end
                
                % 2. Create fold assignments via randperm (same as crossvalind)
                perm = randperm(N);
                foldIdx = zeros(N, 1);
                foldSize = floor(N / obj.NumFolds);
                remainder = mod(N, obj.NumFolds);
                
                idx = 1;
                for f = 1:obj.NumFolds
                    sz = foldSize + (f <= remainder);
                    foldIdx(perm(idx:idx+sz-1)) = f;
                    idx = idx + sz;
                end
                
                obj.Cache.N = N;
                obj.Cache.foldIdx = foldIdx;
                
                % 3. Restore global RNG so downstream code is unaffected
                if obj.Seed ~= -1
                    rng(oldRng);
                end
            end
            
            % 4. Retrieve Indices
            testIdx  = (obj.Cache.foldIdx == k);
            trainIdx = ~testIdx;
        end
    end
end