classdef Box < handle
    % BOX Represents a hyperrectangle (bounding box) for the PRIM algorithm.
    % Keeps track of spatial boundaries, quality score, and box statistics.
    
    properties
        Bounds      % [D x 2] matrix where Col 1 = Min limit, Col 2 = Max limit
        Score       (1,1) double = -Inf
        NSamples    (1,1) double = 0
        BoxMean     (1,1) double = NaN
    end
    
    methods
        function obj = Box(X, Y, objectiveFn)
            % BOX Constructor.
            % Initialize as the maximal box enclosing all given X data.
            if nargin > 0
                [~, D] = size(X);
                obj.Bounds = zeros(D, 2);
                for j = 1:D
                    obj.Bounds(j, 1) = -Inf;
                    obj.Bounds(j, 2) = Inf;
                end
                
                if nargin > 2 && ~isempty(objectiveFn)
                    obj.NSamples = length(Y);
                    obj.Score = objectiveFn.evaluate(Y, X);
                    obj.BoxMean = mean(Y);
                end
            end
        end
        
        function mask = isInside(obj, X)
            % ISINSIDE Returns boolean mask [N x 1] for samples inside the box.
            [N, D] = size(X);
            mask = true(N, 1);
            for j = 1:D
                % Points must be between min and max bounds inclusive
                inDim = (X(:, j) >= obj.Bounds(j, 1)) & (X(:, j) <= obj.Bounds(j, 2));
                mask = mask & inDim;
            end
        end
        
        function newBox = copy(obj)
            % COPY Creates a deep copy of the box.
            newBox = OceanStats.Supervised.PRIM.Box();
            newBox.Bounds = obj.Bounds;
            newBox.Score = obj.Score;
            newBox.NSamples = obj.NSamples;
            newBox.BoxMean = obj.BoxMean;
        end
    end
end
