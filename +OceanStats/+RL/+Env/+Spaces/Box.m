classdef Box < OceanStats.RL.Env.Spaces.ISpace
    % BOX  A bounded (or half-bounded) box in R^n: low <= x <= high element-wise. Use +-Inf for unbounded sides.
    %   sp = OceanStats.RL.Env.Spaces.Box([0 0], [1 1]);   sp.contains([0.5 0.5])
    % sample() is uniform on fully bounded coordinates, standard normal shifted to the finite bound on half-bounded ones,
    % and standard normal on unbounded ones (the Gymnasium convention).
    properties (SetAccess = protected)
        Shape
        Low double
        High double
    end
    methods
        function obj = Box(low, high)
            arguments
                low double {mustBeNonempty}
                high double {mustBeNonempty}
            end
            if isscalar(low) && ~isscalar(high), low = low * ones(size(high)); end
            if isscalar(high) && ~isscalar(low), high = high * ones(size(low)); end
            if ~isequal(size(low), size(high))
                error('OceanStats:RL:Env:Spaces:Box:SizeMismatch', 'low and high must have the same size.');
            end
            if any(isnan(low(:))) || any(isnan(high(:)))
                error('OceanStats:RL:Env:Spaces:Box:BadBounds', 'Bounds must not be NaN.');
            end
            if any(low(:) > high(:))
                error('OceanStats:RL:Env:Spaces:Box:BadBounds', 'low must not exceed high.');
            end
            obj.Low = low;  obj.High = high;  obj.Shape = size(low);
        end
        function tf = contains(obj, x)
            tf = isnumeric(x) && isequal(size(x), obj.Shape) && all(isfinite(x(:)) | isinf(obj.Low(:)) | isinf(obj.High(:))) ...
                && all(x(:) >= obj.Low(:)) && all(x(:) <= obj.High(:)) && ~any(isnan(x(:)));
        end
        function x = sample(obj, stream)
            if nargin < 2, stream = []; end
            stream = OceanStats.RL.Env.Spaces.ISpace.stream_(stream);
            x = zeros(obj.Shape);
            for k = 1:numel(x)
                lo = obj.Low(k);  hi = obj.High(k);
                if isfinite(lo) && isfinite(hi), x(k) = lo + (hi - lo) * rand(stream);
                elseif isfinite(lo), x(k) = lo + abs(randn(stream));
                elseif isfinite(hi), x(k) = hi - abs(randn(stream));
                else, x(k) = randn(stream);
                end
            end
        end
        function s = describe(obj)
            s = sprintf('Box(shape=%s)', mat2str(obj.Shape));
        end
    end
end
