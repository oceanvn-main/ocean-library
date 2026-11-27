classdef MultiDiscrete < OceanStats.RL.Env.Spaces.ISpace
    % MULTIDISCRETE  A vector of independent Discrete coordinates, coordinate k in {1, ..., NVec(k)}.
    %   sp = OceanStats.RL.Env.Spaces.MultiDiscrete([3 4]);   sp.contains([2 4])
    properties (SetAccess = protected)
        Shape
        NVec (1,:) double
    end
    methods
        function obj = MultiDiscrete(nvec)
            arguments
                nvec (1,:) double {mustBeInteger, mustBePositive, mustBeNonempty}
            end
            obj.NVec = nvec;  obj.Shape = [1 numel(nvec)];
        end
        function tf = contains(obj, x)
            tf = isnumeric(x) && isequal(size(x), obj.Shape) && all(isfinite(x)) && all(x == fix(x)) && all(x >= 1) && all(x <= obj.NVec);
        end
        function x = sample(obj, stream)
            if nargin < 2, stream = []; end
            stream = OceanStats.RL.Env.Spaces.ISpace.stream_(stream);
            x = 1 + floor(rand(stream, 1, numel(obj.NVec)) .* obj.NVec);
        end
        function s = describe(obj)
            s = sprintf('MultiDiscrete(%s)', mat2str(obj.NVec));
        end
    end
end
