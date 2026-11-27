classdef Discrete < OceanStats.RL.Env.Spaces.ISpace
    % DISCRETE  The integers {Start, ..., Start+N-1}. Default Start = 1 (MATLAB indexing), so a Discrete state or action
    % space of size N can index a table directly.
    %   sp = OceanStats.RL.Env.Spaces.Discrete(48);   sp.N, sp.Start, sp.contains(3)
    properties (SetAccess = protected)
        Shape = [1 1]
        N (1,1) double {mustBeInteger, mustBePositive} = 1
        Start (1,1) double {mustBeInteger} = 1
    end
    methods
        function obj = Discrete(n, options)
            arguments
                n (1,1) double {mustBeInteger, mustBePositive}
                options.Start (1,1) double {mustBeInteger} = 1
            end
            obj.N = n;
            obj.Start = options.Start;
        end
        function tf = contains(obj, x)
            tf = isnumeric(x) && isscalar(x) && isfinite(x) && x == fix(x) && x >= obj.Start && x <= obj.Start + obj.N - 1;
        end
        function x = sample(obj, stream)
            if nargin < 2, stream = []; end
            stream = OceanStats.RL.Env.Spaces.ISpace.stream_(stream);
            x = obj.Start + floor(rand(stream) * obj.N);
        end
        function s = describe(obj)
            s = sprintf('Discrete(%d, start=%d)', obj.N, obj.Start);
        end
    end
end
