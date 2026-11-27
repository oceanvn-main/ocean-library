classdef (Abstract) ISpace
    % ISPACE  Describes the set of valid observations or actions of an environment (Gymnasium "space" / dm_env "spec").
    % Algorithms introspect the space instead of assuming an indexing convention.
    %
    %   tf = space.contains(x)          is x a member?
    %   x  = space.sample([stream])     uniform draw using a RandStream (default: the global stream)
    %   space.Shape                      size of one element, e.g. [1 1] for a scalar
    %   space.describe()                 short text
    %
    % A space is a value object (no state, no randomness of its own).
    properties (Abstract, SetAccess = protected)
        Shape
    end
    methods (Abstract)
        tf = contains(obj, x)
        x = sample(obj, stream)
        s = describe(obj)
    end
    methods (Access = protected, Static)
        function stream = stream_(stream)
            if nargin < 1 || isempty(stream), stream = RandStream.getGlobalStream(); end
        end
    end
end
