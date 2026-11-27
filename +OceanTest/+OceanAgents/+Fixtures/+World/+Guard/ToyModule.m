classdef (Abstract) ToyModule < handle
    % TOYMODULE  Toy module interface certified by the "modules" test door (K16 probe mechanics, W1 scope).
    %
    %   c = m.capacity()       positive integer (TM-1, static)
    %   m.reset(seed)          (re)seed the module's own stream
    %   x = m.sample()         1 x 3 draw from the module's own stream (TM-3 same seed -> same samples; TM-4 never
    %                          touches the global stream)
    %   y = m.apply(x)         finite, same size as x (TM-2)
    % The guard admits any implementation that passes the axioms - it never looks at the class name beyond the
    % door's Admits interface.

    properties
        DrawLog double = zeros(1, 0)     % probe-mechanics tests log the first draw of each probe stream here
    end

    methods (Abstract)
        c = capacity(obj)
        reset(obj, seed)
        x = sample(obj)
        y = apply(obj, x)
    end
end
