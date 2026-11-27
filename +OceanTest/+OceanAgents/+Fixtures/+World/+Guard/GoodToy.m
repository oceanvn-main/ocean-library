classdef GoodToy < OceanTest.OceanAgents.Fixtures.World.Guard.ToyModule
    % GOODTOY  A correct ToyModule: satisfies TM-1..TM-4 (TM-5 is assumed).
    properties (Access = protected)
        Stream = []
    end

    methods
        function c = capacity(~)
            c = 4;
        end

        function reset(obj, seed)
            obj.Stream = RandStream('mrg32k3a', 'Seed', seed);
        end

        function x = sample(obj)
            if isempty(obj.Stream), obj.reset(0); end
            x = rand(obj.Stream, 1, 3);
        end

        function y = apply(~, x)
            y = sort(x);
        end
    end
end
