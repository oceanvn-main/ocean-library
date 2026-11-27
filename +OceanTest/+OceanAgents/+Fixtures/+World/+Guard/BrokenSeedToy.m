classdef BrokenSeedToy < OceanTest.OceanAgents.Fixtures.World.Guard.GoodToy
    % BROKENSEEDTOY  Violates exactly TM-3: reset(seed) mixes in a call counter, so the same seed does not
    % reproduce the same samples.
    properties (Access = private)
        Calls double = 0
    end
    methods
        function reset(obj, seed)
            obj.Calls = obj.Calls + 1;
            obj.Stream = RandStream('mrg32k3a', 'Seed', seed + obj.Calls);
        end
    end
end
