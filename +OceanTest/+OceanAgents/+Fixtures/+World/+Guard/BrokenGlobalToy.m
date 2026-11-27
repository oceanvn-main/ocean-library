classdef BrokenGlobalToy < OceanTest.OceanAgents.Fixtures.World.Guard.GoodToy
    % BROKENGLOBALTOY  Violates exactly TM-4: sample also draws once from MATLAB's GLOBAL stream (deliberately; the
    % samples themselves still come from the module's own stream, so TM-3 holds).
    methods
        function x = sample(obj)
            x = sample@OceanTest.OceanAgents.Fixtures.World.Guard.GoodToy(obj);
            rand();                                             % deliberate defect for the TM-4 probe
        end
    end
end
