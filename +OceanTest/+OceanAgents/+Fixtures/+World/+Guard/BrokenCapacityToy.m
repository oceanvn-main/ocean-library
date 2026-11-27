classdef BrokenCapacityToy < OceanTest.OceanAgents.Fixtures.World.Guard.GoodToy
    % BROKENCAPACITYTOY  Violates exactly TM-1 (declares capacity 0); everything else is GoodToy.
    methods
        function c = capacity(~)
            c = 0;
        end
    end
end
