classdef BrokenOutputToy < OceanTest.OceanAgents.Fixtures.World.Guard.GoodToy
    % BROKENOUTPUTTOY  Violates exactly TM-2: apply returns NaN for inputs above 0.8 (found only by probing).
    methods
        function y = apply(~, x)
            y = sort(x);
            y(y > 0.8) = NaN;
        end
    end
end
