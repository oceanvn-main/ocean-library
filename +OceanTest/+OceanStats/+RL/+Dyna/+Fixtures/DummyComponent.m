classdef DummyComponent < OceanStats.RL.Dyna.IDynaComponent
    properties (Constant)
        Latex = "y=x"
        Reference = "Dyna Phase 1 test fixture"
    end
    methods
        function value = name(~)
            value = "dummy";
        end
    end
end
