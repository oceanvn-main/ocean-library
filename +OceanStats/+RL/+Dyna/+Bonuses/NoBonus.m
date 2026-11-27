classdef NoBonus < OceanStats.RL.Dyna.Bonuses.IPlanningBonus
    % NOBONUS Null-object planning bonus for ordinary Dyna-Q.
    methods
        function bonus=evaluate(~,~,~,~,~), bonus=0; end
        function onRealVisit(~,~,~,~), end
        function reset(~), end
    end
end
