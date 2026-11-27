classdef (Abstract) IPlanningBonus < handle
    % IPLANNINGBONUS Reward adjustment applied only to simulated backups.
    methods (Abstract)
        bonus = evaluate(obj, stateIndex, actionIndex, currentRealStep, lastRealVisit)
        onRealVisit(obj, stateIndex, actionIndex, currentRealStep)
        reset(obj)
    end
end
