classdef (Abstract) IStepSize < OceanStats.RL.TD.ITDComponent
    % ISTEPSIZE  Strategy: the step size alpha for an update, as a function of the visit count n >= 1 of the
    % entry being updated (state for prediction, state-action pair for control).
    methods (Abstract)
        a = alpha(obj, n)
    end
end
