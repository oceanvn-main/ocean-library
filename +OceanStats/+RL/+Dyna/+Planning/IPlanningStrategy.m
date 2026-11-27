classdef (Abstract) IPlanningStrategy < handle
    % IPLANNINGSTRATEGY Selects modeled state-action pairs for Dyna backups.
    methods (Abstract)
        observeReal(obj, transition, model, values, gamma)
        tf = canPlan(obj, model)
        candidate = next(obj, model, values, gamma, stream)
        afterUpdate(obj, event, model, values, gamma)
        reset(obj)
        stats = inspect(obj)
    end
end
