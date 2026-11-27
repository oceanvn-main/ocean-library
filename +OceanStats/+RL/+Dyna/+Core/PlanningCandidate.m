classdef PlanningCandidate
    %PLANNINGCANDIDATE Immutable work item selected by a planning strategy.
    properties (SetAccess = immutable)
        StateIndex (1,1) double
        ActionIndex (1,1) double
        PairID (1,1) double
        Priority (1,1) double
        ModelVersion (1,1) double
    end
    methods
        function obj = PlanningCandidate(stateIndex, actionIndex, pairID, priority, modelVersion)
            arguments
                stateIndex
                actionIndex
                pairID
                priority = 0
                modelVersion = 0
            end
            obj.StateIndex = uint_(stateIndex, false, "StateIndex");
            obj.ActionIndex = uint_(actionIndex, false, "ActionIndex");
            obj.PairID = uint_(pairID, false, "PairID");
            obj.Priority = finite_(priority, "Priority");
            if obj.Priority < 0, fail_("Priority must be nonnegative."); end
            obj.ModelVersion = uint_(modelVersion, true, "ModelVersion");
        end
    end
end
function x = finite_(x, name)
if ~(isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x)), fail_(name + " must be a finite real scalar."); end
x = double(x);
end
function x = uint_(x, zero, name)
x = finite_(x, name);
if x ~= fix(x) || x < double(~zero), fail_(name + " has an invalid integer value."); end
end
function fail_(message), error('OceanStats:RL:Dyna:Core:InvalidRecord', '%s', message); end
