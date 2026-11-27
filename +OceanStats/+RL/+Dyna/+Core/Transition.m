classdef Transition
    %TRANSITION Immutable real or simulated Dyna transition.

    properties (SetAccess = immutable)
        StateIndex (1,1) double
        ActionIndex (1,1) double
        Reward (1,1) double
        NextStateIndex (1,1) double
        Terminated (1,1) logical
        Truncated (1,1) logical
        Origin (1,1) string
        RealStep (1,1) double
        PlanningStep (1,1) double
    end

    methods
        function obj = Transition(options)
            arguments
                options.StateIndex
                options.ActionIndex
                options.Reward
                options.NextStateIndex
                options.Terminated = false
                options.Truncated = false
                options.Origin = "real"
                options.RealStep = 0
                options.PlanningStep = 0
            end
            require_(options, ["StateIndex" "ActionIndex" "Reward" "NextStateIndex"]);
            obj.StateIndex = positiveInteger_(options.StateIndex, "StateIndex");
            obj.ActionIndex = positiveInteger_(options.ActionIndex, "ActionIndex");
            obj.Reward = finiteScalar_(options.Reward, "Reward");
            obj.NextStateIndex = nonnegativeInteger_(options.NextStateIndex, "NextStateIndex");
            obj.Terminated = flag_(options.Terminated, "Terminated");
            obj.Truncated = flag_(options.Truncated, "Truncated");
            if obj.Terminated && obj.Truncated
                fail_("Terminated and Truncated cannot both be true.");
            end
            if ~obj.Terminated && obj.NextStateIndex == 0
                fail_("NextStateIndex may be zero only for a terminated transition.");
            end
            if obj.Truncated && obj.NextStateIndex == 0
                fail_("A truncated transition requires an indexable final observation.");
            end
            obj.Origin = string(options.Origin);
            if ~isscalar(obj.Origin) || ~any(obj.Origin == ["real" "planning" "exact"])
                fail_("Origin must be real, planning, or exact.");
            end
            if obj.Truncated && obj.Origin ~= "real"
                fail_("Only a real transition may be truncated.");
            end
            obj.RealStep = nonnegativeInteger_(options.RealStep, "RealStep");
            obj.PlanningStep = nonnegativeInteger_(options.PlanningStep, "PlanningStep");
        end
    end
end

function require_(options, names)
for name = names
    if ~isfield(options, name), fail_(name + " is required."); end
end
end
function x = finiteScalar_(x, name)
if ~(isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x)), fail_(name + " must be a finite real scalar."); end
x = double(x);
end
function x = positiveInteger_(x, name)
x = nonnegativeInteger_(x, name);
if x == 0, fail_(name + " must be positive."); end
end
function x = nonnegativeInteger_(x, name)
x = finiteScalar_(x, name);
if x < 0 || x ~= fix(x), fail_(name + " must be a nonnegative integer."); end
end
function x = flag_(x, name)
if ~(islogical(x) && isscalar(x)), fail_(name + " must be a scalar logical."); end
end
function fail_(message)
error('OceanStats:RL:Dyna:Core:InvalidRecord', '%s', message);
end
