classdef UpdateEvent
    %UPDATEEVENT Complete operands and result of one committed Q update.
    properties (SetAccess = immutable)
        Origin (1,1) string
        StateIndex (1,1) double
        ActionIndex (1,1) double
        OldQ (1,1) double
        Target (1,1) double
        TDError (1,1) double
        Alpha (1,1) double
        NewQ (1,1) double
        PlanningPriority (1,1) double
        RealStep (1,1) double
        PlanningStep (1,1) double
        ModelVersion (1,1) double
    end
    methods
        function obj = UpdateEvent(options)
            arguments
                options.Origin
                options.StateIndex
                options.ActionIndex
                options.OldQ
                options.Target
                options.TDError
                options.Alpha
                options.NewQ
                options.PlanningPriority = 0
                options.RealStep = 0
                options.PlanningStep = 0
                options.ModelVersion = 0
            end
            needed = ["Origin" "StateIndex" "ActionIndex" "OldQ" "Target" "TDError" "Alpha" "NewQ"];
            for n = needed, if ~isfield(options, n), fail_(n + " is required."); end, end
            obj.Origin = string(options.Origin);
            if ~isscalar(obj.Origin) || ~any(obj.Origin == ["real" "planning" "exact"]), fail_("Origin is invalid."); end
            obj.StateIndex = uint_(options.StateIndex, false, "StateIndex");
            obj.ActionIndex = uint_(options.ActionIndex, false, "ActionIndex");
            obj.OldQ = finite_(options.OldQ, "OldQ"); obj.Target = finite_(options.Target, "Target");
            obj.TDError = finite_(options.TDError, "TDError"); obj.Alpha = finite_(options.Alpha, "Alpha");
            if obj.Alpha <= 0 || obj.Alpha > 1, fail_("Alpha must be in (0, 1]."); end
            obj.NewQ = finite_(options.NewQ, "NewQ");
            obj.PlanningPriority = finite_(options.PlanningPriority, "PlanningPriority");
            if obj.PlanningPriority < 0, fail_("PlanningPriority must be nonnegative."); end
            obj.RealStep = uint_(options.RealStep, true, "RealStep");
            obj.PlanningStep = uint_(options.PlanningStep, true, "PlanningStep");
            obj.ModelVersion = uint_(options.ModelVersion, true, "ModelVersion");
        end
    end
end
function x = finite_(x, name)
if ~(isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x)), fail_(name + " must be a finite real scalar."); end
x = double(x);
end
function x = uint_(x, zero, name), x = finite_(x, name); if x ~= fix(x) || x < double(~zero), fail_(name + " has an invalid integer value."); end, end
function fail_(message), error('OceanStats:RL:Dyna:Core:InvalidRecord', '%s', message); end
