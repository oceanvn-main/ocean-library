classdef ModelEvent
    %MODELEVENT Immutable explanation of one model mutation.
    properties (SetAccess = immutable)
        StateIndex (1,1) double
        ActionIndex (1,1) double
        ObservationCount (1,1) double
        PreviousOutcomes struct
        NewOutcomes struct
        PredecessorChanges struct
        ModelVersion (1,1) double
    end
    methods
        function obj = ModelEvent(options)
            arguments
                options.StateIndex
                options.ActionIndex
                options.ObservationCount
                options.PreviousOutcomes = struct([])
                options.NewOutcomes = struct([])
                options.PredecessorChanges = struct()
                options.ModelVersion
            end
            needed = ["StateIndex" "ActionIndex" "ObservationCount" "ModelVersion"];
            for n = needed, if ~isfield(options, n), fail_(n + " is required."); end, end
            obj.StateIndex = uint_(options.StateIndex, false, "StateIndex");
            obj.ActionIndex = uint_(options.ActionIndex, false, "ActionIndex");
            obj.ObservationCount = uint_(options.ObservationCount, false, "ObservationCount");
            if ~isstruct(options.PreviousOutcomes), fail_("PreviousOutcomes must be a struct array."); end
            if ~isstruct(options.NewOutcomes), fail_("NewOutcomes must be a struct array."); end
            if ~(isstruct(options.PredecessorChanges) && isscalar(options.PredecessorChanges)), fail_("PredecessorChanges must be a scalar struct."); end
            obj.PreviousOutcomes = options.PreviousOutcomes;
            obj.NewOutcomes = options.NewOutcomes;
            obj.PredecessorChanges = options.PredecessorChanges;
            obj.ModelVersion = uint_(options.ModelVersion, false, "ModelVersion");
        end
    end
end
function x = uint_(x, zero, name)
if ~(isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x) && x == fix(x) && x >= double(~zero)), fail_(name + " has an invalid integer value."); end
x = double(x);
end
function fail_(message), error('OceanStats:RL:Dyna:Core:InvalidRecord', '%s', message); end
