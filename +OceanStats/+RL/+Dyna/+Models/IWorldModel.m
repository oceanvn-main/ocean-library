classdef (Abstract) IWorldModel < handle
    % IWORLDMODEL Contract for finite tabular Dyna world models.

    properties (SetAccess = immutable)
        NumStates (1,1) double
        NumActions (1,1) double
        ValidActionMask (:,:) logical
    end

    properties (SetAccess = protected)
        Version (1,1) uint64 = uint64(0)
    end

    methods (Abstract)
        event = update(obj, transition)
        tf = contains(obj, stateIndex, actionIndex)
        pairs = observedPairs(obj)
        outcome = sample(obj, stateIndex, actionIndex, stream)
        outcomes = distribution(obj, stateIndex, actionIndex)
        pred = predecessors(obj, nextStateIndex)
        snapshot = inspect(obj, stateIndex, actionIndex)
        reset(obj)
    end

    methods
        function tf = isReadOnly(~), tf = false; end
        function mode = planningBackupMode(~), mode = "sample"; end
        function id = pairID(obj, stateIndex, actionIndex)
            obj.validatePair(stateIndex, actionIndex, false);
            id = sub2ind([obj.NumStates, obj.NumActions], stateIndex, actionIndex);
        end
    end

    methods (Access = protected)
        function obj = IWorldModel(numStates, numActions, validActionMask)
            arguments
                numStates (1,1) double {mustBeInteger,mustBePositive}
                numActions (1,1) double {mustBeInteger,mustBePositive}
                validActionMask (:,:) logical
            end
            if ~isequal(size(validActionMask), [numStates, numActions])
                error('OceanStats:RL:Dyna:Models:MaskSize', ...
                    'ValidActionMask must have size NumStates-by-NumActions.');
            end
            obj.NumStates = numStates;
            obj.NumActions = numActions;
            obj.ValidActionMask = validActionMask;
        end

        function validatePair(obj, stateIndex, actionIndex, requireValid)
            arguments
                obj
                stateIndex (1,1) double {mustBeInteger,mustBePositive}
                actionIndex (1,1) double {mustBeInteger,mustBePositive}
                requireValid (1,1) logical = true
            end
            if stateIndex > obj.NumStates || actionIndex > obj.NumActions
                error('OceanStats:RL:Dyna:Models:PairOutOfRange', ...
                    'State/action index exceeds model dimensions.');
            end
            if requireValid && ~obj.ValidActionMask(stateIndex, actionIndex)
                error('OceanStats:RL:Dyna:Models:InvalidAction', ...
                    'Action %d is invalid in state %d.', actionIndex, stateIndex);
            end
        end
    end
end
