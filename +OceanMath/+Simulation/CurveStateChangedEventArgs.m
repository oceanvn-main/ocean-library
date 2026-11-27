classdef CurveStateChangedEventArgs < event.EventData
    % CURVESTATECHANGEDEVENTARGS  Event payload for CurveSimulator state changes.
    %
    % Carries the old and new KinematicState snapshots after each step().
    % Listeners receive the complete physical context of the transition.
    %
    % See also: OceanMath.Simulation.CurveSimulator
    %           OceanMath.Simulation.KinematicState

    properties (SetAccess = immutable)
        OldState  % KinematicState before this step
        NewState  % KinematicState after this step
    end

    methods
        function obj = CurveStateChangedEventArgs(oldState, newState)
            obj.OldState = oldState;
            obj.NewState = newState;
        end
    end
end
