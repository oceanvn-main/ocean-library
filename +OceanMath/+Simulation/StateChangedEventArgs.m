classdef StateChangedEventArgs < event.EventData
    % STATECHANGEDEVENTARGS  Custom event payload for ISimulator state changes.
    %
    % Carries the complete context of what happened in a single simulation
    % step. This object is constructed by ISimulator and passed to all
    % listeners (GUI callbacks, SimulationHistory) when 'StateChanged' fires.
    %
    % Properties are SetAccess = immutable — the payload cannot be modified
    % after construction, ensuring thread-safety and predictable behavior.
    %
    % See also: OceanMath.Simulation.ISimulator
    %           OceanMath.Simulation.SimulationHistory

    properties (SetAccess = immutable)
        OldState  % State index before this step (t-1)
        NewState  % State index after this step  (t)
        Time      % The new time step value       (t)
    end

    methods
        function obj = StateChangedEventArgs(oldState, newState, time)
            obj.OldState = oldState;
            obj.NewState = newState;
            obj.Time     = time;
        end
    end
end
