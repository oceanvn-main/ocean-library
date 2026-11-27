classdef (Abstract) ISimulator < handle
    % ISIMULATOR  Abstract observable engine for time-stepping stochastic processes.
    %
    % This is the bridge between the static Math Engine (Layer 2) and the
    % dynamic Visualization/GUI layers. It encapsulates an IStochasticProcess,
    % maintains a current "Time" and "State", and broadcasts events whenever
    % the state changes.
    %
    % Design Principle (Observer Pattern):
    %   The simulator knows nothing about rendering, plotting, or GUIs. It
    %   simply ticks forward and shouts "I changed!" Any UI component or
    %   data recording component (like SimulationHistory) can listen.
    %
    % Events:
    %   StateChanged - Fired immediately after the current state updates.
    %                  Passes a StateChangedEventArgs object containing
    %                  the old state, the new state, and the current time.
    %
    % Contract:
    %   step(obj)      - Advance time by 1 unit, sample next state, fire event.
    %   reset(obj)     - Return time to 0, clear current state.
    %
    % See also: OceanMath.Simulation.MarkovSimulator
    %           OceanMath.Simulation.SimulationHistory
    %           OceanMath.Information.Processes.IStochasticProcess

    % ------------------------------------------------------------------ %
    events
        % STATECHANGED  Fired after every successful step().
        % Payload: StateChangedEventArgs with OldState, NewState, Time.
        StateChanged
    end

    % ------------------------------------------------------------------ %
    properties (SetAccess = protected)
        % Process_ : the IStochasticProcess being simulated
        % Time_    : integer, current time step t >= 0
        % State_   : the current realized state index
        Process_
        Time_    (1,1) double {mustBeNonnegative, mustBeInteger} = 0
        State_
    end

    % ------------------------------------------------------------------ %
    methods
        function setProcess(obj, process)
            % SETPROCESS  Inject or swap the stochastic process. Also resets.
            arguments
                obj
                process OceanMath.Information.Processes.IStochasticProcess
            end
            obj.Process_ = process;
            obj.reset();
        end

        function t = getTime(obj)
            % GETTIME  Return current time step t.
            t = obj.Time_;
        end

        function s = getCurrentState(obj)
            % GETCURRENTSTATE  Return the current realized state index.
            s = obj.State_;
        end
    end

    % ------------------------------------------------------------------ %
    methods (Abstract)
        % STEP  Advance the simulation by one discrete time unit.
        %
        % Concrete classes must:
        %   1. Sample the next state from the process.
        %   2. Increment Time_.
        %   3. Call notifyStateChange_(oldState, newState, time).
        step(obj)

        % RESET  Return the simulation to t=0.
        reset(obj)
    end

    % ------------------------------------------------------------------ %
    methods (Access = protected)
        function notifyStateChange_(obj, oldState, newState, time)
            % NOTIFYSTATECHANGE_  Bundle event args and fire 'StateChanged'.
            eventData = OceanMath.Simulation.StateChangedEventArgs( ...
                oldState, newState, time);
            notify(obj, 'StateChanged', eventData);
        end
    end
end
