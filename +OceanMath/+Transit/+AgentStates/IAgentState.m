classdef (Abstract) IAgentState
    % IAGENTSTATE  Interface for TransitAgent state objects (State Pattern).
    %
    % Each concrete state (RunningState, DwellingState, ArrivedState)
    % implements these methods to encapsulate state-specific behavior.
    % The TransitAgent delegates all state-dependent queries to the
    % active state object, eliminating if-else chains.
    %
    % See also: OceanMath.Transit.AgentStates.RunningState,
    %           OceanMath.Transit.AgentStates.DwellingState,
    %           OceanMath.Transit.AgentStates.ArrivedState

    methods (Abstract)
        % STEP  Advance the agent by one time step.
        %   May trigger state transitions via agent.setState().
        step(obj, agent, dt, realDt)

        % GETPOSITION  Return [x, y] position in this state.
        pos = getPosition(obj, agent)

        % GETTANGENT  Return heading direction vector.
        T = getTangent(obj, agent)

        % GETPROGRESS  Return route completion fraction [0, 1].
        p = getProgress(obj, agent)

        % ISARRIVED  True only for ArrivedState.
        tf = isArrived(obj)

        % GETNAME  Return state name string for display/debug.
        name = getName(obj)
    end
end
