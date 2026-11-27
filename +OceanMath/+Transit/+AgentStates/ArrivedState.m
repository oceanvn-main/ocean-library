classdef ArrivedState < OceanMath.Transit.AgentStates.IAgentState
    % ARRIVEDSTATE  Agent has reached the terminal station (no-op).
    %
    % All queries return terminal values. step() does nothing.
    % This is a terminal state — no transitions out.
    %
    % See also: OceanMath.Transit.AgentStates.IAgentState

    methods
        function step(~, ~, ~, ~)
            % No-op: journey is complete.
        end

        function pos = getPosition(~, agent)
            lastId = agent.Route_(end);
            st = agent.Network_.getStation(lastId);
            pos = st.Position;
        end

        function T = getTangent(~, ~)
            T = [0, 0];  % stationary
        end

        function p = getProgress(~, ~)
            p = 1.0;
        end

        function tf = isArrived(~)
            tf = true;
        end

        function name = getName(~)
            name = "arrived";
        end
    end
end
