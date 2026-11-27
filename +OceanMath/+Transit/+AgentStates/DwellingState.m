classdef DwellingState < OceanMath.Transit.AgentStates.IAgentState
    % DWELLINGSTATE  Agent is paused at a station (dwell countdown).
    %
    % Decrements DwellRemaining_ each tick. When countdown reaches zero,
    % transitions to RunningState on the next segment.
    %
    % See also: OceanMath.Transit.AgentStates.IAgentState

    methods
        function step(~, agent, ~, realDt)
            % STEP  Count down dwell time, then advance to next segment.
            % Drain by REAL WORLD seconds, completely independent of TimeWarp
            agent.DwellRemaining_ = agent.DwellRemaining_ - realDt;
            if agent.DwellRemaining_ <= 0
                agent.DwellRemaining_ = 0;
                agent.advanceToNextSegment_();
                % advanceToNextSegment_ sets state back to RunningState
            end
        end

        function pos = getPosition(~, agent)
            % Position = the station we just arrived at (destination of current segment)
            stationId = agent.Route_(agent.CurrentSegIdx_ + 1);
            st = agent.Network_.getStation(stationId);
            pos = st.Position;
        end

        function T = getTangent(~, ~)
            T = [0, 0];  % stationary
        end

        function p = getProgress(~, agent)
            nSeg = agent.getSegmentCount();
            p = agent.CurrentSegIdx_ / nSeg;
        end

        function tf = isArrived(~)
            tf = false;
        end

        function name = getName(~)
            name = "dwelling";
        end
    end
end
