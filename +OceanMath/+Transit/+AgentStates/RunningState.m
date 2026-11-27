classdef RunningState < OceanMath.Transit.AgentStates.IAgentState
    % RUNNINGSTATE  Agent is actively moving along a curve segment.
    %
    % Steps the CurveSimulator each tick. Detects segment end (or stall)
    % and transitions to DwellingState or ArrivedState.
    %
    % See also: OceanMath.Transit.AgentStates.IAgentState

    methods
        function step(~, agent, ~, ~)
            % STEP  Advance the CurveSimulator safely using the exact clock cycle duration.
            agent.Simulator_.step(agent.Dt_);
            s = agent.Simulator_.getState();

            % Record history
            agent.recordHistory_(s);

            % Check if segment ended
            if s.Parameter >= 1
                nSeg = agent.getSegmentCount();
                if agent.CurrentSegIdx_ >= nSeg
                    % Last segment — terminal station reached
                    agent.setState(OceanMath.Transit.AgentStates.ArrivedState());
                else
                    % Intermediate station — dwell or advance
                    if agent.DwellTime_ > 0
                        agent.setState(OceanMath.Transit.AgentStates.DwellingState());
                        agent.DwellRemaining_ = agent.DwellTime_;
                    else
                        agent.advanceToNextSegment_();
                    end
                end
            end
        end

        function pos = getPosition(~, agent)
            pos = agent.Simulator_.getState().Position;
        end

        function T = getTangent(~, agent)
            T = agent.Simulator_.getState().Tangent;
        end

        function p = getProgress(~, agent)
            nSeg = agent.getSegmentCount();
            segProgress = agent.Simulator_.getState().Parameter;
            p = (agent.CurrentSegIdx_ - 1 + segProgress) / nSeg;
        end

        function tf = isArrived(~)
            tf = false;
        end

        function name = getName(~)
            name = "running";
        end
    end
end
