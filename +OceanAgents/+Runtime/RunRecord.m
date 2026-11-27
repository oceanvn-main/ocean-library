classdef RunRecord
    % RUNRECORD  What a runner reports about one run() call (value record; a run never throws for callback faults,
    % it returns Status "Failed" with the stage and the exception instead).
    %
    %   rec.Status          "Completed" | "Failed"
    %   rec.NumEpisodes     episodes that ran to their end
    %   rec.EpisodeReturns  E x N undiscounted sums of valid rewards (row e = completed episode e, column = agent)
    %   rec.EpisodeLengths  E x 1 environment steps per completed episode
    %   rec.StepCalls       total environment step() calls made (including a call that threw)
    %   rec.UpdateCounts    1 x N parameter updates reported by each agent's learner during this run
    %   rec.FailureStage    "" | "reset" | "decide" | "validate" | "step" | "observe"
    %   rec.Failure         the MException, or [] when completed
    %   rec.AgentIDs        1 x N agent ids (column order of EpisodeReturns / UpdateCounts)
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 4.6.

    properties (SetAccess = immutable)
        Status string = "Completed"
        NumEpisodes double = 0
        EpisodeReturns double = zeros(0, 0)
        EpisodeLengths double = zeros(0, 1)
        StepCalls double = 0
        UpdateCounts double = zeros(1, 0)
        FailureStage string = ""
        Failure = []
        AgentIDs string = strings(1, 0)
    end

    properties (Constant)
        Stages = ["reset" "decide" "validate" "step" "observe"]
    end

    methods
        function obj = RunRecord(options)
            arguments
                options.Status (1,1) string {mustBeMember(options.Status, ["Completed" "Failed"])} = "Completed"
                options.AgentIDs (1,:) string = strings(1, 0)
                options.EpisodeReturns double = []
                options.EpisodeLengths (:,1) double {mustBeNonnegative, mustBeInteger} = zeros(0, 1)
                options.StepCalls (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                options.UpdateCounts (1,:) double {mustBeNonnegative, mustBeInteger} = zeros(1, 0)
                options.FailureStage (1,1) string = ""
                options.Failure = []
            end
            n = numel(options.AgentIDs);
            e = numel(options.EpisodeLengths);
            returns = options.EpisodeReturns;
            if isempty(returns), returns = zeros(e, n); end
            if ~isequal(size(returns), [e n])
                error('OceanAgents:Core:InvalidRecord', 'EpisodeReturns must be %d x %d.', e, n);
            end
            updates = options.UpdateCounts;
            if isempty(updates), updates = zeros(1, n); end
            if numel(updates) ~= n
                error('OceanAgents:Core:InvalidRecord', 'UpdateCounts must have one entry per agent.');
            end
            failed = options.Status == "Failed";
            if failed ~= (options.FailureStage ~= "") || failed ~= ~isempty(options.Failure)
                error('OceanAgents:Core:InvalidRecord', ...
                    'A Failed record needs a FailureStage and an MException; a Completed one has neither.');
            end
            if failed && ~(any(options.FailureStage == obj.Stages) && isa(options.Failure, 'MException'))
                error('OceanAgents:Core:InvalidRecord', 'FailureStage must be one of %s and Failure an MException.', ...
                    strjoin(obj.Stages, ', '));
            end
            obj.Status = options.Status;
            obj.AgentIDs = options.AgentIDs;
            obj.NumEpisodes = e;
            obj.EpisodeReturns = returns;
            obj.EpisodeLengths = options.EpisodeLengths;
            obj.StepCalls = options.StepCalls;
            obj.UpdateCounts = updates;
            obj.FailureStage = options.FailureStage;
            obj.Failure = options.Failure;
        end

        function tf = failed(obj)
            tf = obj.Status == "Failed";
        end
    end
end
