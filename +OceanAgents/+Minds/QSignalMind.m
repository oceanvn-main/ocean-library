classdef QSignalMind < OceanAgents.Minds.SignalMind
    % QSIGNALMIND  The A1 epsilon-greedy Q agent as a signal mind: its values are a TableMemory, its learner lives in
    % the mind, and every action reaches the world through its body (AB1 section 7). With an empty body or a robot
    % body that has the same moves, a run is identical to the A1 PolicyMind agent (AB1 success tests S1, S2).
    %
    %   q = OceanStats.RL.TD.Values.TabularQ(env.ValidMask, 0);
    %   mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
    %   mind.addLearner(OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec), "q");
    %   OceanAgents.Embodied.Dualism.form(mind, OceanAgents.Body.EmptyBody(string(1:4)));
    %   host = OceanAgents.Runtime.AgentHost("agent1", mind, 'Gamma', 1);

    properties (SetAccess = private)
        Policy
        Values
        Labels string
    end

    properties (Access = private)
        RawLabels_
    end

    methods
        function obj = QSignalMind(q, labels, options)
            arguments
                q (1,1) OceanStats.RL.TD.Values.TabularActionValues
                labels
                options.Epsilon (1,1) double = 0.1
                options.StateCodec = []
                options.MemoryID (1,1) string = "values"
            end
            args = {'Epsilon', options.Epsilon};
            if ~isempty(options.StateCodec), args = [args, {'StateCodec', options.StateCodec}]; end
            obj.Policy = OceanAgents.Policies.EpsilonGreedyQPolicy(q, labels, args{:});
            obj.addPolicy("egreedy", obj.Policy);
            obj.RawLabels_ = labels;
            obj.Labels = reshape(string(labels), 1, []);
            obj.Values = OceanAgents.Memory.TableMemory(options.MemoryID, q, obj.Labels);
            obj.addMemory(obj.Values);
            obj.CommandType = OceanAgents.Body.IBody.moveType(obj.Labels);
            % the mind sees the world only through these (AB2): the cell as a digit, what the Laws permit, behaviors
            obj.addPerception(OceanAgents.Minds.Perception.DigitPerception());
            obj.addPerception(OceanAgents.Minds.Perception.MovesPerception(obj.Labels));
            obj.addPerception(OceanAgents.Minds.Perception.BehaviorPerception());
        end

        function d = describe(obj)
            d = struct('Name', "QSignalMind", 'Profile', "signal-mind", ...
                'Components', struct('Policy', obj.Policy.describe()));
        end

        function checkCompatible(obj, obsSpace, actSpace, role)
            obj.Policy.checkCompatible(obsSpace, actSpace, role);
        end

        function beginEpisode(obj, view)
            beginEpisode@OceanAgents.Minds.SignalMind(obj, view);
            obj.beginPolicies_(view);                            % bookkeeping only, never the world (AB2 I10)
        end

        function dec = decide(obj, view, stream)
            dec = obj.decideWith_(view, stream, obj.Labels, obj.RawLabels_);
        end

        function observe(~, ~)
            % OBSERVE  The world's feedback is not passed to the policy (AB2 I10); learning goes through learn().
        end

        function endEpisode(obj)
            obj.endPolicies_();
            endEpisode@OceanAgents.Minds.SignalMind(obj);
        end

        function s = exportState(obj)
            ls = cellfun(@(l) l.exportState(), obj.Learners, 'UniformOutput', false);
            s = struct('Name', "QSignalMind", 'Policy', obj.Policy.exportState(), 'Learners', {ls});
        end

        function importState(obj, s)
            if ~isfield(s, 'Name') || s.Name ~= "QSignalMind"
                error('OceanAgents:Policies:Incompatible', 'State is not a QSignalMind state.');
            end
            obj.Policy.importState(s.Policy);
            for k = 1:numel(obj.Learners), obj.Learners{k}.importState(s.Learners{k}); end
        end
    end
end
