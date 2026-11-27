classdef EpsilonGreedyQPolicy < OceanAgents.Core.IPolicy
    % EPSILONGREEDYQPOLICY  epsilon-greedy over an agent's own action-value table Q(s, a):
    %   pi(a | s) = eps / |L|  +  (1 - eps) * 1[a in G] / |G|,     0 for a outside L,
    % where L is the set of actions legal in the current VIEW (its ActionMask, never the table's own valid mask) and
    % G = argmax_{a in L} Q(row(s), a) is the set of greedy actions; ties split the greedy mass evenly.
    %
    %   q   = OceanStats.RL.TD.Values.TabularQ(true(48, 4));
    %   pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.1, ...
    %             'StateCodec', OceanAgents.Artifacts.DiscreteStateCodec(48));
    %   p   = pol.probabilities(view)     % no randomness consumed
    %   d   = pol.act(view, stream)       % exactly one rand(stream): inverse-CDF sample of p; Likelihood = p(chosen)
    %
    % q is a HANDLE: the agent's learner (OneStepQLearner) holds the same table and updates it; the policy only reads
    % it. That sharing is intended and stays inside one agent. labels(k) is the external label of Q's column k.
    % Decision.PolicyVersion = total number of Q updates so far (sum of q.Counts), so a decision records which
    % version of the table produced it.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md sections 6.4 and 9.3; metadata in meta/EpsilonGreedyQPolicy.meta.json.

    properties (SetAccess = private)
        Q                       % OceanStats.RL.TD.Values.TabularQ (shared with this agent's learner)
        Epsilon (1,1) double    % exploration probability in [0, 1]
        ActionCodec             % OceanAgents.Artifacts.ActionCodec: Q column <-> external label
        StateCodec              % OceanAgents.Artifacts.DiscreteStateCodec: observation -> Q row
    end

    methods
        function obj = EpsilonGreedyQPolicy(q, labels, options)
            arguments
                q (1,1) OceanStats.RL.TD.Values.TabularQ
                labels {mustBeNumeric, mustBeReal}
                options.Epsilon (1,1) double {mustBeBetween(options.Epsilon, 0, 1)} = 0.1
                options.StateCodec = []
            end
            codec = OceanAgents.Artifacts.ActionCodec(labels);
            [S, A] = size(q.Q);
            if A ~= codec.numActions()
                error('OceanAgents:Policies:Incompatible', 'Q has %d columns but %d action labels were given.', ...
                    A, codec.numActions());
            end
            states = options.StateCodec;
            if isempty(states), states = OceanAgents.Artifacts.DiscreteStateCodec(S); end
            if ~isa(states, 'OceanAgents.Artifacts.DiscreteStateCodec') || states.N ~= S
                error('OceanAgents:Policies:Incompatible', 'StateCodec must be a DiscreteStateCodec over the %d rows of Q.', S);
            end
            obj.Q = q;
            obj.Epsilon = options.Epsilon;
            obj.ActionCodec = codec;
            obj.StateCodec = states;
        end

        function d = describe(obj)
            d = struct('Name', "EpsilonGreedyQPolicy", 'Version', "1", 'NumActions', obj.ActionCodec.numActions(), ...
                'ProvidesLikelihood', true, 'ActionLabels', obj.ActionCodec.Labels, 'Epsilon', obj.Epsilon);
        end

        function checkCompatible(obj, obsSpace, actSpace, ~)
            % CHECKCOMPATIBLE  Discrete spaces whose states match the StateCodec and whose labels match the Q columns.
            if ~obj.StateCodec.matchesSpace(obsSpace)
                error('OceanAgents:Policies:Incompatible', 'Observation space does not match the Q rows (%d states from %d).', ...
                    obj.StateCodec.N, obj.StateCodec.Start);
            end
            if ~isa(actSpace, 'OceanStats.RL.Env.Spaces.Discrete') || ...
                    ~obj.ActionCodec.sameLabelSet(actSpace.Start:(actSpace.Start + actSpace.N - 1))
                error('OceanAgents:Policies:Incompatible', 'Action space does not match the Q column labels %s.', ...
                    mat2str(obj.ActionCodec.Labels));
            end
        end

        function p = probabilities(obj, view)
            % PROBABILITIES  pi(. | s) over the VIEW's columns (A0 section 9.3). Consumes no randomness.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
            end
            legal = view.ActionMask;
            if ~any(legal)
                error('OceanAgents:Core:IllegalAction', 'No legal action in a terminal view (agent %s).', view.AgentID);
            end
            if ~obj.ActionCodec.sameLabelSet(view.ActionLabels)
                error('OceanAgents:Policies:Incompatible', 'View action labels %s differ from the Q column labels %s.', ...
                    mat2str(view.ActionLabels), mat2str(obj.ActionCodec.Labels));
            end
            row = obj.StateCodec.encode(view.Observation);
            qView = obj.Q.estimates(row);                        % Q(row, :) in Q-column order
            qView = qView(obj.ActionCodec.column(view.ActionLabels));   % ... reordered to the view's columns
            greedy = legal & (qView == max(qView(legal)));       % G: argmax over L only (ties kept)
            p = zeros(1, numel(legal));
            p(legal) = obj.Epsilon / nnz(legal);
            p(greedy) = p(greedy) + (1 - obj.Epsilon) / nnz(greedy);
        end

        function dec = act(obj, view, stream)
            % ACT  One rand(stream) draw u; chosen column = smallest k with u < cumsum(p)(k).
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
                stream (1,1) RandStream
            end
            p = obj.probabilities(view);
            k = sampleInverseCdf_(p, rand(stream));
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', view.ActionLabels(k), 'Probabilities', p, ...
                'Likelihood', p(k), 'PolicyVersion', sum(obj.Q.Counts(:)), ...
                'Provenance', struct('Policy', "EpsilonGreedyQPolicy", 'Epsilon', obj.Epsilon));
        end

        function setEpsilon(obj, epsilon)
            % SETEPSILON  Change the exploration probability (e.g. a schedule driven by the experiment script).
            arguments
                obj
                epsilon (1,1) double {mustBeBetween(epsilon, 0, 1)}
            end
            obj.Epsilon = epsilon;
        end

        function s = exportState(obj)
            % EXPORTSTATE  The policy's own setting. Q and its counts are the learner's trainable state and are
            % exported by OneStepQLearner.exportState (one owner, no double bookkeeping).
            s = struct('Name', "EpsilonGreedyQPolicy", 'Epsilon', obj.Epsilon);
        end

        function importState(obj, s)
            arguments
                obj
                s (1,1) struct
            end
            if ~isfield(s, 'Name') || s.Name ~= "EpsilonGreedyQPolicy" || ~isfield(s, 'Epsilon')
                error('OceanAgents:Policies:Incompatible', 'State is not an EpsilonGreedyQPolicy state.');
            end
            obj.setEpsilon(s.Epsilon);
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3: ActorView in, Decision out; reads a TabularQ.
            spec = struct();
            spec.ComponentName = "EpsilonGreedyQPolicy";
            spec.Inputs = struct('Name', "view", 'Domain', "Agent-ActorView", 'Dtype', "OceanAgents.Core.ActorView", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "decision", 'Domain', "Agent-Decision", 'Dtype', "OceanAgents.Core.Decision", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "epsgreedy");
            spec.ComposesFrom = {"OceanStats.RL.TD.Values.TabularQ"};
        end
    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Pillar 1: pedagogical metadata from meta/EpsilonGreedyQPolicy.meta.json plus the
            % OceanAgents pillar fields (SymbolicLatex, Symbols, IsSymbolic, Reference; A0 section 8).
            here = fileparts(mfilename('fullpath'));
            meta = jsondecode(fileread(fullfile(here, 'meta', 'EpsilonGreedyQPolicy.meta.json')));
            meta.SymbolicLatex = string(meta.equations.policy.latex);
            meta.Assumptions = meta.pedagogy.assumptions;
            meta.IsSymbolic = true;
            meta.Reference = string(meta.textbook.reference);
        end
    end
end
