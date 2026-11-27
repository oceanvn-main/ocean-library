classdef FactorLearner < OceanAgents.Core.ILearner
    % FACTORLEARNER  A learner that learns about ONE part of a factored action (PG1 contract): each transition is
    % passed to the inner learner with only that part of the action (labels 1..m of the part, its permitted values
    % next). The other parts (e.g. a coin's put) are never "learned".
    %
    %   q = OceanStats.RL.TD.Values.TabularQ(true(3, 2), 0);
    %   inner = OceanAgents.Learners.OneStepQLearner(q, 1:2, 'Alpha', 0.1, 'Gamma', 1, 'StateCodec', codec);
    %   L = OceanAgents.Learners.FactorLearner(inner, F, "guess");

    properties (SetAccess = private)
        Inner
        Factors
        Factor string
    end

    properties (Dependent)
        Q                                   % the inner learner's table, when it has one (for memory binding)
        FP                                  % the inner learner's fictitious-play component, when it has one
    end

    methods
        function obj = FactorLearner(inner, factors, factor)
            arguments
                inner (1,1) OceanAgents.Core.ILearner
                factors (1,1) OceanAgents.Core.ActionFactors
                factor (1,1) string
            end
            factors.index(factor);                          % throws for an unknown factor
            obj.Inner = inner;
            obj.Factors = factors;
            obj.Factor = factor;
        end

        function q = get.Q(obj)
            q = [];
            if isprop(obj.Inner, 'Q'), q = obj.Inner.Q; end
        end

        function fp = get.FP(obj)
            fp = [];
            if isprop(obj.Inner, 'FP'), fp = obj.Inner.FP; end
        end

        function d = describe(obj)
            d = obj.Inner.describe();
            d.Name = "FactorLearner(" + string(d.Name) + ", " + obj.Factor + ")";
        end

        function checkCompatible(~, ~)
        end

        function consume(obj, t)
            F = obj.Factors;
            f = F.index(obj.Factor);
            k = find(t.ActionLabels == t.Action, 1);
            s = F.parts(k);
            next = false(1, F.Sizes(f));                     % no next actions (e.g. after the end): nothing permitted
            if ~isempty(t.NextMask)
                if numel(t.NextMask) ~= F.numJoint()
                    error('OceanAgents:Learners:Incompatible', ['FactorLearner: the next mask has %d actions; the ' ...
                        'factors make %d.'], numel(t.NextMask), F.numJoint());
                end
                next = F.partMask(logical(t.NextMask), obj.Factor);
            end
            part = OceanAgents.Core.Transition('AgentID', t.AgentID, 'EpisodeID', t.EpisodeID, ...
                'Observation', t.Observation, 'Action', s(f), 'ActionLabels', 1:F.Sizes(f), 'Reward', t.Reward, ...
                'NextObservation', t.NextObservation, 'NextActionLabels', 1:F.Sizes(f), 'NextMask', next, ...
                'Terminated', t.Terminated, 'Truncated', t.Truncated, 'Duration', t.Duration, 'Discount', t.Discount, ...
                'PolicyVersion', t.PolicyVersion, 'Likelihood', [], 'NextSensed', t.NextSensed);
            obj.Inner.consume(part);
        end

        function n = update(obj, boundary)
            n = obj.Inner.update(boundary);
        end

        function s = exportState(obj)
            s = obj.Inner.exportState();
        end

        function importState(obj, s)
            obj.Inner.importState(s);
        end
    end
end
