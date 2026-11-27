classdef BeliefMemory < OceanAgents.Memory.IMemory
    % BELIEFMEMORY  What a player believes about the others: a fictitious-play learner from the Game Theory package
    % (OceanStats.GameTheory.Learning.FictitiousPlay) with its empirical beliefs (PG1 contract). Written only by its
    % bound learner (FictitiousPlayLearner, through its own update); read by the policy and by queries:
    %
    %   struct('Op', "belief")     -> Beliefs (one 1 x m_j row per opponent), Observations
    %   struct('Op', "payoffs")    -> Expected (1 x m_i expected payoffs of each own strategy), Best (index)

    properties (SetAccess = private)
        FP                                  % OceanStats.GameTheory.Learning.FictitiousPlay (handle)
    end

    methods
        function obj = BeliefMemory(id, fp, options)
            arguments
                id (1,1) string
                fp (1,1) OceanStats.GameTheory.Learning.FictitiousPlay
                options.BodySignature = []
            end
            obj@OceanAgents.Memory.IMemory(id, "belief", 'BodySignature', options.BodySignature);
            obj.FP = fp;
        end

        function tf = isUpdatedBy(obj, learner)
            tf = false;
            if ~isprop(learner, 'FP'), return; end
            fp = learner.FP;                                  % [] when a wrapper's inner learner has none
            tf = isscalar(fp) && isa(fp, 'OceanStats.GameTheory.Learning.FictitiousPlay') && fp == obj.FP;
        end

        function bindLearner(obj, learner, learnerID)
            if ~obj.isUpdatedBy(learner)
                error('OceanAgents:Memory:NotWriter', 'BeliefMemory "%s": the learner does not update this belief.', obj.ID);
            end
            obj.bindWriter(learnerID);
        end
    end

    methods (Access = protected)
        function payload = answer(obj, q)
            if ~isfield(q, 'Op'), obj.badQuery_('a query needs Op.'); end
            switch string(q.Op)
                case "belief"
                    payload = struct('Beliefs', {obj.FP.beliefs()}, 'Observations', obj.FP.Belief.Observations);
                case "payoffs"
                    U = obj.FP.expectedPayoffs();
                    [~, best] = max(U);
                    payload = struct('Expected', U, 'Best', best);
                otherwise
                    obj.badQuery_('unknown Op "%s".', string(q.Op));
            end
        end

        function store(obj, ~)
            obj.badQuery_('the belief is written by its bound learner''s own update, not by write signals.');
        end
    end
end
