classdef FictitiousPlayPolicy < OceanAgents.Core.IPolicy
    % FICTITIOUSPLAYPOLICY  Acts with fictitious play's response to the current beliefs (best response, or the smooth
    % logit one), restricted to the moves the Laws permit (PG1 contract). It reads the same FictitiousPlay object the
    % FictitiousPlayLearner updates (learner -> memory -> policy, design D27).
    %
    %   pol = OceanAgents.Policies.FictitiousPlayPolicy(fp);

    properties (SetAccess = private)
        FP
    end

    methods
        function obj = FictitiousPlayPolicy(fp)
            arguments
                fp (1,1) OceanStats.GameTheory.Learning.FictitiousPlay
            end
            obj.FP = fp;
        end

        function d = describe(obj)
            d = struct('Name', "FictitiousPlayPolicy", 'Version', "1", 'NumActions', numel(obj.FP.expectedPayoffs()), ...
                'ProvidesLikelihood', true, 'Response', obj.FP.Response);
        end

        function p = probabilities(obj, view)
            sigma = obj.FP.responseProbabilities();
            if numel(sigma) ~= numel(view.ActionLabels)
                error('OceanAgents:Policies:Incompatible', 'The view has %d actions; fictitious play knows %d.', ...
                    numel(view.ActionLabels), numel(sigma));
            end
            legal = logical(view.ActionMask);
            if ~any(legal), error('OceanAgents:Core:IllegalAction', 'No legal action in this view.'); end
            p = sigma .* legal;
            if sum(p) == 0
                U = obj.FP.expectedPayoffs();                % every best response is forbidden: best permitted one
                U(~legal) = -Inf;
                p = double(U == max(U));
            end
            p = p / sum(p);
        end

        function dec = act(obj, view, stream)
            p = obj.probabilities(view);
            c = cumsum(p);
            k = find(rand(stream) * c(end) <= c, 1);
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', view.ActionLabels(k), 'Probabilities', p, 'Likelihood', p(k));
        end
    end
end
