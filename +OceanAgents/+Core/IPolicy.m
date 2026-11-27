classdef (Abstract) IPolicy < handle
    % IPOLICY  Chooses one primitive action from an ActorView. A policy never holds an environment, a runner, a host
    % or another agent's component; its randomness comes only from the stream its host passes in.
    %
    %   d   = policy.act(view, stream)     one Decision (external label + optional probabilities/likelihood)
    %   p   = policy.probabilities(view)   1 x A over view columns, zero off the legal set; consumes NO randomness
    %
    % observe() may update episodic memory only; learned parameters are changed by an ILearner, never while acting.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 3.

    methods (Abstract)
        d = describe(obj)                 % struct: Name, Version, NumActions, ProvidesLikelihood
        dec = act(obj, view, stream)
        p = probabilities(obj, view)
    end

    methods
        function checkCompatible(obj, obsSpace, actSpace, role) %#ok<INUSD>
            % CHECKCOMPATIBLE  Throw OceanAgents:Policies:Incompatible if this policy cannot act for the role/spaces.
            % Default: the action count must match when the policy declares one.
            d = obj.describe();
            if isfield(d, 'NumActions') && ~isempty(d.NumActions) && isprop(actSpace, 'N') && d.NumActions ~= actSpace.N
                error('OceanAgents:Policies:Incompatible', '%s acts over %d actions; the role has %d.', ...
                    d.Name, d.NumActions, actSpace.N);
            end
        end
        function beginEpisode(~, ~), end
        function observe(~, ~), end
        function endEpisode(~), end
        function s = exportState(~), s = struct(); end
        function importState(~, ~), end
    end
end
