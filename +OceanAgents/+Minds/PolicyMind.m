classdef PolicyMind < OceanAgents.Core.IDecisionProcess
    % POLICYMIND  The simplest mind: a reactive agent whose every decision is its policy's action for the current
    % view. It forwards each IDecisionProcess call to one IPolicy and adds nothing: no beliefs, no plans, no model of
    % other agents. Richer minds (belief-based, deliberative) replace this class without touching the environment.
    %
    %   pol  = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.1);
    %   mind = OceanAgents.Minds.PolicyMind(pol);
    %   d    = mind.decide(view, stream)       % == pol.act(view, stream)
    %   mind.describe().Profile                % "reactive-policy"
    %
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 6.6.

    properties (SetAccess = immutable)
        Policy      % OceanAgents.Core.IPolicy (the one component this mind owns)
    end

    methods
        function obj = PolicyMind(policy)
            arguments
                policy (1,1) OceanAgents.Core.IPolicy
            end
            obj.Policy = policy;
        end

        function d = describe(obj)
            d = struct('Name', "PolicyMind", 'Profile', "reactive-policy", ...
                'Components', struct('Policy', obj.Policy.describe()));
        end

        function checkCompatible(obj, obsSpace, actSpace, role)
            obj.Policy.checkCompatible(obsSpace, actSpace, role);
        end

        function beginEpisode(obj, view)
            obj.Policy.beginEpisode(view);
        end

        function dec = decide(obj, view, stream)
            dec = obj.Policy.act(view, stream);
        end

        function observe(obj, feedback)
            obj.Policy.observe(feedback);
        end

        function endEpisode(obj)
            obj.Policy.endEpisode();
        end

        function s = exportState(obj)
            s = struct('Name', "PolicyMind", 'Policy', obj.Policy.exportState());
        end

        function importState(obj, s)
            arguments
                obj
                s (1,1) struct
            end
            if ~isfield(s, 'Name') || s.Name ~= "PolicyMind" || ~isfield(s, 'Policy')
                error('OceanAgents:Policies:Incompatible', 'State is not a PolicyMind state.');
            end
            obj.Policy.importState(s.Policy);
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1: a container, no formula of its own (IsSymbolic = false).
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('reactive: every decision is %s.act(view)', class(obj.Policy))};
            meta.IsSymbolic = false;
            meta.Reference = "runtime/container component (A0 section 8)";
        end

        function spec = getPipelineSpec(obj)
            % GETPIPELINESPEC  Pillar 3: ActorView and Feedback in, Decision out; composed from its policy.
            spec = struct();
            spec.ComponentName = "PolicyMind";
            spec.Inputs = [ ...
                struct('Name', "view", 'Domain', "Agent-ActorView", 'Dtype', "OceanAgents.Core.ActorView", 'Shape', "1 x 1"), ...
                struct('Name', "feedback", 'Domain', "Agent-Feedback", 'Dtype', "OceanAgents.Core.Feedback", 'Shape', "1 x 1")];
            spec.Outputs = struct('Name', "decision", 'Domain', "Agent-Decision", 'Dtype', "OceanAgents.Core.Decision", 'Shape', "1 x 1");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "policymind");
            spec.ComposesFrom = {string(class(obj.Policy))};
        end
    end
end
