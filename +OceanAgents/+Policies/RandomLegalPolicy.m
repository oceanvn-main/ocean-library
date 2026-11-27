classdef RandomLegalPolicy < OceanAgents.Core.IPolicy
    % RANDOMLEGALPOLICY  Uniform over the actions that are legal in the current view:
    %   pi(a | o) = 1 / |L(o)|  for a in L(o),  0 otherwise,
    % where L(o) is read from the VIEW's ActionMask (never from a stored table). Label-agnostic: it acts for any
    % discrete action set, so it is the natural behaviour policy for off-policy learning and for baselines.
    %
    %   pol = OceanAgents.Policies.RandomLegalPolicy();
    %   d   = pol.act(view, stream)       % one rand(stream) draw, inverse-CDF sampling; Likelihood = 1/|L|
    %   p   = pol.probabilities(view)     % no randomness consumed
    %
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 6.4.

    methods
        function obj = RandomLegalPolicy()
        end

        function d = describe(~)
            d = struct('Name', "RandomLegalPolicy", 'Version', "1", 'NumActions', [], ...
                'ProvidesLikelihood', true, 'ActionLabels', zeros(1, 0));
        end

        function checkCompatible(~, ~, actSpace, ~)
            % CHECKCOMPATIBLE  Any discrete action space is fine.
            if ~isa(actSpace, 'OceanStats.RL.Env.Spaces.Discrete')
                error('OceanAgents:Policies:Incompatible', 'RandomLegalPolicy needs a Discrete action space, got %s.', ...
                    class(actSpace));
            end
        end

        function p = probabilities(~, view)
            % PROBABILITIES  1/|L| on the view's legal columns, 0 elsewhere. Consumes no randomness.
            arguments
                ~
                view (1,1) OceanAgents.Core.ActorView
            end
            legal = view.ActionMask;
            if ~any(legal)
                error('OceanAgents:Core:IllegalAction', 'No legal action in a terminal view (agent %s).', view.AgentID);
            end
            p = zeros(1, numel(legal));
            p(legal) = 1 / nnz(legal);
        end

        function dec = act(obj, view, stream)
            % ACT  Sample a legal action uniformly with exactly one rand(stream) draw.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
                stream (1,1) RandStream
            end
            p = obj.probabilities(view);
            k = sampleInverseCdf_(p, rand(stream));
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', view.ActionLabels(k), 'Probabilities', p, ...
                'Likelihood', p(k), 'Provenance', struct('Policy', "RandomLegalPolicy"));
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1: the uniform-over-legal formula.
            meta = struct();
            meta.SymbolicLatex = "\pi(a\mid o) = 1/\lvert\mathcal{L}(o)\rvert,\ a \in \mathcal{L}(o)";
            meta.Symbols = [ ...
                struct('Symbol', '\pi(a\mid o)', 'Description', 'probability of choosing action a given observation o'), ...
                struct('Symbol', '\mathcal{L}(o)', 'Description', 'set of actions legal in the current view (its ActionMask)')];
            meta.Assumptions = {'legality comes only from the view''s ActionMask', 'at least one legal action'};
            meta.IsSymbolic = true;
            meta.Reference = "uniform random behaviour policy over admissible actions";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3: ActorView in, Decision out.
            spec = policyPipelineSpec_("RandomLegalPolicy", "randomlegal");
        end
    end
end


function spec = policyPipelineSpec_(name, label)
spec = struct();
spec.ComponentName = name;
spec.Inputs = struct('Name', "view", 'Domain', "Agent-ActorView", 'Dtype', "OceanAgents.Core.ActorView", 'Shape', "1 x 1");
spec.Outputs = struct('Name', "decision", 'Domain', "Agent-Decision", 'Dtype', "OceanAgents.Core.Decision", 'Shape', "1 x 1");
spec.Kind = "NONVIRTUAL";
spec.DiagramSymbol = struct('Type', "default", 'Label', label);
spec.ComposesFrom = {};
end
