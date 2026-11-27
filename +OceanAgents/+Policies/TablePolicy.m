classdef TablePolicy < OceanAgents.Core.IPolicy
    % TABLEPOLICY  Acts from a fixed PolicyArtifact of Kind "ActionTable" (deterministic) or "ProbabilityTable"
    % (stochastic). The artifact's entries are converted to columns ONCE, at construction, through its ActionCodec and
    % declared Representation, so a label is never mistaken for a column (IG02 / RI04).
    %
    %   a   = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
    %             'Table', pi, 'ActionLabels', [0 1 2]);
    %   pol = OceanAgents.Policies.TablePolicy(a);
    %   d   = pol.act(view, stream)
    %
    % Legality is taken from the VIEW's ActionMask at decision time. If the table's action is illegal there (or a
    % probability row puts mass on an illegal action), the policy raises OceanAgents:Policies:IllegalTableAction:
    % a stale or mismatched table is surfaced as a fault (RI02), never quietly renormalised or replaced.
    % Randomness: an ActionTable draws nothing; a ProbabilityTable draws exactly one rand(stream) per act.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 6.4.

    properties (SetAccess = immutable)
        Artifact                % OceanAgents.Artifacts.PolicyArtifact (value copy)
    end

    properties (SetAccess = immutable, GetAccess = private)
        Codec_                  % ActionCodec of the artifact columns
        States_                 % DiscreteStateCodec of the artifact rows
        Columns_ double = []    % ActionTable: S x 1 columns;  ProbabilityTable: S x A probabilities
    end

    methods
        function obj = TablePolicy(artifact)
            arguments
                artifact (1,1) OceanAgents.Artifacts.PolicyArtifact
            end
            if ~ismember(artifact.Kind, ["ActionTable", "ProbabilityTable"])
                error('OceanAgents:Policies:Incompatible', 'TablePolicy needs an ActionTable or ProbabilityTable, not a %s.', ...
                    artifact.Kind);
            end
            obj.Artifact = artifact;
            obj.Codec_ = artifact.actionCodec();
            obj.States_ = artifact.stateCodec();
            if artifact.Kind == "ActionTable"
                obj.Columns_ = artifact.columnTable();      % validates entries (Artifacts:Incompatible)
            else
                artifact.validateEntries();
                obj.Columns_ = artifact.Table;
            end
        end

        function d = describe(obj)
            d = struct('Name', "TablePolicy", 'Version', obj.Artifact.Version, ...
                'NumActions', numel(obj.Artifact.ActionLabels), 'ProvidesLikelihood', true, ...
                'ActionLabels', obj.Artifact.ActionLabels, 'Kind', obj.Artifact.Kind);
        end

        function checkCompatible(obj, obsSpace, actSpace, role)
            % CHECKCOMPATIBLE  The artifact must fit the role and spaces (PolicyArtifact.validateAgainst); a mismatch
            % is reported as OceanAgents:Policies:Incompatible with the artifact error attached as its cause.
            try
                obj.Artifact.validateAgainst(obsSpace, actSpace, string(role));
            catch cause
                err = MException('OceanAgents:Policies:Incompatible', 'TablePolicy cannot act for role "%s": %s', ...
                    string(role), cause.message);
                throw(err.addCause(cause));
            end
        end

        function p = probabilities(obj, view)
            % PROBABILITIES  The table's distribution over the VIEW's columns. Consumes no randomness.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
            end
            p = obj.viewProbabilities_(view);
        end

        function dec = act(obj, view, stream)
            % ACT  ActionTable: the table's action (no draw). ProbabilityTable: one rand(stream), inverse CDF.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
                stream (1,1) RandStream
            end
            p = obj.viewProbabilities_(view);
            if obj.Artifact.Kind == "ActionTable"
                k = find(p == 1, 1);
            else
                k = sampleInverseCdf_(p, rand(stream));
            end
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', view.ActionLabels(k), 'Probabilities', p, ...
                'Likelihood', p(k), 'Provenance', struct('Policy', "TablePolicy", 'ArtifactVersion', obj.Artifact.Version));
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1: a lookup table has no formula (IsSymbolic = false).
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('%s over %d states x %d actions, representation %s, role %s', ...
                obj.Artifact.Kind, obj.Artifact.NumStates, numel(obj.Artifact.ActionLabels), ...
                obj.Artifact.Representation, obj.Artifact.Role), ...
                'legality from the view; an illegal table action is a fault, never renormalised'};
            meta.IsSymbolic = false;
            meta.Reference = "tabular policy lookup (PolicyArtifact)";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3: ActorView in, Decision out.
            spec = struct();
            spec.ComponentName = "TablePolicy";
            spec.Inputs = struct('Name', "view", 'Domain', "Agent-ActorView", 'Dtype', "OceanAgents.Core.ActorView", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "decision", 'Domain', "Agent-Decision", 'Dtype', "OceanAgents.Core.Decision", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "tablepolicy");
            spec.ComposesFrom = {};
        end
    end

    methods (Access = private)
        function p = viewProbabilities_(obj, view)
            % Row of the observation -> distribution over artifact columns -> the same mass on the view's columns.
            if ~obj.Codec_.sameLabelSet(view.ActionLabels)
                error('OceanAgents:Policies:Incompatible', 'View action labels %s differ from the table''s labels %s.', ...
                    mat2str(view.ActionLabels), mat2str(obj.Codec_.Labels));
            end
            row = obj.States_.encode(view.Observation);
            A = numel(view.ActionLabels);
            if obj.Artifact.Kind == "ActionTable"
                q = zeros(1, A);
                q(obj.Columns_(row)) = 1;                  % artifact column space
            else
                q = obj.Columns_(row, :);
            end
            if ~any(q > 0)
                error('OceanAgents:Policies:IllegalTableAction', 'The table defines no action for observation %s.', ...
                    mat2str(view.Observation));
            end
            % artifact column j <-> view column viewCol(j) (label-based; the orders may differ)
            [~, viewCol] = ismember(obj.Codec_.Labels, view.ActionLabels);
            p = zeros(1, A);
            p(viewCol) = q;
            if any(p(~view.ActionMask) > 0)
                bad = view.ActionLabels(p > 0 & ~view.ActionMask);
                error('OceanAgents:Policies:IllegalTableAction', ...
                    'Table action(s) %s at observation %s are illegal in the current view (legal: %s).', ...
                    mat2str(bad), mat2str(view.Observation), mat2str(view.legalLabels()));
            end
        end
    end
end
