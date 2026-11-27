classdef FactoredPolicy < OceanAgents.Core.IPolicy
    % FACTOREDPOLICY  An action made of parts, each decided by its own policy (PG1 contract): e.g. "put" by a biased
    % coin and "guess" by a learned policy. Each part's policy sees the same observation and the part's own labels
    % 1..m (with the values that appear in a permitted joint action); the joint action is their combination. Parts
    % are decided in factor order from the same stream, so a run is reproducible. The permitted joint actions
    % must be a grid (every combination of permitted parts); any other mask -> OceanAgents:Policies:NonGridMask.
    %
    %   F = OceanAgents.Core.ActionFactors(["put" "guess"], {["even" "odd"], ["even" "odd"]});
    %   coin = OceanAgents.Policies.TablePolicy(OceanAgents.Artifacts.PolicyArtifact('Kind', "ProbabilityTable", ...
    %              'Representation', "ColumnIndex", 'Table', repmat([0.9 0.1], 3, 1), 'ActionLabels', 1:2, 'StateStart', 0));
    %   pol = OceanAgents.Policies.FactoredPolicy(F, {coin, OceanAgents.Policies.RandomLegalPolicy()});

    properties (SetAccess = private)
        Factors
        Parts cell = {}
    end

    methods
        function obj = FactoredPolicy(factors, parts)
            arguments
                factors (1,1) OceanAgents.Core.ActionFactors
                parts (1,:) cell
            end
            if numel(parts) ~= numel(factors.Names) || ~all(cellfun(@(p) isa(p, 'OceanAgents.Core.IPolicy'), parts))
                error('OceanAgents:Policies:Incompatible', 'FactoredPolicy: one IPolicy per factor.');
            end
            obj.Factors = factors;
            obj.Parts = parts;
        end

        function d = describe(obj)
            d = struct('Name', "FactoredPolicy", 'Version', "1", 'NumActions', obj.Factors.numJoint(), ...
                'ProvidesLikelihood', false, 'Factors', obj.Factors.Names, ...
                'Parts', {cellfun(@(p) p.describe(), obj.Parts, 'UniformOutput', false)});
        end

        function dec = act(obj, view, stream)
            F = obj.Factors;
            if numel(view.ActionLabels) ~= F.numJoint()
                error('OceanAgents:Policies:Incompatible', 'The view has %d actions; the factors make %d.', ...
                    numel(view.ActionLabels), F.numJoint());
            end
            obj.requireGrid_(view.ActionMask);
            subs = zeros(1, numel(F.Names));
            for f = 1:numel(F.Names)
                partView = obj.partView_(view, F.Names(f));
                d = obj.Parts{f}.act(partView, stream);
                subs(f) = d.Action;
            end
            k = F.joint(subs);
            if ~view.ActionMask(k)
                error('OceanAgents:Policies:IllegalTableAction', ...
                    'The parts chose a combination (%s) the Laws do not permit.', mat2str(subs));
            end
            dec = OceanAgents.Core.Decision('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Action', view.ActionLabels(k));
        end

        function p = probabilities(obj, view)
            % PROBABILITIES  Product of the parts' probabilities over the joint actions (parts are independent).
            F = obj.Factors;
            obj.requireGrid_(view.ActionMask);
            P = cell(1, numel(F.Names));
            for f = 1:numel(F.Names), P{f} = obj.Parts{f}.probabilities(obj.partView_(view, F.Names(f))); end
            p = zeros(1, F.numJoint());
            for k = 1:F.numJoint()
                s = F.parts(k);
                q = 1;
                for f = 1:numel(F.Names), q = q * P{f}(s(f)); end
                p(k) = q;
            end
        end

        function beginEpisode(obj, view)
            for f = 1:numel(obj.Parts), obj.Parts{f}.beginEpisode(obj.partView_(view, obj.Factors.Names(f))); end
        end

        function endEpisode(obj)
            for f = 1:numel(obj.Parts), obj.Parts{f}.endEpisode(); end
        end

        function s = exportState(obj)
            s = struct('Parts', {cellfun(@(p) p.exportState(), obj.Parts, 'UniformOutput', false)});
        end

        function importState(obj, s)
            for f = 1:numel(obj.Parts), obj.Parts{f}.importState(s.Parts{f}); end
        end
    end

    methods (Access = private)
        function requireGrid_(obj, mask)
            % REQUIREGRID_  Parts are decided independently, so the permitted joint actions must be every combination
            % of the permitted parts (a grid). A mask that forbids some combinations of permitted parts cannot be met
            % by independent parts without changing their distributions; it is refused rather than resampled.
            F = obj.Factors;
            grid = true(1, F.numJoint());
            for f = 1:numel(F.Names)
                pm = F.partMask(mask, F.Names(f));
                for k = 1:F.numJoint()
                    s = F.parts(k);
                    grid(k) = grid(k) && pm(s(f));
                end
            end
            if ~isequal(grid, logical(reshape(mask, 1, [])))
                error('OceanAgents:Policies:NonGridMask', ['FactoredPolicy needs the permitted actions to be every ' ...
                    'combination of the permitted parts; this mask forbids some combinations (%s).'], ...
                    mat2str(find(grid & ~mask)));
            end
        end
        function v = partView_(obj, view, name)
            m = obj.Factors.partMask(logical(view.ActionMask), name);
            v = OceanAgents.Core.ActorView('AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, ...
                'FrameID', view.FrameID, 'Tick', view.Tick, 'Phase', view.Phase, 'Observation', view.Observation, ...
                'ActionLabels', 1:numel(m), 'ActionMask', m);
        end
    end
end
