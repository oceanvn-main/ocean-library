classdef CodecAndArtifactTest < OceanTest.BaseTestCase
    % CODECANDARTIFACTTEST  ActionCodec, DiscreteStateCodec and PolicyArtifact (A0 sections 6.1-6.3, 9.6).
    % Gates: IG02 / RI04 (label vs column), IG12 / C15 (artifact compatibility), AP06 (no silent state fallback).

    methods (Static, Access = private)
        function a = actionTable(entry, representation, varargin)
            % One-state ActionTable over environment labels [0 1 2] (A0 section 9.6).
            a = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', representation, ...
                'Table', entry, 'ActionLabels', [0 1 2], 'NumStates', 1, varargin{:});
        end
        function v = view(labels, mask, obs)
            v = OceanAgents.Core.ActorView('AgentID', "a", 'EpisodeID', 1, 'FrameID', 1, ...
                'Observation', obs, 'ActionLabels', labels, 'ActionMask', mask);
        end
    end

    methods (Test)
        % ---------------- ActionCodec ----------------
        function actionCodecMapsLabelsAndColumns(tc)
            c = OceanAgents.Artifacts.ActionCodec([0 1 2]);
            tc.verifyEqual(c.column(2), 3);
            tc.verifyEqual(c.column(0), 1);
            tc.verifyEqual(c.column([2 0]), [3 1]);
            tc.verifyEqual(c.labelOf(3), 2);
            tc.verifyEqual(c.labelOf([1 2]), [0 1]);
            tc.verifyEqual(c.numActions(), 3);
            tc.verifyTrue(c.matches([0 1 2]));
            tc.verifyFalse(c.matches([2 1 0]));          % same set, different column order
            tc.verifyTrue(c.sameLabelSet([2 1 0]));
            tc.verifyFalse(c.matches([1 2 3]));
        end

        function actionCodecUnknownLabel(tc)
            c = OceanAgents.Artifacts.ActionCodec([0 1 2]);
            tc.verifyError(@() c.column(3), 'OceanAgents:Artifacts:UnknownLabel');
            tc.verifyError(@() c.column(0.5), 'OceanAgents:Artifacts:UnknownLabel');
            tc.verifyError(@() c.labelOf(0), 'OceanAgents:Artifacts:UnknownLabel');
            tc.verifyError(@() c.labelOf(4), 'OceanAgents:Artifacts:UnknownLabel');
        end

        function actionCodecRejectsBadLabels(tc)
            tc.verifyError(@() OceanAgents.Artifacts.ActionCodec([1 1 2]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Artifacts.ActionCodec([]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Artifacts.ActionCodec([1 NaN]), 'OceanAgents:Core:NonFinite');
        end

        function actionCodecFromSpace(tc)
            c = OceanAgents.Artifacts.ActionCodec.fromSpace(OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0));
            tc.verifyEqual(c.Labels, [0 1 2]);
            c = OceanAgents.Artifacts.ActionCodec.fromSpace(OceanStats.RL.Env.Spaces.Discrete(4));
            tc.verifyEqual(c.Labels, 1:4);
        end

        % ---------------- DiscreteStateCodec ----------------
        function stateCodecEncodes(tc)
            c = OceanAgents.Artifacts.DiscreteStateCodec(5, 'Start', 0);
            tc.verifyEqual(c.encode(0), 1);
            tc.verifyEqual(c.encode(4), 5);
            tc.verifyEqual(c.decode(5), 4);
            c1 = OceanAgents.Artifacts.DiscreteStateCodec(48);
            tc.verifyEqual(c1.encode(37), 37);
            tc.verifyTrue(c1.matchesSpace(OceanStats.RL.Env.Spaces.Discrete(48)));
            tc.verifyFalse(c1.matchesSpace(OceanStats.RL.Env.Spaces.Discrete(48, 'Start', 0)));
        end

        function stateCodecNeverFallsBackToRowOne(tc)
            % AP06: an unknown observation is an error, never row 1.
            c = OceanAgents.Artifacts.DiscreteStateCodec(5, 'Start', 0);
            bad = {-1, 5, 2.5, NaN, Inf, [1 2], "1", true};
            for k = 1:numel(bad)
                tc.verifyError(@() c.encode(bad{k}), 'OceanAgents:Artifacts:UnknownState', ...
                    sprintf('case %d should be UnknownState', k));
            end
        end

        % ---------------- PolicyArtifact: 9.6 label vs column ----------------
        function labelVersusColumnHandCase(tc)
            % A0 9.6: entry 2 with "Label" -> label 2 (column 3); with "ColumnIndex" -> column 2 = label 1.
            v = tc.view([0 1 2], true(1, 3), 1);
            rs = RandStream('mrg32k3a', 'Seed', 1);
            byLabel = OceanAgents.Policies.TablePolicy(tc.actionTable(2, "Label"));
            tc.verifyEqual(byLabel.act(v, rs).Action, 2);
            tc.verifyEqual(tc.actionTable(2, "Label").columnTable(), 3);
            byColumn = OceanAgents.Policies.TablePolicy(tc.actionTable(2, "ColumnIndex"));
            tc.verifyEqual(byColumn.act(v, rs).Action, 1);
            tc.verifyEqual(tc.actionTable(2, "ColumnIndex").columnTable(), 2);
        end

        function columnZeroIsIncompatible(tc)
            % A0 9.6: entry 0 with "ColumnIndex" -> Artifacts:Incompatible at validateAgainst.
            a = tc.actionTable(0, "ColumnIndex");     % constructs (entries are checked later)
            obsSp = OceanStats.RL.Env.Spaces.Discrete(1);
            actSp = OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0);
            tc.verifyError(@() a.validateAgainst(obsSp, actSp, "default"), 'OceanAgents:Artifacts:Incompatible');
            tc.verifyError(@() OceanAgents.Policies.TablePolicy(a), 'OceanAgents:Artifacts:Incompatible');
            % the same entry is a perfectly good LABEL
            tc.actionTable(0, "Label").validateAgainst(obsSp, actSp, "default");
        end

        function representationIsRequired(tc)
            tc.verifyError(@() OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Table', 1, ...
                'ActionLabels', [0 1 2]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Artifacts.PolicyArtifact('Kind', "QTable", 'Representation', "Label", ...
                'Table', zeros(2, 3), 'ActionLabels', [0 1 2]), 'OceanAgents:Core:InvalidRecord');
        end

        function artifactShapeChecks(tc)
            mk = @(kind, T, varargin) OceanAgents.Artifacts.PolicyArtifact('Kind', kind, ...
                'Representation', "ColumnIndex", 'Table', T, 'ActionLabels', [1 2], varargin{:});
            tc.verifyError(@() mk("ProbabilityTable", ones(3, 1)), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() mk("ActionTable", ones(3, 2)), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() mk("ActionTable", ones(3, 1), 'NumStates', 4), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() mk("QTable", [0 NaN; 0 0]), 'OceanAgents:Core:NonFinite');
            a = mk("QTable", zeros(4, 2));
            tc.verifyEqual(a.NumStates, 4);
            tc.verifyEqual(a.StateStart, 1);
            tc.verifyEqual(a.Role, "default");
        end

        % ---------------- PolicyArtifact.validateAgainst (C15, IG12) ----------------
        function validateAgainstMismatches(tc)
            T = [1; 2; 1; 2];
            a = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "ColumnIndex", ...
                'Table', T, 'ActionLabels', [0 1], 'NumStates', 4, 'StateStart', 0, 'Role', "row");
            obsOK = OceanStats.RL.Env.Spaces.Discrete(4, 'Start', 0);
            actOK = OceanStats.RL.Env.Spaces.Discrete(2, 'Start', 0);
            a.validateAgainst(obsOK, actOK, "row");                                    % fits: no error
            id = 'OceanAgents:Artifacts:Incompatible';
            tc.verifyError(@() a.validateAgainst(obsOK, actOK, "column"), id);         % role
            tc.verifyError(@() a.validateAgainst(OceanStats.RL.Env.Spaces.Discrete(5, 'Start', 0), actOK, "row"), id); % N
            tc.verifyError(@() a.validateAgainst(OceanStats.RL.Env.Spaces.Discrete(4), actOK, "row"), id);  % Start
            tc.verifyError(@() a.validateAgainst(obsOK, OceanStats.RL.Env.Spaces.Discrete(2), "row"), id);  % labels {1,2}
            tc.verifyError(@() a.validateAgainst(obsOK, OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0), "row"), id);
            tc.verifyError(@() a.validateAgainst(obsOK, OceanStats.RL.Env.Spaces.Box(0, 1), "row"), id);
        end

        function validateAgainstOutOfRangeEntries(tc)
            obsSp = OceanStats.RL.Env.Spaces.Discrete(2);
            actSp = OceanStats.RL.Env.Spaces.Discrete(2);
            id = 'OceanAgents:Artifacts:Incompatible';
            mk = @(kind, rep, T) OceanAgents.Artifacts.PolicyArtifact('Kind', kind, 'Representation', rep, ...
                'Table', T, 'ActionLabels', [1 2]);
            tc.verifyError(@() mk("ActionTable", "Label", [1; 3]).validateAgainst(obsSp, actSp, "default"), id);
            tc.verifyError(@() mk("ActionTable", "ColumnIndex", [1; 1.5]).validateAgainst(obsSp, actSp, "default"), id);
            tc.verifyError(@() mk("ProbabilityTable", "ColumnIndex", [0.5 0.6; 1 0]).validateAgainst(obsSp, actSp, "default"), id);
            tc.verifyError(@() mk("ProbabilityTable", "ColumnIndex", [1.5 -0.5; 1 0]).validateAgainst(obsSp, actSp, "default"), id);
            % an all-zero row (no policy at e.g. a terminal state) is allowed
            mk("ProbabilityTable", "ColumnIndex", [0.25 0.75; 0 0]).validateAgainst(obsSp, actSp, "default");
            mk("QTable", "ColumnIndex", [-3 7; 0 1e6]).validateAgainst(obsSp, actSp, "default");
        end

        function provenanceRejectsHandlesAnywhere(tc)
            % F3: an artifact is copied into every TablePolicy built from it, so a handle anywhere in Provenance
            % would be shared between agents. Rejected at construction, at any depth through structs and cells.
            mk = @(prov) tc.actionTable(2, "Label", 'Provenance', prov);
            id = 'OceanAgents:Core:InvalidRecord';
            tc.verifyError(@() mk(containers.Map()), id);                                  % top-level handle
            tc.verifyError(@() mk(RandStream('mrg32k3a')), id);
            tc.verifyError(@() mk(struct('env', OceanStats.RL.Env.Classic.CliffWalkingEnv())), id); % handle in a field
            tc.verifyError(@() mk(struct('a', struct('b', {{1, containers.Map()}}))), id);  % nested struct -> cell
            tc.verifyError(@() mk(struct('f', @sin)), id);                                  % function handle
            tc.verifyError(@() mk(struct('s', struct('x', {1, java.lang.Object()}))), id);  % struct array element
            tc.verifyError(@() mk(struct('c', {{struct('q', OceanStats.RL.TD.Values.TabularQ(true(1, 2)))}})), id);
            tc.verifyError(@() mk([struct('a', 1) struct('a', 2)]), id);                   % not scalar
            tc.verifyError(@() mk("text"), id);                                             % not a struct
            % one shared rule with the Core records (OceanAgents.Core.assertValueData, critic round 2): any object other
            % than string/datetime/duration/categorical is refused, including value classes and dictionaries that
            % could wrap a handle
            tc.verifyError(@() mk(struct('space', OceanStats.RL.Env.Spaces.Discrete(3))), id);
            tc.verifyError(@() mk(struct('d', dictionary(1, {containers.Map()}))), id);
            tc.verifyError(@() mk(struct('d', dictionary("k", 1))), id);
            % plain nested value data is fine and is kept as given
            prov = struct('source', "td", 'seed', 7, 'meta', struct('tags', {{"a", [1 2 3]}}), ...
                'when', datetime(2026, 9, 28), 'space', "Discrete(3, start=1)");
            a = mk(prov);
            tc.verifyEqual(a.Provenance, prov);
            pol = OceanAgents.Policies.TablePolicy(a);
            tc.verifyEqual(pol.Artifact.Provenance, prov);
        end

        function artifactIsValueData(tc)
            % Records/artifacts hold value data: a copy cannot be changed through another copy.
            a = tc.actionTable(2, "Label", 'Provenance', struct('source', "td"));
            b = a;
            tc.verifyFalse(isa(a, 'handle'));
            tc.verifyError(@() setTable(b), 'MATLAB:class:SetProhibited');
            tc.verifyEqual(a.Table, 2);
            function setTable(x)
                x.Table = 1;
            end
        end
    end
end
