classdef PolicyTest < OceanTest.BaseTestCase
    % POLICYTEST  RandomLegalPolicy, TablePolicy, EpsilonGreedyQPolicy (A0 sections 6.4, 9.3).
    % Gates: IG03 (sampled frequencies match the reported probabilities), RI02 (legality from the view, not a table).

    properties (Constant)
        Seed = 20260928
    end

    methods (Static, Access = private)
        function v = view(obs, labels, mask, frameID)
            if nargin < 4, frameID = 1; end
            v = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', frameID, ...
                'Observation', obs, 'ActionLabels', labels, 'ActionMask', mask);
        end
        function q = qWithRow(S, A, row, values)
            % Fresh TabularQ with one row set exactly (0 + dT is exact).
            q = OceanStats.RL.TD.Values.TabularQ(true(S, A));
            dT = zeros(S, A);
            dT(row, :) = values;
            q.applyIncrement(dT);
        end
        function rs = stream(seed)
            rs = RandStream('mrg32k3a', 'Seed', seed);
        end
        function pol = cliffPolicy(q, epsilon)
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', epsilon, ...
                'StateCodec', OceanAgents.Artifacts.DiscreteStateCodec(48));
        end
    end

    methods (Test)
        % ---------------- EpsilonGreedyQPolicy: A0 9.3 ----------------
        function epsGreedyHandCaseAllLegal(tc)
            % Q(37,:) = [-0.5 -50 0 0], eps = 0.1: G = {3,4}, p = [0.025 0.025 0.475 0.475].
            q = tc.qWithRow(48, 4, 37, [-0.5 -50 0 0]);
            pol = tc.cliffPolicy(q, 0.1);
            p = pol.probabilities(tc.view(37, 1:4, true(1, 4)));
            tc.verifyEqual(p, [0.025 0.025 0.475 0.475], 'AbsTol', 1e-15);
            tc.verifyEqual(sum(p), 1, 'AbsTol', 1e-15);
        end

        function epsGreedyHandCaseMasked(tc)
            % mask [T T T F]: L = {1,2,3}, G = {3}; p = [0.1/3 0.1/3 0.1/3+0.9 0]. Column 4 ties for the max in Q
            % but is illegal in the VIEW, so it gets nothing (RI02: the view's mask, not the table's).
            q = tc.qWithRow(48, 4, 37, [-0.5 -50 0 0]);
            pol = tc.cliffPolicy(q, 0.1);
            p = pol.probabilities(tc.view(37, 1:4, [true true true false]));
            tc.verifyEqual(p, [0.1/3 0.1/3 0.1/3 + 0.9 0], 'AbsTol', 1e-15);
            tc.verifyEqual(p(4), 0);
        end

        function epsGreedyUsesViewMaskNotTableMask(tc)
            % The TABLE says action 4 is invalid at row 1 (Q stays 0 there), but the VIEW says it is legal: the view
            % wins. Q(1,:) = [-1 -1 -1 0] after the increment, so action 4 is the unique greedy action.
            q = OceanStats.RL.TD.Values.TabularQ([true(1, 3) false; true(1, 4)]);
            q.applyIncrement([-1 -1 -1 0; 0 0 0 0]);
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.2);
            p = pol.probabilities(tc.view(1, 1:4, true(1, 4)));
            tc.verifyEqual(p, [0.05 0.05 0.05 0.85], 'AbsTol', 1e-15);
        end

        function epsGreedyTiesSplitEvenly(tc)
            q = tc.qWithRow(2, 4, 1, [5 5 5 1]);
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.4);
            p = pol.probabilities(tc.view(1, 1:4, true(1, 4)));
            tc.verifyEqual(p, [0.1 + 0.6/3, 0.1 + 0.6/3, 0.1 + 0.6/3, 0.1], 'AbsTol', 1e-15);
            pol.setEpsilon(0);
            tc.verifyEqual(pol.probabilities(tc.view(1, 1:4, true(1, 4))), [1 1 1 0] / 3, 'AbsTol', 1e-15);
            pol.setEpsilon(1);
            tc.verifyEqual(pol.probabilities(tc.view(1, 1:4, [true false true true])), [1 0 1 1] / 3, 'AbsTol', 1e-15);
        end

        function epsGreedyLabelsAreNotColumns(tc)
            % Q columns hold labels [3 2 1 0] (reverse order); the view lists labels [0 1 2 3]. Q(1,:) = [9 0 0 0]
            % means label 3 is greedy, which is the view's LAST column.
            q = tc.qWithRow(1, 4, 1, [9 0 0 0]);
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, [3 2 1 0], 'Epsilon', 0);
            v = tc.view(1, [0 1 2 3], true(1, 4));
            tc.verifyEqual(pol.probabilities(v), [0 0 0 1]);
            d = pol.act(v, tc.stream(1));
            tc.verifyEqual(d.Action, 3);
            d.validateAgainst(v);
        end

        function epsGreedyOneDrawPerActAndNoneForProbabilities(tc)
            q = tc.qWithRow(48, 4, 37, [-0.5 -50 0 0]);
            pol = tc.cliffPolicy(q, 0.1);
            v = tc.view(37, 1:4, true(1, 4));
            globalBefore = rng;
            rsA = tc.stream(tc.Seed);
            rsB = tc.stream(tc.Seed);
            % probabilities: no draw at all
            pol.probabilities(v);
            tc.verifyEqual(rsA.State, rsB.State);
            % act: exactly one rand(stream), and the action is the inverse-CDF image of that draw
            for k = 1:25
                d = pol.act(v, rsA);
                u = rand(rsB);
                tc.verifyEqual(rsA.State, rsB.State, 'act must consume exactly one rand(stream)');
                expected = find(u < cumsum(d.Probabilities), 1);
                tc.verifyEqual(d.Action, v.ActionLabels(expected));
            end
            tc.verifyEqual(rng, globalBefore, 'the global stream must never be touched');
        end

        function epsGreedyDecisionIsConsistent(tc)
            q = tc.qWithRow(48, 4, 37, [-0.5 -50 0 0]);
            pol = tc.cliffPolicy(q, 0.1);
            rs = tc.stream(tc.Seed);
            v = tc.view(37, 1:4, [true true true false], 7);
            for k = 1:50
                d = pol.act(v, rs);
                d.validateAgainst(v);                                   % legal, no mass off L, Likelihood matches
                tc.verifyEqual(d.Likelihood, d.Probabilities(v.columnOf(d.Action)));
                tc.verifyEqual(d.Probabilities, pol.probabilities(v));
                tc.verifyEqual([d.EpisodeID d.FrameID], [1 7]);
                tc.verifyNotEqual(d.Action, 4);
            end
            tc.verifyEqual(d.PolicyVersion, sum(q.Counts(:)));
        end

        function epsGreedyEmpiricalFrequenciesMatch(tc)
            % IG03. n = 40000 seeded draws. Each action count is Binomial(n, p_k); the check allows 4.5 standard
            % deviations sqrt(p_k (1 - p_k) / n) per action: per-action two-sided false-alarm 6.8e-6, so < 3e-5 for
            % the 4 actions together (Bonferroni). The seed makes the run deterministic; the band says why it
            % is a fair test rather than a tuned one. Also checks every Decision's Likelihood.
            n = 40000;
            q = tc.qWithRow(48, 4, 37, [-0.5 -50 0 0]);
            pol = tc.cliffPolicy(q, 0.1);
            cases = {true(1, 4), [true true true false]};
            for c = 1:numel(cases)
                v = tc.view(37, 1:4, cases{c});
                p = pol.probabilities(v);
                rs = tc.stream(tc.Seed + c);
                counts = zeros(1, 4);
                for k = 1:n
                    d = pol.act(v, rs);
                    col = v.columnOf(d.Action);
                    counts(col) = counts(col) + 1;
                end
                freq = counts / n;
                band = 4.5 * sqrt(p .* (1 - p) / n);
                tc.verifyLessThanOrEqual(abs(freq - p), band, sprintf('mask case %d', c));
                tc.verifyEqual(counts(p == 0), zeros(1, nnz(p == 0)), 'zero-probability actions are never drawn');
            end
        end

        function epsGreedyErrors(tc)
            q = tc.qWithRow(48, 4, 37, [-0.5 -50 0 0]);
            pol = tc.cliffPolicy(q, 0.1);
            tc.verifyError(@() pol.probabilities(tc.view(49, 1:4, true(1, 4))), 'OceanAgents:Artifacts:UnknownState');
            tc.verifyError(@() pol.probabilities(tc.view(0, 1:4, true(1, 4))), 'OceanAgents:Artifacts:UnknownState');
            tc.verifyError(@() pol.act(tc.view(37, 1:4, false(1, 4)), tc.stream(1)), 'OceanAgents:Core:IllegalAction');
            tc.verifyError(@() pol.probabilities(tc.view(37, 0:3, true(1, 4))), 'OceanAgents:Policies:Incompatible');
            tc.verifyError(@() OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:3), 'OceanAgents:Policies:Incompatible');
            tc.verifyError(@() OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 1.5), ...
                'MATLAB:validators:mustBeBetweenScalarBounds');
            obsSp = OceanStats.RL.Env.Spaces.Discrete(48);
            pol.checkCompatible(obsSp, OceanStats.RL.Env.Spaces.Discrete(4), "default");
            tc.verifyError(@() pol.checkCompatible(obsSp, OceanStats.RL.Env.Spaces.Discrete(4, 'Start', 0), "default"), ...
                'OceanAgents:Policies:Incompatible');
            tc.verifyError(@() pol.checkCompatible(OceanStats.RL.Env.Spaces.Discrete(47), ...
                OceanStats.RL.Env.Spaces.Discrete(4), "default"), 'OceanAgents:Policies:Incompatible');
        end

        function epsGreedySharesItsTableAndExportsOnlyEpsilon(tc)
            q = tc.qWithRow(2, 2, 1, [0 1]);
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:2, 'Epsilon', 0);
            v = tc.view(1, 1:2, true(1, 2));
            tc.verifyEqual(pol.probabilities(v), [0 1]);
            q.applyIncrement([2 0; 0 0]);                 % the learner's write is visible to the policy at once
            tc.verifyEqual(pol.probabilities(v), [1 0]);
            tc.verifySameHandle(pol.Q, q);
            s = pol.exportState();
            tc.verifyEqual(s.Epsilon, 0);
            other = OceanAgents.Policies.EpsilonGreedyQPolicy(OceanStats.RL.TD.Values.TabularQ(true(2, 2)), 1:2, 'Epsilon', 0.3);
            other.importState(s);
            tc.verifyEqual(other.Epsilon, 0);
        end

        % ---------------- RandomLegalPolicy ----------------
        function randomLegalUniformOverLegal(tc)
            pol = OceanAgents.Policies.RandomLegalPolicy();
            v = tc.view([0 0], [0 1 2], [true false true]);
            p = pol.probabilities(v);
            tc.verifyEqual(p, [0.5 0 0.5]);
            rsA = tc.stream(3);  rsB = tc.stream(3);
            n = 20000;  counts = zeros(1, 3);
            actions = zeros(1, n);  expected = zeros(1, n);  lik = zeros(1, n);
            for k = 1:n
                d = pol.act(v, rsA);
                expected(k) = v.ActionLabels(find(rand(rsB) < cumsum(p), 1));
                d.validateAgainst(v);
                actions(k) = d.Action;  lik(k) = d.Likelihood;
                counts(v.columnOf(d.Action)) = counts(v.columnOf(d.Action)) + 1;
            end
            tc.verifyEqual(rsA.State, rsB.State, 'n acts consumed exactly n rand(stream) draws');
            tc.verifyEqual(actions, expected, 'inverse-CDF image of each draw');
            tc.verifyEqual(lik, 0.5 * ones(1, n));
            % 4.5-sigma binomial band per action (see epsGreedyEmpiricalFrequenciesMatch)
            tc.verifyLessThanOrEqual(abs(counts / n - p), 4.5 * sqrt(p .* (1 - p) / n));
            tc.verifyEqual(counts(2), 0);
            tc.verifyError(@() pol.act(tc.view(1, 1:2, false(1, 2)), rsA), 'OceanAgents:Core:IllegalAction');
        end

        % ---------------- TablePolicy ----------------
        function tablePolicyActionTableIllegalIsAFault(tc)
            % RI02: the stored action is illegal in this view -> fault, not a silent substitute.
            a = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
                'Table', [2; 0], 'ActionLabels', [0 1 2]);
            pol = OceanAgents.Policies.TablePolicy(a);
            rs = tc.stream(1);  ref = tc.stream(1);
            v = tc.view(1, [0 1 2], true(1, 3));
            d = pol.act(v, rs);
            tc.verifyEqual(d.Action, 2);
            tc.verifyEqual(d.Probabilities, [0 0 1]);
            tc.verifyEqual(d.Likelihood, 1);
            tc.verifyEqual(rs.State, ref.State, 'a deterministic table draws nothing');
            tc.verifyError(@() pol.act(tc.view(1, [0 1 2], [true true false]), rs), 'OceanAgents:Policies:IllegalTableAction');
            tc.verifyError(@() pol.probabilities(tc.view(1, [0 1 2], [true true false])), ...
                'OceanAgents:Policies:IllegalTableAction');
            tc.verifyError(@() pol.act(tc.view(3, [0 1 2], true(1, 3)), rs), 'OceanAgents:Artifacts:UnknownState');
        end

        function tablePolicyProbabilityTableNoRenormalising(tc)
            a = OceanAgents.Artifacts.PolicyArtifact('Kind', "ProbabilityTable", 'Representation', "ColumnIndex", ...
                'Table', [0.2 0.3 0.5; 0 0 0], 'ActionLabels', [0 1 2]);
            pol = OceanAgents.Policies.TablePolicy(a);
            v = tc.view(1, [0 1 2], true(1, 3));
            tc.verifyEqual(pol.probabilities(v), [0.2 0.3 0.5]);
            % mass on an illegal column: fault, never renormalised to [0.4 0.6 0]-style rows
            tc.verifyError(@() pol.act(tc.view(1, [0 1 2], [true false true]), tc.stream(1)), ...
                'OceanAgents:Policies:IllegalTableAction');
            % an all-zero row defines no action
            tc.verifyError(@() pol.act(tc.view(2, [0 1 2], true(1, 3)), tc.stream(1)), ...
                'OceanAgents:Policies:IllegalTableAction');
            % one draw per act, inverse CDF, consistent decision; the view's column order may differ from the table's
            vPerm = tc.view(1, [2 0 1], true(1, 3));
            tc.verifyEqual(pol.probabilities(vPerm), [0.5 0.2 0.3]);
            rsA = tc.stream(9);  rsB = tc.stream(9);
            for k = 1:30
                d = pol.act(vPerm, rsA);
                u = rand(rsB);
                tc.verifyEqual(rsA.State, rsB.State);
                tc.verifyEqual(d.Action, vPerm.ActionLabels(find(u < cumsum([0.5 0.2 0.3]), 1)));
                d.validateAgainst(vPerm);
            end
        end

        function tablePolicyCompatibility(tc)
            a = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
                'Table', [2; 0], 'ActionLabels', [0 1 2], 'Role', "row");
            pol = OceanAgents.Policies.TablePolicy(a);
            obsSp = OceanStats.RL.Env.Spaces.Discrete(2);
            pol.checkCompatible(obsSp, OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0), "row");
            tc.verifyError(@() pol.checkCompatible(obsSp, OceanStats.RL.Env.Spaces.Discrete(3), "row"), ...
                'OceanAgents:Policies:Incompatible');
            tc.verifyError(@() pol.checkCompatible(obsSp, OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0), "column"), ...
                'OceanAgents:Policies:Incompatible');
            tc.verifyError(@() pol.probabilities(tc.view(1, [1 2 3], true(1, 3))), 'OceanAgents:Policies:Incompatible');
            q = OceanAgents.Artifacts.PolicyArtifact('Kind', "QTable", 'Representation', "ColumnIndex", ...
                'Table', zeros(2, 3), 'ActionLabels', [0 1 2]);
            tc.verifyError(@() OceanAgents.Policies.TablePolicy(q), 'OceanAgents:Policies:Incompatible');
        end

        % ---------------- pillars ----------------
        function pillars(tc)
            q = tc.qWithRow(48, 4, 37, [-0.5 -50 0 0]);
            eg = tc.cliffPolicy(q, 0.1);
            a = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
                'Table', 1, 'ActionLabels', [0 1]);
            pols = {OceanAgents.Policies.RandomLegalPolicy(), OceanAgents.Policies.TablePolicy(a), eg};
            symbolic = [true false true];
            for k = 1:numel(pols)
                m = pols{k}.getMetadata();
                tc.verifyEqual(m.IsSymbolic, symbolic(k));
                tc.verifyTrue(all(isfield(m, {'SymbolicLatex', 'Symbols', 'Assumptions', 'Reference'})));
                s = pols{k}.getPipelineSpec();
                tc.verifyEqual(s.Inputs(1).Domain, "Agent-ActorView");
                tc.verifyEqual(s.Outputs(1).Domain, "Agent-Decision");
                d = pols{k}.describe();
                tc.verifyTrue(all(isfield(d, {'Name', 'Version', 'NumActions', 'ProvidesLikelihood'})));
            end
            m = OceanAgents.Policies.EpsilonGreedyQPolicy.getMetadata();      % static, from the sidecar
            tc.verifyEqual(m.SymbolicLatex, ...
                "\pi(a\mid s) = \varepsilon/\lvert\mathcal{L}\rvert + (1-\varepsilon)\,\mathbb{1}[a\in G]/\lvert G\rvert");
            tc.verifyEqual(string(m.id), "EpsilonGreedyQPolicy");
            tc.verifyTrue(all(isfield(m, {'display_name', 'textbook', 'pedagogy', 'equations', 'parameters', ...
                'complexity', 'strengths', 'weaknesses', 'when_to_use', 'references'})));
            tc.verifyTrue(all(isfield(m.Symbols, {'Symbol', 'Description'})));
            tc.verifyEqual(OceanAgents.Policies.RandomLegalPolicy().getMetadata().SymbolicLatex, ...
                "\pi(a\mid o) = 1/\lvert\mathcal{L}(o)\rvert,\ a \in \mathcal{L}(o)");
        end
    end
end
