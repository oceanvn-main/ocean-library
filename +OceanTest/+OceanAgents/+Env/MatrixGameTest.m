classdef MatrixGameTest < OceanTest.BaseTestCase
    % MATRIXGAMETEST  OceanAgents.Env.Reference.RepeatedMatrixGameEnv: simultaneous requests, observations from completed
    % rounds only, payoffs from game.payoff, horizon termination, batch validated before mutation, and the matching-
    % pennies sampling check of A0 section 9.4 (C20). Expected values: A0_Contract_Freeze.md sections 5.2 and 9.4.

    methods (Static, Access = private)
        function g = pennies()
            % Matching pennies (A0 9.4): PayoffA = [1 -1; -1 1], PayoffB = -PayoffA.
            A = [1 -1; -1 1];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', -A);
        end

        function env = make(varargin)
            env = OceanAgents.Env.Reference.RepeatedMatrixGameEnv( ...
                OceanTest.OceanAgents.Env.MatrixGameTest.pennies(), 'Roster', ["p1" "p2"], varargin{:});
        end

        function b = batch(frame, actions)
            % One Decision per requested slot, in roster order, with the given labels.
            ds = OceanAgents.Core.Decision.empty(1, 0);
            k = 0;
            for slot = find(frame.Requested)
                k = k + 1;
                ds(k) = OceanAgents.Core.Decision('AgentID', frame.Roster(slot), 'EpisodeID', frame.EpisodeID, ...
                    'FrameID', frame.FrameID, 'Action', actions(k));
            end
            b = OceanAgents.Core.ActionBatch('EpisodeID', frame.EpisodeID, 'FrameID', frame.FrameID, 'Decisions', ds);
        end
    end

    methods (Test)
        function resetFrameRequestsEveryLiveSlot(tc)
            env = tc.make('Rounds', 3);
            f = env.reset(0);
            tc.verifyEqual(f.Roster, ["p1" "p2"]);
            tc.verifyEqual(f.Live, [true true]);
            tc.verifyEqual(f.Requested, [true true]);
            tc.verifyEqual(f.RewardValid, [false false]);      % no transition yet
            tc.verifyFalse(f.EpisodeEnded);
            for k = 1:2
                tc.verifyEqual(f.Views{k}.ActionLabels, [1 2]);
                tc.verifyEqual(f.Views{k}.ActionMask, [true true]);
                tc.verifyEqual(f.Views{k}.Observation, [0 0]);  % zeros at round 0
            end
        end

        function purePayoffsExact(tc)
            % A0 9.4: (1,1) -> [1 -1]; (1,2) -> [-1 1]; and by the matrices (2,1) -> [-1 1]; (2,2) -> [1 -1].
            profiles = [1 1; 1 2; 2 1; 2 2];
            expected = [1 -1; -1 1; -1 1; 1 -1];
            for r = 1:4
                env = tc.make('Rounds', 1);
                f = env.reset(r);
                f = env.step(tc.batch(f, profiles(r, :)));
                tc.verifyEqual(f.Rewards, expected(r, :), mat2str(profiles(r, :)));
                tc.verifyEqual(f.RewardValid, [true true]);
            end
        end

        function observationsUseCompletedRoundsOnly(tc)
            env = tc.make('Rounds', 5, 'Observe', "own-payoff");
            f = env.reset(0);
            f = env.step(tc.batch(f, [1 2]));                  % u = [-1 1]
            tc.verifyEqual(f.Views{1}.Observation, [1 -1]);
            tc.verifyEqual(f.Views{2}.Observation, [2 1]);
            % the views of the current frame are fixed before any decision is made: they cannot depend on the batch
            envA = tc.make('Rounds', 5, 'Observe', "public-profile");
            envB = tc.make('Rounds', 5, 'Observe', "public-profile");
            fa = envA.reset(0);  fb = envB.reset(0);
            fa = envA.step(tc.batch(fa, [2 1]));
            fb = envB.step(tc.batch(fb, [2 1]));
            tc.verifyEqual(fa.Views{1}.Observation, [2 1 -1]); % [lastProfile  own payoff]
            tc.verifyEqual(fa.Views{2}.Observation, [2 1 1]);
            tc.verifyEqual(fa.Views, fb.Views);
            fa2 = envA.step(tc.batch(fa, [1 1]));              % different pending actions in the two worlds ...
            fb2 = envB.step(tc.batch(fb, [2 2]));
            tc.verifyEqual(fa.Views{1}.Observation, fb.Views{1}.Observation);   % ... were invisible beforehand
            tc.verifyEqual(fa2.Views{1}.Observation, [1 1 1]);
            tc.verifyEqual(fb2.Views{1}.Observation, [2 2 1]);
        end

        function horizonTerminates(tc)
            T = 3;
            env = tc.make('Rounds', T);
            f = env.reset(0);
            for t = 1:T
                prev = f.FrameID;
                f = env.step(tc.batch(f, [1 1]));
                tc.verifyEqual(f.Tick, t);
            end
            tc.verifyTrue(f.EpisodeEnded);
            tc.verifyEqual(f.EndReason, "horizon");
            tc.verifyEqual(f.Terminated, [true true]);
            tc.verifyEqual(f.Truncated, [false false]);
            tc.verifyEqual(f.Live, [false false]);
            tc.verifyEqual(f.Requested, [false false]);
            for k = 1:2
                fbk = f.feedbackFor(k, prev);
                tc.verifyTrue(fbk.Terminated);
                tc.verifyEqual(fbk.Reason, "horizon");
                tc.verifyEqual(fbk.NextView.ActionMask, [false false]);
                tc.verifyEqual(fbk.NextView.Observation, [1 (3 - 2 * k)]);   % [own action, own payoff]
            end
            tc.verifyError(@() env.step(tc.batch(f, [])), 'OceanAgents:Env:NotReady');
        end

        function batchValidatedBeforeMutation(tc)
            env = tc.make('Rounds', 4);
            f = env.reset(0);
            frame0 = env.FrameID;
            tc.verifyError(@() env.step(tc.batch(f, [3 1])), 'OceanAgents:Core:IllegalAction');
            d1 = OceanAgents.Core.Decision('AgentID', "p1", 'EpisodeID', 1, 'FrameID', f.FrameID, 'Action', 1);
            d2 = OceanAgents.Core.Decision('AgentID', "p2", 'EpisodeID', 1, 'FrameID', f.FrameID, 'Action', 1);
            dx = OceanAgents.Core.Decision('AgentID', "p3", 'EpisodeID', 1, 'FrameID', f.FrameID, 'Action', 1);
            mk = @(ds, fid) OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', fid, 'Decisions', ds);
            tc.verifyError(@() env.step(mk(d1, f.FrameID)), 'OceanAgents:Core:MissingDecision');
            tc.verifyError(@() env.step(mk([d1 d1 d2], f.FrameID)), 'OceanAgents:Core:DuplicateDecision');
            tc.verifyError(@() env.step(mk([d1 d2 dx], f.FrameID)), 'OceanAgents:Core:UnexpectedDecision');
            tc.verifyError(@() env.step(mk([d1 d2], f.FrameID + 1)), 'OceanAgents:Core:StaleFrame');
            tc.verifyError(@() env.step(42), 'OceanAgents:Core:InvalidRecord');
            tc.verifyEqual(env.Round, 0);                        % nothing moved
            tc.verifyEqual(env.FrameID, frame0);
            f = env.step(mk([d2 d1], f.FrameID));                % any order of decisions is fine
            tc.verifyEqual(env.Round, 1);
            tc.verifyEqual(f.Rewards, [1 -1]);
        end

        function matchingPenniesSampling(tc)
            % A0 9.4: p1 plays label 1 w.p. 0.7 (own RandStream), p2 always 1. E[u1] = 0.4, Var = 0.84,
            % T = 10000 -> band 4*sqrt(0.84/T) = 0.0367. u2 = -u1 in every round.
            T = 10000;
            p = 0.7;
            env = tc.make('Rounds', T);
            pol1 = OceanTest.OceanAgents.Fixtures.Env.FixedProbabilityPolicy([p 1 - p]);
            pol2 = OceanTest.OceanAgents.Fixtures.Env.FixedProbabilityPolicy([1 0]);
            s1 = RandStream('mrg32k3a', 'Seed', 2026);
            s2 = RandStream('mrg32k3a', 'Seed', 7);
            u = zeros(T, 2);
            a1 = zeros(T, 1);
            f = env.reset(0);
            for t = 1:T
                dA = pol1.act(f.Views{1}, s1);
                dB = pol2.act(f.Views{2}, s2);
                a1(t) = dA.Action;
                f = env.step(OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, ...
                    'Decisions', [dA dB]));
                u(t, :) = f.Rewards;
            end
            tc.verifyTrue(f.EpisodeEnded);
            tc.verifyEqual(u(:, 2), -u(:, 1));                   % zero-sum, exactly
            mu = p * 1 + (1 - p) * (-1);
            band = 4 * sqrt((1 - mu^2) / T);
            tc.verifyEqual(band, 0.0367, 'AbsTol', 5e-5);
            tc.verifyLessThanOrEqual(abs(mean(u(:, 1)) - mu), band);
            % independent replay: the same stream reproduces every choice, and the payoff is the matrix entry
            replay = RandStream('mrg32k3a', 'Seed', 2026);
            draws = rand(replay, T, 1);
            tc.verifyEqual(a1, 1 + (draws > p));
            A = [1 -1; -1 1];
            tc.verifyEqual(u(:, 1), A(sub2ind([2 2], a1, ones(T, 1))));
        end

        function idsMonotonicAndLifecycle(tc)
            env = tc.make('Rounds', 2);
            tc.verifyError(@() env.step(42), 'OceanAgents:Env:NotReady');   % before reset
            ids = [];
            for e = 1:3
                f = env.reset(e);
                tc.verifyEqual(f.EpisodeID, e);
                tc.verifyEqual(f.ResearchInfo.Seed, e);
                ids(end + 1) = f.FrameID; %#ok<AGROW>
                while ~f.EpisodeEnded
                    f = env.step(tc.batch(f, [1 2]));
                    ids(end + 1) = f.FrameID; %#ok<AGROW>
                end
            end
            tc.verifyEqual(ids, 1:9);                            % 3 frames per episode, never reset
            env.close();
            env.close();                                          % idempotent
            tc.verifyError(@() env.reset(0), 'OceanAgents:Env:NotReady');
        end

        function describeAndPillars(tc)
            env = tc.make('Rounds', 10, 'Observe', "public-profile");
            spec = env.describe();
            tc.verifyEqual(spec.Timing, "simultaneous");
            tc.verifyEqual(spec.Roster, ["p1" "p2"]);
            tc.verifyEqual(spec.Horizon, 10);
            tc.verifyEmpty(spec.Capabilities.list());
            tc.verifyEqual(spec.ActionSpaces{2}.N, 2);
            tc.verifyTrue(spec.ObservationSpaces{1}.contains([2 1 -1]));
            tc.verifyEqual(env.EpisodeID, 0);                    % describe does not start an episode
            meta = env.getMetadata();
            native = OceanStats.RL.Env.Classic.RandomWalkEnv();
            tc.verifyEqual(sort(fieldnames(meta)), sort(fieldnames(native.getMetadata())));
            tc.verifyFalse(meta.IsSymbolic);
            ps = env.getPipelineSpec();
            tc.verifyEqual(sort(fieldnames(ps)), sort(fieldnames(native.getPipelineSpec())));
            tc.verifyEqual(ps.Outputs.Domain, "Agent-DecisionFrame");
            tc.verifyTrue(isa(env, 'OceanAgents.Core.IJointEnvironment'));
        end

        function constructorChecks(tc)
            tc.verifyError(@() OceanAgents.Env.Reference.RepeatedMatrixGameEnv(tc.pennies(), ...
                'Roster', ["a" "b" "c"]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Env.Reference.RepeatedMatrixGameEnv(tc.pennies(), ...
                'Roster', ["a" "a"]), 'OceanAgents:Core:InvalidRecord');
            tc.verifyError(@() OceanAgents.Env.Reference.RepeatedMatrixGameEnv(42), 'OceanAgents:Env:UnsupportedSpace');
            env = OceanAgents.Env.Reference.RepeatedMatrixGameEnv(tc.pennies());
            tc.verifyEqual(env.Roster, ["p1" "p2"]);             % default roster
        end
    end
end
