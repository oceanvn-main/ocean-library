classdef ParityGuessTest < OceanTest.BaseTestCase
    % PARITYGUESSTEST  PG1 (PG1_Parity_Guess_Contract.md): the parity-guess game, factored actions, the Player class
    % with observers, and the purpose of the experiment - a player that learns from the other exploits it.
    %   Theory: P2 wins p2/(1+p2) where p2 = P(P2 guesses right) = 0.1 + 0.8 g, g = P(P2 guesses even);
    %   random guess (g = 1/2) -> 1/3; best response (g = 1) -> 0.474.

    methods (Static)
        function p = coin(pEven)
            p = OceanAgents.Policies.TablePolicy(OceanAgents.Artifacts.PolicyArtifact('Kind', "ProbabilityTable", ...
                'Representation', "ColumnIndex", 'Table', repmat([pEven 1 - pEven], 3, 1), 'ActionLabels', 1:2, ...
                'StateStart', 0));
        end

        function c = perceptions()
            c = {OceanAgents.Minds.Perception.MovesPerception(string(1:4)), OceanAgents.Minds.Perception.DigitPerception()};
        end

        function [p1, p2, env, fp] = players(variant)
            T = @OceanTest.OceanAgents.Integration.ParityGuessTest.coin;
            P = @OceanTest.OceanAgents.Integration.ParityGuessTest.perceptions;
            env = OceanAgents.Env.Reference.ParityGuessEnv('Roster', ["p1" "p2"]);
            F = env.Factors;
            fp = [];
            m1 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, {T(0.9), ...
                OceanAgents.Policies.RandomLegalPolicy()}), 1:4, 'Perceptions', P());
            switch variant
                case "random"
                    m2 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, {T(0.7), ...
                        OceanAgents.Policies.RandomLegalPolicy()}), 1:4, 'Perceptions', P());
                case "q"
                    codec = OceanAgents.Artifacts.DiscreteStateCodec(3, 'Start', 0);
                    q = OceanStats.RL.TD.Values.TabularQ(true(3, 2), 0);
                    m2 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, {T(0.7), ...
                        OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:2, 'Epsilon', 0.05, 'StateCodec', codec)}), 1:4, ...
                        'Perceptions', P());
                    m2.addMemory(OceanAgents.Memory.TableMemory("values", q, ["even" "odd"]));
                    m2.addLearner(OceanAgents.Learners.FactorLearner(OceanAgents.Learners.OneStepQLearner(q, 1:2, ...
                        'Alpha', 0.1, 'Gamma', 1, 'StateCodec', codec), F, "guess"), "q");
                case "fp"
                    fp = OceanStats.GameTheory.Learning.FictitiousPlay(eye(2), 1);
                    m2 = OceanAgents.Minds.PolicySignalMind(OceanAgents.Policies.FactoredPolicy(F, {T(0.7), ...
                        OceanAgents.Policies.FictitiousPlayPolicy(fp)}), 1:4, 'Perceptions', P());
                    m2.addMemory(OceanAgents.Memory.BeliefMemory("belief", fp));
                    m2.addLearner(OceanAgents.Learners.FictitiousPlayLearner(fp), "fp");
            end
            p1 = OceanAgents.Player("p1", m1);
            p2 = OceanAgents.Player("p2", m2);
        end

        function [rec, trace, env] = play(variant, games)
            [p1, p2, env] = OceanTest.OceanAgents.Integration.ParityGuessTest.players(variant);
            trace = OceanAgents.Observe.PlayerTrace();
            p2.attach(trace);
            rec = OceanAgents.Runtime.JointRunner(env, {p1, p2}, 'Seed', 5).run(games);
        end

        function g = guessedEven(trace, env, gameRange)
            % fraction of P2's TURNS (not games) guessing even, over the turns of the games in gameRange. Per turn,
            % as the theory: a per-game average is biased, because a right guess ends a game and "even" is right
            % more often, so even-guesses are over-represented among the last turns of games.
            parts = arrayfun(@(a) env.Factors.parts(a), trace.Actions, 'UniformOutput', false);
            guess = cellfun(@(s) s(2), parts);
            inRange = ismember(trace.Episodes, gameRange);
            g = mean(guess(inRange) == 1);
        end
    end

    methods (Test)
        function theGameFollowsItsRules(tc)
            env = OceanAgents.Env.Reference.ParityGuessEnv('Roster', ["a" "b"], 'MaxTurns', 3);
            F = env.Factors;
            D = @(who, f, put, guess) OceanAgents.Core.Decision('AgentID', who, 'EpisodeID', f.EpisodeID, ...
                'FrameID', f.FrameID, 'Action', F.joint([put guess]));
            B = @(f, d) OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, 'Phase', f.Phase, ...
                'Decisions', d);
            tc.verifyEqual([F.joint([1 1]) F.joint([2 1]) F.joint([1 2]) F.joint([2 2])], 1:4);
            f = env.reset(0);
            tc.verifyEqual(f.Requested, [true true], 'simultaneous');
            tc.verifyEqual([f.Views{1}.Observation f.Views{2}.Observation], [0 0], 'nothing revealed yet');
            f = env.step(B(f, [D("a", f, 1, 1), D("b", f, 2, 2)]));               % a guesses even (b put odd): wrong
            tc.verifyFalse(f.EpisodeEnded);                                       % b guesses odd (a put even): wrong
            tc.verifyEqual([f.Views{1}.Observation f.Views{2}.Observation], [2 1], 'each sees the other''s last put');
            f = env.step(B(f, [D("a", f, 1, 2), D("b", f, 1, 1)]));   % a guesses odd (b put even): wrong; b guesses even
            tc.verifyEqual(f.Rewards, [-1 1]);                        % (a put even): right -> b wins
            tc.verifyEqual(f.EndReason, "win:b");
            f = env.reset(0);
            f = env.step(B(f, [D("a", f, 1, 1), D("b", f, 1, 1)]));               % both right
            tc.verifyEqual([f.Rewards, f.EpisodeEnded], [0 0 1]);
            tc.verifyEqual(f.EndReason, "draw:both-right");
            f = env.reset(0);
            for t = 1:3, f = env.step(B(f, [D("a", f, 1, 2), D("b", f, 1, 2)])); end
            tc.verifyEqual(f.EndReason, "draw:turns");
            tc.verifyTrue(all(f.Terminated) && ~any(f.Truncated), 'the turn limit is a rule of the game');
        end

        function factoredActionsAndTheFactorLearner(tc)
            F = OceanAgents.Core.ActionFactors(["put" "guess"], {["even" "odd"], ["even" "odd"]});
            tc.verifyEqual(F.parts(3), [1 2]);
            tc.verifyEqual(F.partMask(logical([0 0 1 1]), "guess"), [false true]);
            tc.verifyEqual(F.partMask(logical([0 0 1 1]), "put"), [true true]);
            pol = OceanAgents.Policies.FactoredPolicy(F, {OceanTest.OceanAgents.Integration.ParityGuessTest.coin(1), ...
                OceanTest.OceanAgents.Integration.ParityGuessTest.coin(0)});    % put always even, guess always odd
            v = OceanAgents.Core.ActorView('AgentID', "x", 'EpisodeID', 1, 'FrameID', 1, 'Observation', 0, ...
                'ActionLabels', 1:4, 'ActionMask', true(1, 4));
            d = pol.act(v, RandStream('mrg32k3a', 'Seed', 1));
            tc.verifyEqual(d.Action, F.joint([1 2]));
            tc.verifyEqual(pol.probabilities(v), [0 0 1 0], 'AbsTol', 1e-15);
            q = OceanStats.RL.TD.Values.TabularQ(true(3, 2), 0);
            codec = OceanAgents.Artifacts.DiscreteStateCodec(3, 'Start', 0);
            L = OceanAgents.Learners.FactorLearner(OceanAgents.Learners.OneStepQLearner(q, 1:2, 'Alpha', 1, 'Gamma', 1, ...
                'StateCodec', codec), F, "guess");
            t = OceanAgents.Core.Transition('AgentID', "x", 'EpisodeID', 1, 'Observation', 1, 'Action', 4, ...
                'ActionLabels', 1:4, 'Reward', 1, 'NextObservation', 2, 'NextActionLabels', 1:4, 'NextMask', false(1, 4), ...
                'Terminated', true, 'Discount', 1);
            L.consume(t);  L.update("step");
            tc.verifyEqual(q.Q(2, :), [0 1], 'joint action 4 = (put odd, guess odd): only the guess part learned');
            tc.verifyTrue(L.Q == q);
        end

        function aPlayerCanBeObservedWithoutChangingTheRun(tc)
            [rec1, trace] = tc.play("fp", 60);
            [p1, p2, env] = tc.players("fp");
            rec2 = OceanAgents.Runtime.JointRunner(env, {p1, p2}, 'Seed', 5).run(60);   % no observer
            tc.verifyEqual(rec1.EpisodeReturns, rec2.EpisodeReturns, 'an observer never changes the run');
            tc.verifyEqual(numel(trace.Actions), sum(rec1.EpisodeLengths));
            tc.verifyEqual(sum(trace.Rewards), sum(rec1.EpisodeReturns(:, 2)));
            [q1, q2, env3] = tc.players("random");
            bad = OceanTest.OceanAgents.Fixtures.Signals.FailingObserver();
            q2.attach(bad);
            rec3 = OceanAgents.Runtime.JointRunner(env3, {q1, q2}, 'Seed', 5).run(5);
            tc.verifyEqual(rec3.Status, "Completed", 'a failing observer cannot stop the run');
            tc.verifyEqual(q2.numObservers(), 0);
            tc.verifySubstring(char(bad.Failure), 'display broke');
            q2.attach(trace);  q2.attach(trace);
            tc.verifyEqual(q2.numObservers(), 1, 'an observer is attached once');
            tc.verifyError(@() q2.attach(OceanTest.OceanAgents.Fixtures.Signals.SpyObserver(q1)), ...
                'OceanAgents:Player:ObserverHoldsAgent', 'an observer holding the other player is refused');
            tc.verifyError(@() q2.attach(OceanTest.OceanAgents.Fixtures.Signals.SpyObserver(q1.Mind)), ...
                'OceanAgents:Player:ObserverHoldsAgent');
            tc.verifyEqual(q2.numObservers(), 1);
        end

        function theBeliefIsTheSmoothedCountOfTheOthersPuts(tc)
            % agent level: after the run, P2's belief = (puts even + 1) / (puts + 2) over every put P1 revealed
            [p1, p2, env, fp] = tc.players("fp");
            seen1 = OceanAgents.Observe.PlayerTrace();
            p1.attach(seen1);
            OceanAgents.Runtime.JointRunner(env, {p1, p2}, 'Seed', 5).run(40);
            parts = arrayfun(@(a) env.Factors.parts(a), seen1.Actions, 'UniformOutput', false);
            put = cellfun(@(s) s(1), parts);
            b = fp.beliefs();
            tc.verifyEqual(b{1}(1), (nnz(put == 1) + 1) / (numel(put) + 2), 'AbsTol', 1e-12);
        end

        function edgeCasesAreRefusedOrSkipped(tc)
            % a mask that is not a grid of part masks is refused (independent parts cannot meet it)
            F = OceanAgents.Core.ActionFactors(["put" "guess"], {["even" "odd"], ["even" "odd"]});
            pol = OceanAgents.Policies.FactoredPolicy(F, {OceanAgents.Policies.RandomLegalPolicy(), ...
                OceanAgents.Policies.RandomLegalPolicy()});
            v = OceanAgents.Core.ActorView('AgentID', "x", 'EpisodeID', 1, 'FrameID', 1, 'Observation', 0, ...
                'ActionLabels', 1:4, 'ActionMask', logical([1 1 1 0]));
            tc.verifyError(@() pol.act(v, RandStream('mrg32k3a', 'Seed', 1)), 'OceanAgents:Policies:NonGridMask');
            tc.verifyError(@() pol.probabilities(v), 'OceanAgents:Policies:NonGridMask');
            % a step whose next view was not sensed is not counted again by the fictitious-play learner
            fp = OceanStats.GameTheory.Learning.FictitiousPlay(eye(2), 1);
            L = OceanAgents.Learners.FictitiousPlayLearner(fp);
            T = @(sensed) OceanAgents.Core.Transition('AgentID', "x", 'EpisodeID', 1, 'Observation', 1, 'Action', 1, ...
                'ActionLabels', 1:4, 'Reward', 0, 'NextObservation', 1, 'NextActionLabels', 1:4, 'NextMask', true(1, 4), ...
                'Discount', 1, 'NextSensed', sensed);
            L.consume(T(true));  L.consume(T(false));  L.update("step");
            tc.verifyEqual(L.NumUpdates, 1);
            % a belief memory binds through a FactorLearner wrapper too; a wrapper of another learner is refused
            F = OceanAgents.Core.ActionFactors(["put" "guess"], {["even" "odd"], ["even" "odd"]});
            mem = OceanAgents.Memory.BeliefMemory("belief", fp);
            tc.verifyTrue(mem.isUpdatedBy(OceanAgents.Learners.FactorLearner(L, F, "guess")));
            q = OceanStats.RL.TD.Values.TabularQ(true(3, 2), 0);
            tc.verifyFalse(mem.isUpdatedBy(OceanAgents.Learners.FactorLearner(OceanAgents.Learners.OneStepQLearner( ...
                q, 1:2, 'Alpha', 0.1, 'Gamma', 1, 'StateCodec', OceanAgents.Artifacts.DiscreteStateCodec(3, 'Start', 0)), ...
                F, "guess")));
            % a saved belief must be a valid record
            s = fp.Belief.exportState();
            s.Counts{1} = [-1 2];
            tc.verifyError(@() fp.Belief.importState(s), 'OceanStats:GameTheory:BadBelief');
            s = fp.Belief.exportState();
            s.Observations = s.Observations + 1;
            tc.verifyError(@() fp.Belief.importState(s), 'OceanStats:GameTheory:BadBelief');
            fp.Belief.importState(fp.Belief.exportState());   % a valid round trip is accepted
        end

        function aPlayerThatLearnsFromTheOtherExploitsIt(tc)
            % the purpose of PG1: best response to the opponent's mixed strategy
            games = 800;  late = 301:games;
            result = struct();
            for variant = ["random" "q" "fp"]
                [rec, trace, env] = tc.play(variant, games);
                tc.verifyEqual(rec.Status, "Completed");
                R = rec.EpisodeReturns(:, 2);
                ge = tc.guessedEven(trace, env, late);
                early = tc.guessedEven(trace, env, 1:20);
                p2 = 0.1 + 0.8 * ge;
                expect = p2 / (1 + p2);
                win = mean(R(late) == 1);
                ci = 4 * sqrt(expect * (1 - expect) / numel(late));          % 4 standard errors
                tc.verifyEqual(win, expect, 'AbsTol', ci, sprintf('%s: win rate vs theory', variant));
                result.(char(variant)) = struct('Win', win, 'GuessEven', ge, 'Early', early);
                tc.log(matlab.unittest.Verbosity.Terse, sprintf(['%s: P2 wins %.3f (theory %.3f), guesses even %.3f ' ...
                    '(first 20 games %.3f)'], variant, win, expect, ge, early));
            end
            tc.verifyEqual(result.random.GuessEven, 0.5, 'AbsTol', 0.1);
            tc.verifyEqual(result.random.Win, 1/3, 'AbsTol', 0.06);
            tc.verifyGreaterThan(result.q.GuessEven, 0.9, 'Q-learning learned to guess even');
            tc.verifyGreaterThan(result.fp.GuessEven, 0.97, 'fictitious play best-responds');
            tc.verifyGreaterThan(result.q.Win, result.random.Win + 0.08, 'learning from the other pays');
            tc.verifyGreaterThan(result.fp.Win, result.random.Win + 0.08);
            tc.verifyGreaterThanOrEqual(result.fp.Early, result.q.Early, 'fictitious play learns at least as fast');
        end
    end
end
