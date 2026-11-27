classdef PACExplorationTest < OceanTest.BaseTestCase
    % PACEXPLORATIONTEST  R-max (Strehl, Li & Littman 2009, Algorithm 1) and E³ (Kearns & Singh 2002) against
    % hand-derived values and exact value iteration on the true model.

    methods (Test)
        % ---------------- counts model ----------------
        function countsKeepOnlyTheFirstMSamples(tc)
            c = OceanStats.RL.PAC.TransitionCounts(3, 2, 'Cap', 2);
            tc.verifyFalse(c.record(1, 1, 4, 2, false));
            tc.verifyTrue(c.record(1, 1, 0, 3, false));               % reaches m = 2 now
            tc.verifyFalse(c.record(1, 1, 100, 1, false));            % third sample ignored by the model
            [R, P, Pend] = c.estimates();
            tc.verifyEqual([c.N(1,1), c.Seen(1,1)], [2 3]);
            tc.verifyEqual(R(1,1), 2);
            tc.verifyEqual(full(P(1, :)), [0 0.5 0.5]);
            c.record(2, 2, 1, NaN, true);
            [~, P, Pend] = c.estimates();
            tc.verifyEqual(Pend(2, 2), 1); tc.verifyEqual(full(sum(P(sub2ind([3 2], 2, 2), :))), 0);
        end

        % ---------------- known-state MDP (E³ off-line computations) ----------------
        function knownStatePlansHandValues(tc)
            % s1: a1 -> s2 (r 0), a2 -> s1 (r 1);  s2: a1 -> s3 (r 0), a2 -> s2 (r 0);  s3 unknown.
            c = OceanStats.RL.PAC.TransitionCounts(3, 2);
            c.record(1, 1, 0, 2, false); c.record(1, 2, 1, 1, false);
            c.record(2, 1, 0, 3, false); c.record(2, 2, 0, 2, false);
            [R, P, Pend] = c.estimates();
            p = OceanStats.RL.PAC.knownStatePlans(R, P, Pend, [true; true; false], true(3, 2), 0.9, 2);
            % Escape within T = 2: s2 leaves in one step (a1), s1 in two (a1, a1).
            tc.verifyEqual(p.Escape, [1; 1; 0]);
            tc.verifyEqual([p.ExplorePolicy(1, 2), p.ExplorePolicy(2, 1)], [1 1]);
            % Exploitation: staying in s1 pays 1 / (1 - 0.9) = 10; s0 is worth 0 (MinReward 0).
            tc.verifyEqual(p.ExploitQ(1, :), [0 10], 'AbsTol', 1e-8);
            tc.verifyEqual(p.ExploitPolicy(1), 2);
            tc.verifyEqual(p.ExploitPolicy(3), 0);
            q = OceanStats.RL.PAC.knownStatePlans(R, P, Pend, [true; true; false], true(3, 2), 0.9, 1);
            tc.verifyEqual(q.Escape, [0; 1; 0]);                     % s1 cannot leave in one step
            m = OceanStats.RL.PAC.knownStatePlans(R, P, Pend, [true; true; false], true(3, 2), 0.9, 2, 'MinReward', -1);
            tc.verifyEqual(m.ExploitQ(2, 1), 0.9 * (-1 / 0.1), 'AbsTol', 1e-8);   % escape to s0 worth -10
        end

        function knownStatePlansEpisodeEndHasNoContinuation(tc)
            % s1: a1 ends the episode with reward 3; a2 -> s2 (unknown) with reward 0.
            c = OceanStats.RL.PAC.TransitionCounts(2, 2);
            c.record(1, 1, 3, NaN, true); c.record(1, 2, 0, 2, false);
            [R, P, Pend] = c.estimates();
            p = OceanStats.RL.PAC.knownStatePlans(R, P, Pend, [true; false], true(2, 2), 0.9, 4, 'MinReward', 0);
            tc.verifyEqual(p.ExploitQ(1, :), [3 0], 'AbsTol', 1e-12);      % ending is not escaping
            tc.verifyEqual(p.Escape(1), 1);                                % a2 leaves the known set
            tc.verifyEqual(p.ExplorePolicy(1, 4), 2);
        end

        % ---------------- R-max ----------------
        function rmaxLearnsAnEpisodicTaskWithTerminations(tc)
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 4);
            gamma = 0.9; [~, piStar] = tc.qStar_(env, gamma);
            ag = OceanStats.RL.PAC.RMaxAgent(12, 4, 'KnownCount', 1, 'MaxReward', 1, 'DiscountFactor', gamma);
            tr = ag.run(env, 'Steps', 3000, 'Seed', 1);
            start = env.idx(3, 1);
            tc.verifyGreaterThan(tr.Episodes, 10);                    % the goal was reached and the env reset
            pi = ag.greedyPolicy();
            reach = setdiff(unique(tr.States), env.idx(3, 4));
            tc.verifyEqual(pi(reach), piStar(reach));
            tc.verifyEqual(pi(start), 1);                             % up: avoid the cliff
            tc.verifyEqual(tr.Rewards(end - 4:end) > -100, true(1, 5));
        end

        function rmaxStartsOptimisticEverywhere(tc)
            ag = OceanStats.RL.PAC.RMaxAgent(4, 2, 'MaxReward', 2, 'DiscountFactor', 0.9);
            tc.verifyEqual(ag.Q, 20 * ones(4, 2), 'AbsTol', 1e-12);   % U = Vmax = Rmax / (1 - γ)
            tc.verifyEqual(ag.Sweeps, ceil(log(20 / 0.01) / 0.1));   % ln(Vmax/ε1)/(1-γ); Strehl's form when Rmax = 1
            one = OceanStats.RL.PAC.RMaxAgent(4, 2, 'MaxReward', 1, 'DiscountFactor', 0.9);
            tc.verifyEqual(one.Sweeps, ceil(log(1 / (0.01 * 0.1)) / 0.1));   % the paper's count, rewards in [0,1]
            huge = OceanStats.RL.PAC.RMaxAgent(4, 2, 'MaxReward', 1, 'DiscountFactor', 0.95, 'Epsilon1', 100);
            tc.verifyEqual(huge.Sweeps, 1);                           % never fewer than one sweep
        end

        function rmaxRecoversQStarOnDeterministicChain(tc)
            env = OceanStats.RL.Env.Classic.ChainEnv('Slip', 0);
            gamma = 0.9; [Qstar, piStar] = tc.qStar_(env, gamma);
            % A bound strictly above the true maximum reward (12 > 10) keeps every unknown pair strictly
            % optimistic, so R-max tries all of them and its model Q equals q*.
            ag = OceanStats.RL.PAC.RMaxAgent(5, 2, 'KnownCount', 1, 'MaxReward', 12, 'DiscountFactor', gamma, ...
                'Epsilon1', 1e-9);
            ag.run(env, 'Steps', 300, 'Seed', 1);
            tc.verifyTrue(all(ag.isKnown(), 'all'));
            tc.verifyEqual(ag.Q, Qstar, 'AbsTol', 1e-6);
            tc.verifyEqual(ag.greedyPolicy(), piStar);
        end

        function rmaxStopsExploringWhenTheBoundIsAttained(tc)
            % With Rmax = 10 exactly, U = 10 / (1 - γ) = 100 = q*(5, a): once the end of the chain is reached, the
            % known pair (5, a) ties every unknown pair, so R-max stays there -- optimal from that state, and the
            % "b" actions never need to be learned (exploration stops where the optimistic bound is met).
            env = OceanStats.RL.Env.Classic.ChainEnv('Slip', 0);
            ag = OceanStats.RL.PAC.RMaxAgent(5, 2, 'KnownCount', 1, 'MaxReward', 10, 'DiscountFactor', 0.9);
            tr = ag.run(env, 'Steps', 300, 'Seed', 1);
            tc.verifyEqual(tr.States(end - 10:end), 5 * ones(1, 11));
            tc.verifyEqual(tr.Actions(end), 1);
            tc.verifyEqual(nnz(ag.isKnown()), 5);                     % the five "a" pairs only
        end

        function rmaxFindsTheOptimalPolicyOnStochasticChain(tc)
            env = OceanStats.RL.Env.Classic.ChainEnv();
            gamma = 0.95; [~, piStar] = tc.qStar_(env, gamma);
            ag = OceanStats.RL.PAC.RMaxAgent(5, 2, 'KnownCount', 40, 'MaxReward', 10, 'DiscountFactor', gamma);
            tr = ag.run(env, 'Steps', 20000, 'Seed', 2);
            tc.verifyTrue(all(ag.isKnown(), 'all'));
            tc.verifyEqual(ag.greedyPolicy(), piStar);
            tc.verifyEqual(max(ag.Counts.N, [], 'all'), 40);          % only the first m samples are modelled
            tc.verifyEqual(tr.Modes(end), "known");
        end

        % ---------------- E³ ----------------
        function e3BalancedWanderingAndKnownStates(tc)
            ag = OceanStats.RL.PAC.E3Agent(5, 2, 'KnownCount', 2, 'MaxReward', 10);
            seq = zeros(1, 4);
            for k = 1:4
                seq(k) = ag.act(1, [true true]);
                ag.observe(1, seq(k), 2, 1, false);
            end
            tc.verifyEqual(seq, [1 2 1 2]);                           % least-tried action, ties -> lowest
            tc.verifyTrue(ag.Known(1));                               % each action tried m = 2 times
            tc.verifyEqual(ag.LastMode, "wander");
            T = ceil(log(10 / (0.1 * 0.05)) / 0.05);
            tc.verifyEqual(ag.Horizon, T);                            % ε-horizon time, K&S Eq. 9
            tc.verifyEqual(ag.EscapeThreshold, 0.1 / (2 * T * 10), 'AbsTol', 1e-15);
        end

        function e3RevokesKnownStateWhenANewActionAppears(tc)
            ag = OceanStats.RL.PAC.E3Agent(3, 2, 'KnownCount', 1, 'MaxReward', 1);
            ag.act(1, [true false]); ag.observe(1, 1, 0, 2, false);
            tc.verifyTrue(ag.Known(1));                               % only action 1 was admissible
            a = ag.act(1, [true true]);                               % action 2 becomes admissible
            tc.verifyFalse(ag.Known(1));
            tc.verifyEqual([a, ag.LastMode], [2, "wander"]);
        end

        function e3ExploresThenExploitsOptimallyOnStochasticChain(tc)
            env = OceanStats.RL.Env.Classic.ChainEnv();
            gamma = 0.95; [~, piStar] = tc.qStar_(env, gamma);
            ag = OceanStats.RL.PAC.E3Agent(5, 2, 'KnownCount', 40, 'MaxReward', 10, 'DiscountFactor', gamma);
            tr = ag.run(env, 'Steps', 20000, 'Seed', 3);
            tc.verifyTrue(all(ag.Known));
            tc.verifyEqual(ag.exploitationPolicy(), piStar);
            tc.verifyEqual(ag.escapeProbability(), zeros(5, 1));      % nothing left to escape to
            tc.verifyTrue(any(tr.Modes == "explore"));
            last = ag.Decisions(end, :);
            tc.verifyEqual(last(3), 0);                               % final decision: exploit
            tc.verifyEqual(tr.Modes(end), "exploit");
        end

        function e3ScheduleStrategyIsSeededAndLeavesGlobalRngAlone(tc)
            original = rng; tc.addTeardown(@() rng(original));
            rng(31); expected = rand(); rng(31);
            run1 = tc.runE3_("schedule", 7); run2 = tc.runE3_("schedule", 7);
            tc.verifyEqual(run1.Actions, run2.Actions);
            tc.verifyEqual(rand(), expected);
        end

        function rewardOutsideTheDeclaredRangeIsRejected(tc)
            r = OceanStats.RL.PAC.RMaxAgent(2, 1, 'MaxReward', 1);
            tc.verifyError(@() r.observe(1, 1, 2, 2, false), 'OceanStats:RL:PAC:RewardOutOfRange');
            e = OceanStats.RL.PAC.E3Agent(2, 1, 'MaxReward', 1);
            tc.verifyError(@() e.observe(1, 1, -0.5, 2, false), 'OceanStats:RL:PAC:RewardOutOfRange');
        end

        function chainEnvMatchesItsDefinition(tc)
            env = OceanStats.RL.Env.Classic.ChainEnv();
            [n, p, r] = env.get_transitions(3, 1);
            tc.verifyEqual([n p r], [4 0.8 0; 1 0.2 2]);
            [n, p, r] = env.get_transitions(5, 1);
            tc.verifyEqual([n p r], [5 0.8 10; 1 0.2 2]);
            [~, piStar] = tc.qStar_(env, 0.95);
            tc.verifyEqual(piStar, ones(5, 1));                       % walking to the end is optimal
        end
    end

    methods (Static, Access = private)
        function [Q, pi] = qStar_(env, gamma)
            % Exact q* by value iteration with the library's own expected backups (independent of PAC code).
            model = OceanStats.RL.Dyna.Models.ExactModelAdapter(env);
            S = model.NumStates; V = zeros(S, 1);
            for it = 1:20000
                Vn = V;
                for s = 1:S, Vn(s) = OceanStats.RL.DP.BellmanOptimality.backup(model, s, V, gamma); end
                if max(abs(Vn - V)) < 1e-13, V = Vn; break; end
                V = Vn;
            end
            Q = zeros(S, model.NumActions);
            for s = 1:S
                [~, ~, q] = OceanStats.RL.DP.BellmanOptimality.backup(model, s, V, gamma); Q(s, :) = q;
            end
            [~, pi] = max(Q, [], 2);
        end
        function tr = runE3_(strategy, seed)
            env = OceanStats.RL.Env.Classic.ChainEnv();
            ag = OceanStats.RL.PAC.E3Agent(5, 2, 'KnownCount', 5, 'MaxReward', 10, 'Strategy', strategy, 'Seed', seed);
            tr = ag.run(env, 'Steps', 400, 'Seed', 4);
        end
    end
end
