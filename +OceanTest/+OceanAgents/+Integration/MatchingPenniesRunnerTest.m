classdef MatchingPenniesRunnerTest < OceanTest.BaseTestCase
    % MATCHINGPENNIESRUNNERTEST  Two independent agents, each a PolicyMind(RandomLegalPolicy) with its own stream,
    % play repeated matching pennies (OceanStats.GameTheory NormalFormGame) through the real
    % RepeatedMatrixGameEnv and JointRunner.
    %
    % Checks the seams the unit tests could not: final-round rewards reach both agents although the horizon frame has
    % no live slot (B2/B1 hand-off), zero-sum payoffs per episode, the uniform-play expected payoff 0 within a
    % justified band, and evaluation-order invariance with real components (C03, C20, IG05).

    properties (Constant)
        Rounds = 50
        Episodes = 100
    end

    methods (Static)
        function [runner, hosts] = build(seed, order)
            if nargin < 2, order = []; end
            A = [1 -1; -1 1];
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', A, 'PayoffB', -A);
            C = OceanTest.OceanAgents.Integration.MatchingPenniesRunnerTest;
            env = OceanAgents.Env.Reference.RepeatedMatrixGameEnv(g, 'Roster', ["p1" "p2"], 'Rounds', C.Rounds);
            hosts = {OceanAgents.Runtime.AgentHost("p1", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy())), ...
                     OceanAgents.Runtime.AgentHost("p2", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy()))};
            runner = OceanAgents.Runtime.JointRunner(env, hosts, 'Seed', seed, 'Order', order);
        end
    end

    methods (Test)
        function everyRoundPaidAndZeroSum(tc)
            runner = tc.build(21);
            r = runner.run(tc.Episodes);
            tc.assertEqual(r.Status, "Completed", 'run failed');
            tc.verifyEqual(r.NumEpisodes, tc.Episodes);
            tc.verifyEqual(r.EpisodeLengths, repmat(tc.Rounds, tc.Episodes, 1));
            tc.verifyEqual(r.StepCalls, tc.Episodes * tc.Rounds);        % one joint step per frame, no extra steps
            tc.verifyEqual(r.EpisodeReturns(:, 2), -r.EpisodeReturns(:, 1));
            % each round pays +-1, so an episode return has the parity of Rounds and |return| <= Rounds; if the
            % final round's reward were dropped, returns would have the wrong parity
            tc.verifyTrue(all(mod(r.EpisodeReturns(:, 1) - tc.Rounds, 2) == 0), 'final-round reward missing');
            tc.verifyLessThanOrEqual(abs(r.EpisodeReturns(:, 1)), tc.Rounds);
            % uniform independent play: E[u1] = 0, Var[u1] = 1 per round; mean over n = 5000 rounds has
            % SE = 1/sqrt(5000) = 0.01414; band 4*SE
            n = tc.Episodes * tc.Rounds;
            tc.verifyLessThanOrEqual(abs(sum(r.EpisodeReturns(:, 1)) / n), 4 / sqrt(n));
        end

        function evaluationOrderDoesNotMatter(tc)
            rA = tc.build(33, [1 2]).run(10);
            rB = tc.build(33, [2 1]).run(10);
            tc.verifyEqual(rB.EpisodeReturns, rA.EpisodeReturns);
            tc.verifyEqual(rB.EpisodeLengths, rA.EpisodeLengths);
        end

        function agentsShareNoMutableState(tc)
            [~, hosts] = tc.build(1);
            tc.verifyEmpty(OceanAgents.Runtime.IsolationChecker.sharedHandles(hosts{1}, hosts{2}));
        end
    end
end
