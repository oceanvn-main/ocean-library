classdef TestFirstVisitScan < matlab.unittest.TestCase
    % TESTFIRSTVISITSCAN  Regression for the first-visit bug (found 2026-09-28, fixed 2026-10-01): the Monte Carlo
    % first-visit strategies scanned the episode backwards with a "visited" flag, which keeps the LAST visit of a state
    % (or state-action pair) instead of the first (Sutton & Barto pp. 92, 99, 101). They now mark first visits going
    % forward with FirstVisitPredictionStrategy.firstVisits_ and the backward return loop updates only there.

    methods (Test)
        function firstVisitsAreTheEarliestOccurrences(tc)
            f = @(k, n) OceanStats.RL.MonteCarlo.Strategies.FirstVisitPredictionStrategy.firstVisits_(k, n);
            tc.verifyEqual(f([3 1 3 2 1 3], 3)', logical([1 1 0 1 0 0]));
            tc.verifyEqual(f([0 2 0 2], 2)', logical([0 1 0 0]), 'index 0 (outside the table) is never a visit');
            tc.verifyEqual(f(zeros(1, 0), 4), false(0, 1));
        end

        function theReturnUsedIsTheOneFromTheFirstVisit(tc)
            % Episode S = [A B A], rewards [1 10 100], gamma = 1: returns G = [111 110 100]. First visit of A is t = 1,
            % so V(A) must get 111 (the old backward flag gave 100, the return from the LAST visit).
            states = [1 2 1];  rewards = [1 10 100];
            first = OceanStats.RL.MonteCarlo.Strategies.FirstVisitPredictionStrategy.firstVisits_(states, 2);
            G = 0;  got = nan(1, 2);
            for t = numel(states):-1:1
                G = G + rewards(t);
                if first(t), got(states(t)) = G; end
            end
            tc.verifyEqual(got, [111 110]);
        end

        function strategiesUseTheForwardScan(tc)
            % The three first-visit strategies no longer contain a backward "visited" flag.
            root = fileparts(which('OceanStats.RL.MonteCarlo.Strategies.FirstVisitPredictionStrategy'));
            for f = ["FirstVisitPredictionStrategy" "ExploringStartsStrategy" "OnPolicyControlStrategy"]
                txt = fileread(fullfile(root, f + ".m"));
                tc.verifyEmpty(regexp(txt, 'visited\(', 'once'), f);
                tc.verifyNotEmpty(regexp(txt, 'firstVisits_', 'once'), f);
            end
        end
    end
end
