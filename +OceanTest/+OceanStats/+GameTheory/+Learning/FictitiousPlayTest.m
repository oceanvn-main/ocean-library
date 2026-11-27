classdef FictitiousPlayTest < matlab.unittest.TestCase
    % FICTITIOUSPLAYTEST  OceanStats.GameTheory.Learning (EmpiricalBelief, FictitiousPlay) and
    % OceanStats.GameTheory.Solvers.FictitiousPlaySolver against hand-derived results and known theorems:
    % Laplace/Dirichlet beliefs, best responses, N-player expected payoffs by brute force, convergence in zero-sum
    % games (Robinson 1951), the dominant-strategy outcome of the prisoner's dilemma, and exploiting a biased opponent.

    methods (Test)
        function beliefIsTheSmoothedEmpiricalFrequency(tc)
            b = OceanStats.GameTheory.Learning.EmpiricalBelief([2 3], 'Prior', 1);
            tc.verifyEqual(b.probabilities(1), [0.5 0.5]);
            b.observe([1 3]);  b.observe([1 NaN]);  b.observe([2 3]);
            tc.verifyEqual(b.probabilities(1), [3 2] / 5, 'AbsTol', 1e-15);          % (1+2)/(2+3), (1+1)/(2+3)
            tc.verifyEqual(b.probabilities(2), [1 1 3] / 5, 'AbsTol', 1e-15);        % opponent 2 seen twice
            tc.verifyEqual(b.Observations, [3 2]);
            z = OceanStats.GameTheory.Learning.EmpiricalBelief(3, 'Prior', 0);
            tc.verifyEqual(z.probabilities(1), [1 1 1] / 3, 'AbsTol', 1e-15);         % nothing seen: uniform
            tc.verifyError(@() b.observe([3 1]), 'OceanStats:GameTheory:BadBelief');
        end

        function bestResponseExploitsABiasedOpponent(tc)
            % the parity game's guess: payoff 1 when my guess matches the opponent's put (even = 1, odd = 2)
            fp = OceanStats.GameTheory.Learning.FictitiousPlay(eye(2), 1, 'Prior', 1);
            rs = RandStream('mrg32k3a', 'Seed', 1);
            for t = 1:200
                fp.observe([NaN, 1 + (rand(rs) > 0.9)]);                              % opponent: even 90%
            end
            U = fp.expectedPayoffs();
            b = fp.beliefs();
            tc.verifyEqual(U, b{1}, 'AbsTol', 1e-15, 'with an identity payoff U is the belief itself');
            tc.verifyGreaterThan(U(1), 0.85);
            tc.verifyEqual(fp.responseProbabilities(), [1 0]);                         % guess even
            tc.verifyEqual(fp.act(rs), 1);
        end

        function expectedPayoffsMatchBruteForceForThreePlayers(tc)
            rs = RandStream('mrg32k3a', 'Seed', 4);
            U3 = {randn(rs, 2, 3, 4), randn(rs, 2, 3, 4), randn(rs, 2, 3, 4)};
            g = OceanStats.GameTheory.Core.NormalFormGame('PayoffTensors', U3);
            for i = 1:3
                fp = OceanStats.GameTheory.Learning.FictitiousPlay(g, i, 'Prior', 0.5);
                for t = 1:7, fp.observe([randi(rs, 2), randi(rs, 3), randi(rs, 4)]); end
                P = cell(1, 3);  opp = setdiff(1:3, i);  b = fp.beliefs();
                for k = 1:2, P{opp(k)} = b{k}; end
                m = size(U3{i}, i);  ref = zeros(1, m);
                [A, C, D] = ndgrid(1:2, 1:3, 1:4);
                prof = [A(:) C(:) D(:)];
                w = ones(size(prof, 1), 1);
                for k = opp, w = w .* reshape(P{k}(prof(:, k)), [], 1); end
                u = U3{i}(:);
                for s = 1:m, ref(s) = sum(w(prof(:, i) == s) .* u(prof(:, i) == s)); end
                tc.verifyEqual(fp.expectedPayoffs(), ref, 'AbsTol', 1e-12, sprintf('player %d', i));
            end
        end

        function zeroSumGamesConverge(tc)
            % Robinson (1951): empirical frequencies converge to the equilibrium of a two-player zero-sum game
            mp = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', [1 -1; -1 1], 'PayoffB', [-1 1; 1 -1]);
            rps = [0 -1 1; 1 0 -1; -1 1 0];
            rg = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', rps, 'PayoffB', -rps);
            for g = {mp, rg}
                s = OceanStats.GameTheory.Solvers.FictitiousPlaySolver('Rounds', 4000);
                r = s.solve(g{1});
                m = numel(r.Frequencies{1});
                tc.verifyEqual(r.Frequencies{1}, ones(1, m) / m, 'AbsTol', 0.03);
                tc.verifyEqual(r.Frequencies{2}, ones(1, m) / m, 'AbsTol', 0.03);
                h = s.getHistory();
                tc.verifyLessThan(h.Exploitability(end), 0.1);
                tc.verifyLessThan(h.Exploitability(end), h.Exploitability(10));
            end
        end

        function dominantStrategiesAreFoundAtOnce(tc)
            pd = OceanStats.GameTheory.Core.NormalFormGame('PayoffA', [3 0; 5 1], 'PayoffB', [3 5; 0 1]);
            r = OceanStats.GameTheory.Solvers.FictitiousPlaySolver('Rounds', 50).solve(pd);
            tc.verifyEqual(r.Profiles(2:end, :), repmat([2 2], 49, 1), 'defect is dominant');
            tc.verifyEqual(r.Exploitability, 0, 'AbsTol', 0.1);
            tc.verifyEqual(OceanStats.GameTheory.Solvers.FictitiousPlaySolver.exploitability(pd, {[0 1], [0 1]}), 0);
        end

        function smoothResponseIsALogit(tc)
            fp = OceanStats.GameTheory.Learning.FictitiousPlay(eye(2), 1, 'Response', "smooth", 'Temperature', 0.5);
            fp.observe([NaN 1]);  fp.observe([NaN 1]);                                 % belief [3 1]/4 with prior 1
            U = fp.expectedPayoffs();
            tc.verifyEqual(U, [0.75 0.25], 'AbsTol', 1e-15);
            e = exp(U / 0.5);
            tc.verifyEqual(fp.responseProbabilities(), e / sum(e), 'AbsTol', 1e-12);
            tc.verifyError(@() OceanStats.GameTheory.Learning.FictitiousPlay({}, 1), 'OceanStats:GameTheory:BadPayoff');
        end
    end
end
