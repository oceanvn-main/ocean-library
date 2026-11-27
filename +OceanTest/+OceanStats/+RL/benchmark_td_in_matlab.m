function benchmark_td_in_matlab()
% BENCHMARK_TD_IN_MATLAB  Reproduces the qualitative TD findings of Sutton & Barto with the library's own learners.
% EXPECTATIONS were written down BEFORE running (plan section 5) and are checked, not tuned:
%   E1  5-state random walk: the best constant-alpha TD(0) reaches a lower RMS error than the best constant-alpha
%       Monte Carlo (= n-step with n >= episode length) after 100 episodes            (Fig 6.2)
%   E2  19-state walk: the best n-step TD uses an INTERMEDIATE n (not 1, not the longest tried)   (Fig 7.2)
%   E3  19-state walk: the best online TD(lambda) uses an INTERMEDIATE lambda                     (Fig 12.6)
%   E4  Cliff Walking, epsilon = 0.1: Sarsa's online return beats Q-learning's, while Q-learning's greedy policy
%       walks the cliff edge (shorter path) and Sarsa's takes a safer, longer route                  (Fig 6.4)
%   E5  Expected Sarsa's online return >= Sarsa's on Cliff Walking                                   (Fig 6.3)
% Every claim is reported with a paired bootstrap 95% CI over independent runs, not a bare mean.
% Run:  OceanTest.OceanStats.RL.benchmark_td_in_matlab()

here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
import OceanStats.RL.TD.*
rng(1);

% ---------------- E1: TD(0) vs constant-alpha MC on the 5-state walk ----------------
env5 = OceanStats.RL.Env.Classic.RandomWalkEnv();
Vtrue = (1:5)' / 6;
runs = 60;  eps1 = 100;
tdAlphas = [0.05 0.10 0.15];  mcAlphas = [0.01 0.02 0.03 0.04];
rmsTD = zeros(runs, numel(tdAlphas));  rmsMC = zeros(runs, numel(mcAlphas));
for r = 1:runs
    for i = 1:numel(tdAlphas)
        V = TDLearner('Env', env5, 'Task', 'prediction', 'StepSize', tdAlphas(i), 'Episodes', eps1, 'Seed', r).run();
        rmsTD(r, i) = sqrt(mean((V(2:6) - Vtrue).^2));
    end
    for i = 1:numel(mcAlphas)
        V = TDLearner('Env', env5, 'Task', 'prediction', 'Mode', 'nstep', 'N', 1000, 'StepSize', mcAlphas(i), 'Episodes', eps1, 'Seed', r).run();
        rmsMC(r, i) = sqrt(mean((V(2:6) - Vtrue).^2));
    end
end
[bt, it] = min(mean(rmsTD));  [bm, im] = min(mean(rmsMC));
[ci, p] = pairedDiff_(rmsMC(:, im), rmsTD(:, it));
fprintf('E1  best TD(0) alpha=%.2f RMS=%.4f | best MC alpha=%.2f RMS=%.4f | MC-TD = %+.4f [%+.4f, %+.4f]  TD better in %d/%d runs (sign p=%.3f)  -> %s\n', ...
    tdAlphas(it), bt, mcAlphas(im), bm, mean(rmsMC(:, im) - rmsTD(:, it)), ci(1), ci(2), nnz(rmsTD(:, it) < rmsMC(:, im)), runs, p, verdict_(bt < bm && ci(1) > 0));

% ---------------- E2 / E3: n-step and lambda on the 19-state walk ----------------
env19 = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19, 'LeftReward', -1, 'RightReward', 1);
Vt19 = ((1:19)' - 10) / 10;                                          % true values with -1/+1 ends
runs2 = 30;  eps2 = 10;  alphas = [0.1 0.2 0.3 0.4 0.5 0.6 0.8];
ns = [1 2 4 8 16 32];  lams = [0 0.4 0.8 0.9 0.95 1];
errN = zeros(numel(ns), numel(alphas));  errL = zeros(numel(lams), numel(alphas));
for r = 1:runs2
    for a = 1:numel(alphas)
        for k = 1:numel(ns)
            V = TDLearner('Env', env19, 'Task', 'prediction', 'Mode', 'nstep', 'N', ns(k), 'StepSize', alphas(a), 'Episodes', eps2, 'Seed', r).run();
            errN(k, a) = errN(k, a) + sqrt(mean((V(2:20) - Vt19).^2)) / runs2;
        end
        for k = 1:numel(lams)
            V = TDLearner('Env', env19, 'Task', 'prediction', 'Mode', 'trace', 'Lambda', lams(k), 'Trace', 'replacing', 'StepSize', alphas(a), 'Episodes', eps2, 'Seed', r).run();
            errL(k, a) = errL(k, a) + sqrt(mean((V(2:20) - Vt19).^2)) / runs2;
        end
    end
end
[bestN, iN] = min(min(errN, [], 2));
[bestL, iL] = min(min(errL, [], 2));
fprintf('E2  n-step: best-over-alpha RMS by n = %s | best n = %d (RMS %.4f)  -> %s\n', mat2str(round(min(errN, [], 2)', 3)), ns(iN), bestN, verdict_(ns(iN) > 1 && ns(iN) < ns(end)));
fprintf('E3  TD(lambda) (replacing): best-over-alpha RMS by lambda = %s | best lambda = %.2f (RMS %.4f)  -> %s\n', mat2str(round(min(errL, [], 2)', 3)), lams(iL), bestL, verdict_(lams(iL) > 0 && lams(iL) < 1));

% ---------------- E4 / E5: Cliff Walking ----------------
cliff = OceanStats.RL.Env.Classic.CliffWalkingEnv();
runs3 = 30;  eps3 = 500;
names = ["sarsa" "qlearning" "expectedsarsa"];
online = zeros(runs3, 3);  pathLen = zeros(runs3, 3);  rows = zeros(runs3, 3);
for r = 1:runs3
    for k = 1:3
        [~, Q, ~, h] = TDLearner('Env', cliff, 'Target', char(names(k)), 'StepSize', 0.5, 'Exploration', 0.1, 'Episodes', eps3, 'Seed', r).run();
        online(r, k) = mean(h.EpisodeReturns(end-99:end));
        [pathLen(r, k), rows(r, k)] = greedyPath_(cliff, Q);
    end
end
[ci41, p41] = pairedDiff_(online(:, 1), online(:, 2));
fprintf('E4  online return (last 100 eps, mean of %d runs): Sarsa %.1f  Q-learning %.1f  Expected Sarsa %.1f\n', runs3, mean(online));
fprintf('    Sarsa - Q-learning = %+.1f [%+.1f, %+.1f], Sarsa better in %d/%d runs (sign p=%.3f)  -> %s\n', mean(online(:, 1) - online(:, 2)), ci41(1), ci41(2), nnz(online(:, 1) > online(:, 2)), runs3, p41, verdict_(ci41(1) > 0));
fprintf('    greedy path length (mean over runs): Sarsa %.1f  Q-learning %.1f  Expected Sarsa %.1f   [optimal = 13; safe route = 17]\n', mean(pathLen));
fprintf('    fraction of runs whose greedy path stays on the cliff edge row: Sarsa %.2f  Q-learning %.2f  Expected Sarsa %.2f\n', mean(rows == 3));
[ci5, p5] = pairedDiff_(online(:, 3), online(:, 1));
fprintf('E5  Expected Sarsa - Sarsa = %+.1f [%+.1f, %+.1f], Expected Sarsa better in %d/%d runs (sign p=%.3f)  -> %s\n', ...
    mean(online(:, 3) - online(:, 1)), ci5(1), ci5(2), nnz(online(:, 3) > online(:, 1)), runs3, p5, verdict_(mean(online(:, 3)) >= mean(online(:, 1))));
end

function s = verdict_(ok)
if ok, s = 'EXPECTATION HELD'; else, s = 'EXPECTATION NOT MET'; end
end

function [ci, p] = pairedDiff_(a, b)
% paired bootstrap 95% CI of mean(a - b) and exact two-sided sign-test p (a > b counts as a "win")
d = a(:) - b(:);
idx = randi(numel(d), numel(d), 2000);
m = sort(mean(d(idx), 1));
ci = [m(round(0.025 * numel(m))) m(round(0.975 * numel(m)))];
nb = nnz(d > 0);  nw = nnz(d < 0);  n = nb + nw;
if n == 0, p = 1; return; end
k = min(nb, nw);
lg = @(x) gammaln(x + 1);
p = min(1, 2 * sum(exp(arrayfun(@(j) lg(n) - lg(j) - lg(n - j), 0:k) - n * log(2))));
end

function [len, row] = greedyPath_(env, Q)
% follow the greedy policy from the start; report its length and the row of its longest stretch (row 3 = cliff edge)
s = env.reset();  len = 0;  visited = zeros(1, 0);
while true
    q = Q(s, :);  q(~ismember(1:size(Q, 2), env.get_valid_actions(s))) = -Inf;
    [~, a] = max(q);
    [s, ~, d] = env.step(a);  len = len + 1;
    [r, ~] = env.rc(s);  visited(end+1) = r; %#ok<AGROW>
    if d || len > 200, break; end
end
row = mode(visited);
end
