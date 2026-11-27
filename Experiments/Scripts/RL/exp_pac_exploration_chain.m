%% PAC exploration on the Chain problem: R-max and E³ against ε-greedy Q-learning
% References: R-max -- Brafman & Tennenholtz (2002), discounted form and pseudo-code from Strehl, Li & Littman
%             (2009), Algorithm 1;  E³ -- Kearns & Singh (2002), Sections 5.5 and 6;
%             Chain problem -- Strens (2000), Dearden, Friedman & Russell (1998).
%
% PURPOSE:
%   On the Chain the immediate reward of going back (2) hides the much larger reward at the end (10). Myopic
%   exploration (ε-greedy) tends to settle on going back; R-max (optimism in the face of uncertainty) and E³
%   (explicit escape to unknown states) explore the chain systematically and then exploit.
%
% METRIC: cumulative reward over the run (mean over seeds) against the optimal policy's long-run reward
%   rate (exact, from the stationary distribution of the optimal policy) and against always going back.
%
% Not in the papers: the ε-greedy Q-learning baseline (step size 0.1, ε = 0.1) and the parameter choices
%   (KnownCount m = 20, γ = 0.95) are ours; theoretical m values are far larger.

clearvars -except QuickMode; clc;
if ~exist('QuickMode', 'var'), QuickMode = false; end
if QuickMode, steps = 3000; seeds = 3; else, steps = 10000; seeds = 10; end
gamma = 0.95; m = 20;
env = OceanStats.RL.Env.Classic.ChainEnv();
S = 5; A = 2;

methods = {
    "R-max",                        @(k) OceanStats.RL.PAC.RMaxAgent(S, A, 'KnownCount', m, 'MaxReward', 10, 'DiscountFactor', gamma)
    "E3 (exploration-biased)",      @(k) OceanStats.RL.PAC.E3Agent(S, A, 'KnownCount', m, 'MaxReward', 10, 'DiscountFactor', gamma)
    "E3 (schedule \alpha(t)=1/t)",  @(k) OceanStats.RL.PAC.E3Agent(S, A, 'KnownCount', m, 'MaxReward', 10, 'DiscountFactor', gamma, ...
                                         'Strategy', "schedule", 'Seed', k)
    };
nM = size(methods, 1) + 1;                         % + Q-learning baseline
cum = zeros(nM, steps, seeds);
for k = 1:seeds
    for i = 1:nM - 1
        ag = methods{i, 2}(k);
        tr = ag.run(env, 'Steps', steps, 'Seed', k);
        cum(i, :, k) = cumsum(tr.Rewards);
    end
    cum(nM, :, k) = cumsum(qLearning(env, S, A, steps, gamma, k));
end
names = [string(methods(:, 1)); "\epsilon-greedy Q-learning (\epsilon = 0.1)"];

% Optimal long-run reward rate (always "a") and the myopic loop (always "b"), exact from the model.
rateOpt = rewardRate(env, ones(S, 1)); rateBack = rewardRate(env, 2 * ones(S, 1));
fprintf('Optimal reward rate %.3f per step, always-back %.3f per step\n', rateOpt, rateBack);
for i = 1:nM
    fprintf('  %-34s total reward %8.1f +- %6.1f   (last 1000 steps: %.3f per step)\n', names(i), ...
        mean(cum(i, end, :)), std(cum(i, end, :)) / sqrt(seeds), ...
        mean(cum(i, end, :) - cum(i, end - 999, :)) / 1000);
end

%% Plot (OceanViz theme)
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'PAC exploration on the Chain', 'Position', [100 100 950 560]);
ax = axes(fig); viz.styleAxes(ax); hold(ax, 'on'); grid(ax, 'on');
cols = viz.getColorSequence(nM);
t = 1:steps;
for i = 1:nM
    plot(ax, t, mean(cum(i, :, :), 3), 'LineWidth', 1.8, 'Color', cols(i, :), 'DisplayName', names(i));
end
plot(ax, t, rateOpt * t, '--', 'Color', ax.XColor, 'DisplayName', 'optimal policy (expected)');
plot(ax, t, rateBack * t, ':', 'Color', ax.XColor, 'DisplayName', 'always go back (expected)');
xlabel(ax, 'steps'); ylabel(ax, sprintf('cumulative reward (mean of %d seeds)', seeds));
title(ax, 'Chain (Strens 2000): R-max and E^3 vs \epsilon-greedy');
lgd = legend(ax, 'Location', 'northwest'); lgd.Color = ax.Color; lgd.TextColor = ax.XColor; lgd.EdgeColor = ax.GridColor;

%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================
function rate = rewardRate(env, pi)
    % Long-run expected reward per step of a deterministic policy: stationary distribution d of P_pi, d' r_pi.
    S = numel(pi); P = zeros(S); r = zeros(S, 1);
    for s = 1:S
        [n, p, rr] = env.get_transitions(s, pi(s));
        for k = 1:numel(n), P(s, n(k)) = P(s, n(k)) + p(k); end
        r(s) = p' * rr;
    end
    [V, D] = eig(P'); [~, k] = min(abs(diag(D) - 1));
    d = real(V(:, k)); d = d / sum(d);
    rate = d' * r;
end

function rewards = qLearning(env, S, A, steps, gamma, seed)
    % ε-greedy one-step Q-learning baseline (S&B Section 6.5), own seeded stream.
    rs = RandStream('mrg32k3a', 'Seed', 1000 + seed);
    Q = zeros(S, A); alpha = 0.1; epsilon = 0.1;
    [s, ~] = env.reset('Seed', seed); rewards = zeros(1, steps);
    for t = 1:steps
        if rand(rs) < epsilon, a = randi(rs, A); else, [~, a] = max(Q(s, :)); end
        [s2, r] = env.step(a);
        Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s2, :)) - Q(s, a));
        rewards(t) = r; s = s2;
    end
end
