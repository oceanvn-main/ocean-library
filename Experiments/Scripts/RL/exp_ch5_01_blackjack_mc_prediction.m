% EXP_CH5_01_BLACKJACK_MC_PREDICTION
%   First-Visit Monte Carlo Prediction on Blackjack (Sutton & Barto Example 5.1).
%   Evaluates V(s) under a hard-coded threshold policy and demonstrates
%   that same policy actually playing the game.
%
%   ENVIRONMENT:     BlackjackEnv — 200 states, infinite deck, dealer hits < 17
%   POLICY:          Hard-coded threshold: HIT if player_sum < 20, else STICK
%                    (Suboptimal but well-defined for pedagogical MC prediction)
%   ALGORITHM:       First-Visit MC Prediction → V(s) state-value table
%   HYPERPARAMETERS:
%     N_PREDICT_SMALL  = 10 000 episodes  (early / noisy estimate, Fig 5.1 top)
%     N_PREDICT_LARGE  = 500 000 episodes (converged estimate,     Fig 5.1 bottom)
%     N_PLAY_GAMES     = 50 000 games     (empirical policy evaluation)
%     Gamma            = 1.0              (episodic, no discounting)
%   ASSUMPTIONS:
%     - Infinite deck: cards sampled i.i.d. uniformly from {1..10}
%     - Natural blackjack is NOT treated specially (simplification vs. original rules)
%     - Hard-code rule "hit < 20" is deliberately suboptimal — not learned
%     - Win rate vs. dealer is computed empirically, not analytically
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   OUTPUTS:
%     - 4-panel V(s) surface comparison (10K vs 500K episodes, ace/no-ace)
%     - Policy play statistics (win/draw/loss from 50K games)
%     - Printed summary block
%
%   Reference: Sutton & Barto (2018) §5.1, Example 5.1, Figure 5.1
%
%   Package context: OceanStats.RL — Ch.5 Monte Carlo Methods

clc; clear; close all;
rng(42, 'twister');   % Reproducibility

% =========================================================================
% === 1. HYPERPARAMETERS (all explicit — no silent textbook copies) =======
% =========================================================================
N_PREDICT_SMALL  = 10000;    % Episodes for early V(s) estimate (Fig 5.1 top)
N_PREDICT_LARGE  = 500000;   % Episodes for converged V(s) estimate (Fig 5.1 bottom)
N_PLAY_GAMES     = 50000;    % Games to evaluate the policy empirically
GAMMA            = 1.0;      % Discount factor (1.0 = episodic, no discounting)
CHECK_INTERVAL   = 5000;     % Progress snapshot interval

% ASSUMPTION: hit threshold = 20 (player hits until sum >= 20)
% Justification: This matches S&B Example 5.1 exactly.
% It is NOT the optimal policy — the experiment evaluates it, not optimises it.
HIT_THRESHOLD    = 20;

% =========================================================================
% === 2. ENVIRONMENT SETUP ================================================
% =========================================================================
fprintf('[Setup] Constructing BlackjackEnv...\n');
env = OceanStats.RL.Env.Classic.BlackjackEnv();

fprintf('  States  : %d  ([player_sum 12-21] x [dealer 1-10] x [ace 0/1])\n', ...
    env.get_num_states());
fprintf('  Actions : %d  (0=stick, 1=hit)\n', env.get_num_actions());

% =========================================================================
% === 3. HARD-CODED TARGET POLICY =========================================
% =========================================================================
% ASSUMPTION: The policy is fixed by the experimenter, not learned by the agent.
% It is injected as a function handle conforming to the IEpisodeEnvironment
% generate_episode() contract: @(state, env) → action_value.
%
% State vector = [player_sum, dealer_showing, usable_ace]
%   action = 1 (HIT)  if player_sum < HIT_THRESHOLD
%   action = 0 (STICK) otherwise
%
% This is a pure threshold rule — no look at dealer card, no ace consideration.

target_policy = @(state, ~) double(state(1) < HIT_THRESHOLD);

fprintf('[Policy] Hard-coded threshold: HIT if sum < %d, else STICK\n', ...
    HIT_THRESHOLD);

% =========================================================================
% === 4. MC PREDICTION — SMALL (10K episodes) =============================
% =========================================================================
fprintf('\n[Phase 1] First-Visit MC Prediction — %d episodes...\n', N_PREDICT_SMALL);

rng(42, 'twister');   % Reset seed for reproducible small run

solver_small = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',    'FirstVisitPrediction', ...
    'Gamma',         GAMMA,                 ...
    'NumEpisodes',   N_PREDICT_SMALL,        ...
    'TargetPolicy',  target_policy,          ...
    'CheckInterval', CHECK_INTERVAL,         ...
    'Verbose',       true                    ...
);
[V_small, ~, ~, history_small] = solver_small.run();

fprintf('  V(s) range: [%.3f, %.3f]\n', min(V_small), max(V_small));
fprintf('  States visited: %d / %d\n', ...
    sum(solver_small.NVTable > 0), env.get_num_states());

% =========================================================================
% === 5. MC PREDICTION — LARGE (500K episodes) ============================
% =========================================================================
fprintf('\n[Phase 2] First-Visit MC Prediction — %d episodes...\n', N_PREDICT_LARGE);

rng(42, 'twister');   % Reset seed — deterministic large run

solver_large = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
    'Algorithm',    'FirstVisitPrediction', ...
    'Gamma',         GAMMA,                 ...
    'NumEpisodes',   N_PREDICT_LARGE,        ...
    'TargetPolicy',  target_policy,          ...
    'CheckInterval', N_PREDICT_LARGE / 5,   ...
    'Verbose',       true                    ...
);
[V_large, ~, ~, history_large] = solver_large.run();

fprintf('  V(s) range: [%.3f, %.3f]\n', min(V_large), max(V_large));
fprintf('  States visited: %d / %d\n', ...
    sum(solver_large.NVTable > 0), env.get_num_states());

% =========================================================================
% === 6. POLICY PLAY EVALUATION (hard-coded policy plays N_PLAY_GAMES) ====
% =========================================================================
% Computes win/draw/loss rates from actual gameplay under the hard-coded policy.
% ALL statistics derived from observed outcomes — no formulas or shortcuts.

fprintf('\n[Phase 3] Playing %d games under hard-coded policy...\n', N_PLAY_GAMES);

rng(42, 'twister');   % Separate seed reset for play evaluation

play_rewards = zeros(N_PLAY_GAMES, 1);   % Pre-allocated, no dynamic growth

for game = 1:N_PLAY_GAMES
    ep = env.generate_episode(target_policy, 'MaxLen', 30);
    play_rewards(game) = ep.Rewards(end);   % Terminal reward (+1/0/-1)
end

% Derive all statistics from observations (no hard-coding)
n_wins   = sum(play_rewards ==  1);
n_draws  = sum(play_rewards ==  0);
n_losses = sum(play_rewards == -1);

win_rate   = n_wins   / N_PLAY_GAMES;
draw_rate  = n_draws  / N_PLAY_GAMES;
loss_rate  = n_losses / N_PLAY_GAMES;
avg_return = mean(play_rewards);

fprintf('  Win:  %5d (%.1f%%)\n', n_wins,   win_rate  * 100);
fprintf('  Draw: %5d (%.1f%%)\n', n_draws,  draw_rate * 100);
fprintf('  Loss: %5d (%.1f%%)\n', n_losses, loss_rate * 100);
fprintf('  Avg Return: %+.4f\n', avg_return);

% =========================================================================
% === 7. VISUALIZATION ====================================================
% =========================================================================
fprintf('\n[Viz] Rendering figures...\n');

viz = OceanViz.Core('Theme', 'Dark');
C   = viz.Colors;   % Theme: .Background, .Text, .Grid, palette

all_states = env.get_all_states();   % (200 x 3): [player_sum, dealer, usable_ace]

% ----- Helper: extract V(s) as 2D grid for one ace condition ---------------
function V_grid = extract_v_grid(V, all_states, usable_ace_flag)
    ps_range = 12:21;
    dc_range = 1:10;
    V_grid   = nan(length(ps_range), length(dc_range));
    for row = 1:size(all_states, 1)
        ps = all_states(row, 1);
        dc = all_states(row, 2);
        ua = all_states(row, 3);
        if ua == usable_ace_flag && ps >= 12 && ps <= 21 && dc >= 1 && dc <= 10
            V_grid(ps - 11, dc) = V(row);
        end
    end
end

ps_ticks = 12:21;
dc_ticks = 1:10;

% ----- FIGURE 1: V(s) Surfaces — 4 panels replicating S&B Fig 5.1 --------
fig1 = figure('Name', 'MC Prediction — V(s) Surfaces (S&B Fig 5.1)', ...
    'Color', C.Background, 'Position', [40, 40, 1400, 850]);

panel_configs = { ...
    V_small, 'After 10 000 Episodes',  0, 'No Usable Ace';  ...
    V_small, 'After 10 000 Episodes',  1, 'Usable Ace';     ...
    V_large, 'After 500 000 Episodes', 0, 'No Usable Ace';  ...
    V_large, 'After 500 000 Episodes', 1, 'Usable Ace';     ...
};

subplot_idx = [1, 2, 3, 4];
for k = 1:4
    V_k     = panel_configs{k, 1};
    ep_str  = panel_configs{k, 2};
    ua_flag = panel_configs{k, 3};
    ace_str = panel_configs{k, 4};

    ax_k = subplot(2, 2, subplot_idx(k), 'Parent', fig1);
    V_grid_k = extract_v_grid(V_k, all_states, ua_flag);

    [X, Y] = meshgrid(dc_ticks, ps_ticks);
    h_surf = surf(ax_k, X, Y, V_grid_k, 'EdgeColor', 'none', 'FaceAlpha', 0.90);
    h_surf.DisplayName = sprintf('V(s) | Policy: HIT < %d', HIT_THRESHOLD);

    colormap(ax_k, turbo);
    cb = colorbar(ax_k);
    cb.Label.String  = 'V(s)';
    cb.Label.Color   = C.Text;
    cb.Color         = C.Text;
    clim(ax_k, [-1, 1]);

    view(ax_k, [-35, 30]);
    grid(ax_k, 'on');
    set(ax_k, 'Color', C.Background, 'XColor', C.Text, ...
              'YColor', C.Text, 'ZColor', C.Text, 'GridColor', C.Grid);

    xlabel(ax_k, 'Dealer Showing', 'Color', C.Text, 'FontSize', 9);
    ylabel(ax_k, 'Player Sum',     'Color', C.Text, 'FontSize', 9);
    zlabel(ax_k, 'V(s)',           'Color', C.Text, 'FontSize', 9);
    title(ax_k, sprintf('%s — %s', ep_str, ace_str), ...
        'Color', C.Text, 'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');
    xticks(ax_k, dc_ticks);
    yticks(ax_k, ps_ticks);
    zlim(ax_k, [-1, 1]);

    legend(ax_k, h_surf, 'Location', 'northwest', ...
        'TextColor', C.Text, 'Color', C.Background, 'EdgeColor', 'none');
end

sgtitle(fig1, ...
    sprintf('First-Visit MC Prediction | Policy: HIT if sum < %d (S&B Fig 5.1)', ...
    HIT_THRESHOLD), ...
    'Color', C.Text, 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');

% ----- FIGURE 2: Policy Play Statistics -----------------------------------
fig2 = figure('Name', 'Hard-Coded Policy — Play Statistics', ...
    'Color', C.Background, 'Position', [80, 80, 1000, 450]);

% Panel 1: Win/Draw/Loss bar chart
ax2a = subplot(1, 2, 1, 'Parent', fig2);
bar_vals   = [win_rate, draw_rate, loss_rate] * 100;
bar_labels = {'Win', 'Draw', 'Loss'};
bar_colors = [0.20, 0.82, 0.60;   % Win  — teal-green
              0.70, 0.70, 0.20;   % Draw — golden
              0.90, 0.25, 0.40];  % Loss — rose

h_bars = bar(ax2a, bar_vals, 'FaceColor', 'flat');
h_bars.CData = bar_colors;
h_bars.DisplayName = sprintf('Policy: HIT < %d (N=%d games)', HIT_THRESHOLD, N_PLAY_GAMES);

set(ax2a, 'XTickLabel', bar_labels, 'XTickLabelRotation', 0, ...
    'Color', C.Background, 'XColor', C.Text, 'YColor', C.Text, 'GridColor', C.Grid);
grid(ax2a, 'on');
ylim(ax2a, [0, 60]);
ylabel(ax2a, 'Rate (%)', 'Color', C.Text, 'FontSize', 10);
title(ax2a, sprintf('Outcome Distribution\n(N = %d games)', N_PLAY_GAMES), ...
    'Color', C.Text, 'FontSize', 11, 'FontWeight', 'bold');

% Add value labels on bars
for b = 1:3
    text(ax2a, b, bar_vals(b) + 1.0, sprintf('%.1f%%', bar_vals(b)), ...
        'HorizontalAlignment', 'center', 'Color', C.Text, 'FontSize', 10, 'FontWeight', 'bold');
end

legend(ax2a, h_bars, 'Location', 'north', ...
    'TextColor', C.Text, 'Color', C.Background, 'EdgeColor', 'none');

% Panel 2: Return histogram (shows +1/0/-1 distribution)
ax2b = subplot(1, 2, 2, 'Parent', fig2);
bins    = [-1.5, -0.5, 0.5, 1.5];
counts  = histcounts(play_rewards, bins);
h_hist  = bar(ax2b, [-1, 0, 1], counts / N_PLAY_GAMES * 100, 0.6, 'FaceColor', 'flat');
h_hist.CData = bar_colors;
h_hist.DisplayName = sprintf('Reward distribution (N=%d)', N_PLAY_GAMES);

set(ax2b, 'Color', C.Background, 'XColor', C.Text, 'YColor', C.Text, 'GridColor', C.Grid);
grid(ax2b, 'on');
xticks(ax2b, [-1, 0, 1]);
xticklabels(ax2b, {'-1 (Loss)', '0 (Draw)', '+1 (Win)'});
ylabel(ax2b, 'Rate (%)', 'Color', C.Text, 'FontSize', 10);
title(ax2b, sprintf('Reward Distribution\nAvg Return = %+.4f', avg_return), ...
    'Color', C.Text, 'FontSize', 11, 'FontWeight', 'bold');

% Add value labels
reward_vals = [-1, 0, 1];
for b = 1:3
    pct = counts(b) / N_PLAY_GAMES * 100;
    text(ax2b, reward_vals(b), pct + 0.8, sprintf('%.1f%%', pct), ...
        'HorizontalAlignment', 'center', 'Color', C.Text, 'FontSize', 10, 'FontWeight', 'bold');
end

legend(ax2b, h_hist, 'Location', 'north', ...
    'TextColor', C.Text, 'Color', C.Background, 'EdgeColor', 'none');

sgtitle(fig2, ...
    sprintf('Hard-Coded Policy Play Evaluation | HIT if sum < %d', HIT_THRESHOLD), ...
    'Color', C.Text, 'FontSize', 13, 'FontWeight', 'bold');

% ----- FIGURE 3: V(s) convergence — avg return per checkpoint -------------
fig3 = figure('Name', 'MC Prediction — Convergence', ...
    'Color', C.Background, 'Position', [120, 120, 800, 400]);

ax3 = axes('Parent', fig3, 'Color', C.Background, ...
    'XColor', C.Text, 'YColor', C.Text, 'GridColor', C.Grid);
hold(ax3, 'on');

h_conv = plot(ax3, history_large.CheckPoints, history_large.AvgReturn, ...
    'Color', [0.20, 0.85, 0.90], 'LineWidth', 1.8);
h_conv.DisplayName = 'Avg Return per Episode';

% 5-point moving average smoothing
w       = ones(1, 5) / 5;
smth    = conv(history_large.AvgReturn, w, 'same');
h_smth  = plot(ax3, history_large.CheckPoints, smth, ...
    'Color', [1.0, 0.60, 0.10], 'LineWidth', 2.5, 'LineStyle', '--');
h_smth.DisplayName = 'Smoothed (5-pt MA)';

% Empirical win rate baseline from play evaluation
h_base = yline(ax3, avg_return, ':', 'Color', [0.70, 0.70, 0.70], ...
    'LineWidth', 1.5, 'DisplayName', sprintf('Empirical avg return (%.4f)', avg_return));

grid(ax3, 'on');
xlabel(ax3, 'Episodes', 'Color', C.Text, 'FontSize', 10);
ylabel(ax3, 'Average Return per Episode', 'Color', C.Text, 'FontSize', 10);
title(ax3, sprintf('MC Prediction Convergence | Policy: HIT < %d | gamma=%.1f', ...
    HIT_THRESHOLD, GAMMA), 'Color', C.Text, 'FontSize', 11, 'FontWeight', 'bold');
legend(ax3, [h_conv, h_smth, h_base], 'Location', 'southeast', ...
    'TextColor', C.Text, 'Color', C.Background, 'EdgeColor', 'none');
hold(ax3, 'off');

% =========================================================================
% === 8. SUMMARY PRINTOUT =================================================
% =========================================================================
fprintf('\n===============================================\n');
fprintf('  EXPERIMENT SUMMARY: exp_ch5_01_blackjack_mc_prediction\n');
fprintf('===============================================\n');
fprintf('  Environment    : BlackjackEnv\n');
fprintf('  States         : %d  (10 sums x 10 dealer x 2 ace)\n', env.get_num_states());
fprintf('  Policy         : Hard-coded — HIT if sum < %d\n', HIT_THRESHOLD);
fprintf('  Algorithm      : First-Visit MC Prediction\n');
fprintf('  Gamma          : %.2f\n', GAMMA);
fprintf('-----------------------------------------------\n');
fprintf('  V(s) estimate after %6d episodes:\n', N_PREDICT_SMALL);
fprintf('    Range: [%.3f, %.3f]\n', min(V_small), max(V_small));
fprintf('    States visited: %d / %d\n', sum(solver_small.NVTable > 0), env.get_num_states());
fprintf('  V(s) estimate after %6d episodes:\n', N_PREDICT_LARGE);
fprintf('    Range: [%.3f, %.3f]\n', min(V_large), max(V_large));
fprintf('    States visited: %d / %d\n', sum(solver_large.NVTable > 0), env.get_num_states());
fprintf('-----------------------------------------------\n');
fprintf('  Policy Play Evaluation (%d games):\n', N_PLAY_GAMES);
fprintf('    Win  rate  : %.2f%%\n', win_rate  * 100);
fprintf('    Draw rate  : %.2f%%\n', draw_rate * 100);
fprintf('    Loss rate  : %.2f%%\n', loss_rate * 100);
fprintf('    Avg Return : %+.4f\n', avg_return);
fprintf('  (Context: random play avg return ≈ -0.10 to -0.20\n');
fprintf('   Optimal policy avg return ≈ -0.05 to +0.05)\n');
fprintf('===============================================\n');
