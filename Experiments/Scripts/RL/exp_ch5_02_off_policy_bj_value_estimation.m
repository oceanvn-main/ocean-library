% EXP_CH5_02_OFF_POLICY_BJ_VALUE_ESTIMATION
%   Off-policy Estimation of a Blackjack State Value (Sutton & Barto Example 5.4).
%   Compares Ordinary Importance Sampling (OIS) vs Weighted Importance Sampling (WIS)
%   for estimating V_pi(s_0) on state s_0 = [13, 2, 1] over 100 runs of 10,000 episodes.
%
%   ENVIRONMENT:     BlackjackEnv — 200 states, infinite deck
%   TARGET STATE:    s_0 = [player_sum=13, dealer_showing=2, usable_ace=1]
%   TARGET VALUE:    v_true = -0.27726  (exact analytical/high-sample reference value)
%   TARGET POLICY:   Stick ONLY if player_sum >= 20, else HIT (Example 5.1 policy)
%   BEHAVIOR POLICY: Uniform random: HIT or STICK with equal probability 0.5
%   ALGORITHMS:
%     1. Ordinary Importance Sampling (OIS) — unbiased, high variance
%     2. Weighted Importance Sampling (WIS) — biased at small N, much lower variance
%   HYPERPARAMETERS:
%     N_RUNS       = 100 independent trials
%     N_EPISODES   = 10 000 episodes per trial
%     Gamma        = 1.0 (episodic, no discounting)
%   ASSUMPTIONS:
%     - All episodes forced to begin at initial state s_0 = [13, 2, 1]
%     - Behavior policy b(a|s) = 0.5 for both actions
%     - Target policy pi(stick|s) = 1 if sum >= 20 else 0; pi(hit|s) = 1 if sum < 20 else 0
%     - Initial state value estimates start at V(s_0) = 0
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   OUTPUTS:
%     - Log-scale MSE comparison plot (replicating S&B Figure 5.3)
%     - Printed summary table of final MSE and variance at episode 10, 100, 1000, 10000
%
%   Reference: Sutton & Barto (2018) §5.5, Example 5.4, Figure 5.3
%
%   Package context: OceanStats.RL.MonteCarlo — Ch.5 Monte Carlo Methods

clc; clear; close all;
% Ensure Ocean Library root is on MATLAB search path
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);

rng(42, 'twister');   % Reproducibility

% =========================================================================
% === 1. HYPERPARAMETERS ==================================================
% =========================================================================
N_RUNS     = 100;      % Number of independent runs (100 in S&B Fig 5.3)
N_EPISODES = 10000;    % Episodes per run (10,000 in S&B Fig 5.3)
GAMMA      = 1.0;      % Discount factor (episodic)
V_TRUE     = -0.27726; % Ground-truth target value from S&B Example 5.4

TARGET_STATE = [13, 2, 1]; % [player_sum=13, dealer_showing=2, usable_ace=1]

% =========================================================================
% === 2. ENVIRONMENT & POLICY SETUP =======================================
% =========================================================================
fprintf('[Setup] Constructing BlackjackEnv...\n');
env = OceanStats.RL.Env.Classic.BlackjackEnv();
s0_idx = env.state_to_index(TARGET_STATE);

fprintf('  Target State : [player_sum=%d, dealer=%d, usable_ace=%d] (State index: %d)\n', ...
    TARGET_STATE(1), TARGET_STATE(2), TARGET_STATE(3), s0_idx);
fprintf('  True Value   : %+.5f\n', V_TRUE);
fprintf('  Benchmark    : %d runs x %d episodes per method\n', N_RUNS, N_EPISODES);

% Target Policy pi: @(s_idx, a_idx) -> probability
% Target policy sticks only on 20 or 21 (action_idx 1 = stick, action_idx 2 = hit)
% Note: State is decoded to player sum to evaluate pi(a|s)
all_states = env.get_all_states();
target_policy = @(s_idx, a_idx) target_policy_prob_(all_states, s_idx, a_idx);

% Behavior Policy b: @(s_idx, a_idx) -> 0.5 (uniform random)
behavior_policy = @(~, ~) 0.5;

% =========================================================================
% === 3. TIGHT EXECUTION LOOP FOR 100 RUNS (OIS & WIS) =====================
% =========================================================================
% For high speed across 2,000,000 total episodes, we track V(s_0) trajectory
% directly after each episode per run.

fprintf('\n[Phase 1] Executing %d runs of Ordinary Importance Sampling (OIS)...\n', N_RUNS);

v_history_ord = zeros(N_RUNS, N_EPISODES); % (100 x 10000)

for run_idx = 1:N_RUNS
    solver_ord = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
        'Algorithm',       'OffPolicyPrediction', ...
        'ISType',          'Ordinary',            ...
        'NumEpisodes',     N_EPISODES,            ...
        'Gamma',           GAMMA,                 ...
        'TargetPolicy',    target_policy,         ...
        'BehaviorPolicy',  behavior_policy,       ...
        'ForceStartState', TARGET_STATE,          ...
        'CheckInterval',   1,                     ... Checkpoint every episode
        'Verbose',         false                  ...
    );
    solver_ord.run();

    % Extract V(s_0) snapshot at every episode
    v_snaps = [solver_ord.History.VSnapshot{:}];
    v_history_ord(run_idx, :) = v_snaps(s0_idx, :);

    if mod(run_idx, 20) == 0
        fprintf('  OIS Run %3d / %d completed\n', run_idx, N_RUNS);
    end
end

fprintf('\n[Phase 2] Executing %d runs of Weighted Importance Sampling (WIS)...\n', N_RUNS);

v_history_wei = zeros(N_RUNS, N_EPISODES); % (100 x 10000)

for run_idx = 1:N_RUNS
    solver_wei = OceanStats.RL.MonteCarlo.MonteCarloSolver(env, ...
        'Algorithm',       'OffPolicyPrediction', ...
        'ISType',          'Weighted',            ...
        'NumEpisodes',     N_EPISODES,            ...
        'Gamma',           GAMMA,                 ...
        'TargetPolicy',    target_policy,         ...
        'BehaviorPolicy',  behavior_policy,       ...
        'ForceStartState', TARGET_STATE,          ...
        'CheckInterval',   1,                     ... Checkpoint every episode
        'Verbose',         false                  ...
    );
    solver_wei.run();

    v_snaps = [solver_wei.History.VSnapshot{:}];
    v_history_wei(run_idx, :) = v_snaps(s0_idx, :);

    if mod(run_idx, 20) == 0
        fprintf('  WIS Run %3d / %d completed\n', run_idx, N_RUNS);
    end
end

% =========================================================================
% === 4. COMPUTE MEAN SQUARED ERROR (MSE) ACROSS RUNS =====================
% =========================================================================
% MSE(t) = (1 / N_RUNS) * sum_{i=1}^{N_RUNS} (V_i,t(s_0) - V_TRUE)^2

mse_ord = mean((v_history_ord - V_TRUE).^2, 1); % (1 x 10000)
mse_wei = mean((v_history_wei - V_TRUE).^2, 1); % (1 x 10000)
episodes_axis = 1:N_EPISODES;

% =========================================================================
% === 5. VISUALIZATION (OceanViz.Core + MonteCarloRLStrategy) =============
% =========================================================================
fprintf('\n[Viz] Rendering Figure 5.3 replication with OceanViz...\n');

viz_data = struct( ...
    'Episodes',     episodes_axis, ...
    'MSE_Ordinary', mse_ord,       ...
    'MSE_Weighted', mse_wei,       ...
    'TrueValue',    V_TRUE         ...
);

viz  = OceanViz.Core('Theme', 'Dark');
fig  = figure('Name', 'Off-Policy Value Estimation — S&B Figure 5.3', ...
    'Color', viz.Colors.Background, 'Position', [100, 100, 950, 520]);
ax   = axes('Parent', fig);

strat = OceanViz.Strategies.MonteCarloRLStrategy();
strat.draw(ax, viz_data, struct('Mode', 'ISComparison'), viz.Colors);

% =========================================================================
% === 6. SUMMARY PRINTOUT =================================================
% =========================================================================
check_points = [1, 10, 100, 1000, 10000];

fprintf('\n===========================================================================\n');
fprintf('  EXPERIMENT SUMMARY: exp_ch5_02_off_policy_bj_value_estimation (Fig 5.3)\n');
fprintf('===========================================================================\n');
fprintf('  Target State       : [player_sum=13, dealer=2, usable_ace=1]\n');
fprintf('  Ground Truth V(s0) : %+.5f\n', V_TRUE);
fprintf('  Independent Runs   : %d\n', N_RUNS);
fprintf('  Episodes per Run   : %d\n', N_EPISODES);
fprintf('---------------------------------------------------------------------------\n');
fprintf('  %10s | %18s | %18s | %10s\n', 'Episode', 'Ordinary IS MSE', 'Weighted IS MSE', 'Ratio (O/W)');
fprintf('---------------------------------------------------------------------------\n');

for k = 1:length(check_points)
    ep_k = check_points(k);
    m_ord = mse_ord(ep_k);
    m_wei = mse_wei(ep_k);
    ratio = m_ord / max(m_wei, 1e-12);
    fprintf('  %10d | %18.4f | %18.4f | %10.2fx\n', ep_k, m_ord, m_wei, ratio);
end

fprintf('---------------------------------------------------------------------------\n');
fprintf('  Key Observation: Weighted IS has dramatically lower error at early\n');
fprintf('  episodes (e.g. at ep 10, WIS error is %.1fx lower than Ordinary IS).\n', ...
    mse_ord(10) / max(mse_wei(10), 1e-12));
fprintf('===========================================================================\n');

% =========================================================================
% === LOCAL HELPER FUNCTIONS ==============================================
% =========================================================================
function prob = target_policy_prob_(all_states, s_idx, a_idx)
    % Target policy: stick (a_idx=1) if player_sum >= 20, else hit (a_idx=2)
    if s_idx == 0
        prob = 0.5;
        return;
    end
    ps = all_states(s_idx, 1);
    if ps >= 20
        target_a_idx = 1; % stick
    else
        target_a_idx = 2; % hit
    end
    prob = double(a_idx == target_a_idx);
end
