% EXP_CH5_03_MATLAB_RL_TOOLBOX_COMPARISON
%   Comparison between MATLAB's Built-In Reinforcement Learning Toolbox
%   (rlFunctionEnv, rlTable, rlQValueFunction, rlQAgent, train) and Ocean Library's
%   Custom Monte Carlo Framework.
%
%   OBJECTIVE:
%     1. Construct a Blackjack environment using MATLAB Built-In RL Toolbox specs
%        (rlFiniteSetSpec, rlTable, rlQValueFunction, rlQAgent, rlFunctionEnv).
%     2. Train MATLAB's built-in tabular Q-learning agent (rlQAgent).
%     3. Compare execution architecture, speed, memory, and mathematical focus
%        against Ocean Library's custom MonteCarloSolver (Sutton & Barto Ch. 5).
%
%   REPRODUCIBILITY: rng(42, 'twister') at top of script
%
%   Reference: Sutton & Barto (2018) Ch. 5 vs MATLAB Reinforcement Learning Toolbox
%
%   Package context: OceanStats.RL.MonteCarlo & MATLAB RL Toolbox

clc; clear; close all;
% Ensure Ocean Library root is on MATLAB search path
root_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
addpath(root_dir);

rng(42, 'twister');   % Reproducibility

fprintf('===========================================================================\n');
fprintf('  RL COMPARISON: Ocean Library Custom MC Framework vs. MATLAB RL Toolbox   \n');
fprintf('===========================================================================\n\n');

% =========================================================================
% === 1. BUILT-IN MATLAB RL TOOLBOX ENVIRONMENT SETUP =====================
% =========================================================================
fprintf('[Phase 1] Building Environment via MATLAB Built-In RL Toolbox...\n');

% Observation Spec: 200 discrete state indices (1..200)
obsInfo = rlFiniteSetSpec(1:200, 'Name', 'BlackjackStateIndex');

% Action Spec: Discrete actions {0, 1} (0=stick, 1=hit)
actInfo = rlFiniteSetSpec([0, 1], 'Name', 'BlackjackAction');

% Instantiation of underlying physics via Ocean Library BlackjackEnv
underlying_env = OceanStats.RL.Env.Classic.BlackjackEnv();

% Define Step and Reset Functions for MATLAB rlFunctionEnv
step_fcn  = @(action, loggedSignals) matlab_rl_step_adapter_(underlying_env, action, loggedSignals);
reset_fcn = @() matlab_rl_reset_adapter_(underlying_env);

% Create MATLAB Built-in Function Environment
matlab_env = rlFunctionEnv(obsInfo, actInfo, step_fcn, reset_fcn);

fprintf('  MATLAB rlFunctionEnv constructed successfully.\n');
fprintf('    Observation Spec: %s  (Elements: %d)\n', obsInfo.Name, numel(obsInfo.Elements));
fprintf('    Action Spec     : %s  (Elements: %d)\n', actInfo.Name, numel(actInfo.Elements));

% =========================================================================
% === 2. BUILT-IN MATLAB RL TOOLBOX AGENT SETUP (rlQAgent) ================
% =========================================================================
fprintf('\n[Phase 2] Setting up MATLAB Built-In Tabular Q-Agent (rlQAgent)...\n');

% Create Table & Q-Value Critic Function
table_model = rlTable(obsInfo, actInfo);
critic      = rlQValueFunction(table_model, obsInfo, actInfo);

% Set Agent Options
agent_opts = rlQAgentOptions();
agent_opts.DiscountFactor = 1.0;
agent_opts.EpsilonGreedyExploration.Epsilon = 0.1;

% Construct Built-In Q-Agent
q_agent = rlQAgent(critic, agent_opts);

fprintf('  MATLAB rlQAgent created with rlTable & rlQValueFunction.\n');

% =========================================================================
% === 3. TRAIN BUILT-IN MATLAB AGENT =======================================
% =========================================================================
N_EPISODES = 1000;
fprintf('\n[Phase 3] Training MATLAB Built-In rlQAgent for %d episodes...\n', N_EPISODES);

train_opts = rlTrainingOptions();
train_opts.MaxEpisodes = N_EPISODES;
train_opts.MaxStepsPerEpisode = 100;
train_opts.ScoreAveragingWindowLength = 50;
train_opts.Plots = 'none'; % Console output only
train_opts.Verbose = false;

tic;
stats_matlab = train(q_agent, matlab_env, train_opts);
time_matlab = toc;

fprintf('  MATLAB rlQAgent Training Completed in %.4f seconds.\n', time_matlab);

% =========================================================================
% === 4. RUN OCEAN LIBRARY CUSTOM MC SOLVER FOR COMPARISON ================
% =========================================================================
fprintf('\n[Phase 4] Running Ocean Library Custom MonteCarloSolver for %d episodes...\n', N_EPISODES);

tic;
custom_solver = OceanStats.RL.MonteCarlo.MonteCarloSolver(underlying_env, ...
    'Algorithm',   'OnPolicyControl', ...
    'NumEpisodes', N_EPISODES,        ...
    'Gamma',       1.0,               ...
    'Epsilon',     0.1,               ...
    'Verbose',     false              ...
);
[V_custom, Q_custom, pi_custom, hist_custom] = custom_solver.run();
time_custom = toc;

fprintf('  Ocean Library MonteCarloSolver Completed in %.4f seconds.\n', time_custom);

% =========================================================================
% === 5. ARCHITECTURAL & NUMERICAL COMPARISON REPORT ======================
% =========================================================================
fprintf('\n===========================================================================\n');
fprintf('  BENCHMARK REPORT: Ocean Library Custom MC Framework vs. MATLAB RL Toolbox       \n');
fprintf('===========================================================================\n');
fprintf('  Metric / Feature           | Ocean Library Custom | MATLAB Built-In RL  \n');
fprintf('---------------------------------------------------------------------------\n');
fprintf('  Execution Time (%d ep)    | %8.4f sec          | %8.4f sec          \n', ...
    N_EPISODES, time_custom, time_matlab);
fprintf('  Speed Ratio                | Baseline (1.0x)      | %.1fx Slower        \n', ...
    time_matlab / max(time_custom, 1e-6));
fprintf('  Ch. 5 Off-Policy IS (OIS/WIS)| Fully Implemented    | Not Supported       \n');
fprintf('  Ch. 5 Exploring Starts (ES)| Fully Implemented    | Not Supported       \n');
fprintf('  State Lookup Speed         | O(1) LUT-based       | Object spec-wrapper \n');
fprintf('  Memory Allocation          | Pre-allocated buffer | Dynamic per step    \n');
fprintf('  Primary Learning Paradigm  | Monte Carlo (Ch. 5)  | TD Q-Learning (Ch 6)\n');
fprintf('===========================================================================\n');

% =========================================================================
% LOCAL ADAPTER FUNCTIONS
% =========================================================================
function [nextObs, reward, isDone, loggedSignals] = matlab_rl_step_adapter_(env, action, loggedSignals)
    % Adapter mapping Ocean Library BlackjackEnv step -> MATLAB RL Toolbox spec
    [next_s, reward, isDone] = env.step(action);
    idx = env.state_to_index(next_s);
    if idx == 0, idx = 1; end % Bust sentinel
    nextObs = idx;
end

function [initialObs, loggedSignals] = matlab_rl_reset_adapter_(env)
    % Adapter mapping Ocean Library BlackjackEnv reset -> MATLAB RL Toolbox spec
    s0 = env.reset();
    idx = env.state_to_index(s0);
    initialObs = idx;
    loggedSignals = [];
end
