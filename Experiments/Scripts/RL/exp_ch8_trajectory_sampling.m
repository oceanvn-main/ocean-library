%% Figure 8.8 - Trajectory Sampling: Uniform vs On-Policy Update Distribution
% Reference: Sutton & Barto (2nd ed.), Section 8.6, pages 174-177.
%
% PURPOSE:
%   Investigates whether focusing dynamic programming updates along simulated 
%   trajectories (the on-policy distribution) is more computationally efficient 
%   than sweeping through all state-action pairs uniformly.
%
% TASK SPECIFICATION (Randomly generated MDPs):
%   - |S| states, 2 actions per state, fixed starting state s0 = 1.
%   - Branching factor b: each (s, a) transitions to one of b next states with
%     equal probability (1/b), drawn uniformly at random from all states.
%   - Episodic termination: every transition terminates with probability 0.1
%     (meaning continuation probability is 0.9).
%   - Expected rewards: drawn from standard normal distribution N(0, 1).
%
% UPDATE RULE (One-step expected tabular update, Eq. 8.1):
%   Q(s, a) <- Expected_Reward + 0.9 * mean_j max_a' Q(s'_j, a')
%
% EVALUATION METRIC:
%   At regular intervals (checkpoints), compute v_pi(s0), the exact value of the
%   start state under the greedy policy with respect to current Q, solved via:
%     v = (I - 0.9 * P_pi) \ r_pi

clearvars -except QuickMode; clc;

%% 1. Configuration & Experiment Setup
if ~exist('QuickMode', 'var'), QuickMode = false; end

if QuickMode
    % Fast smoke-test mode for quick validation (completes in ~1 minute)
    cfgSmall = struct('S', 1000,  'B', [1 3 10], 'Tasks', 10, 'Updates', 20000,  'EvalEvery', 500);
    cfgLarge = struct('S', 10000, 'B', 1,        'Tasks', 3,  'Updates', 200000, 'EvalEvery', 5000);
else
    % Full book-scale simulation settings (Figure 8.8)
    % Note: 50 tasks used for 10,000 states to keep execution time reasonable.
    cfgSmall = struct('S', 1000,  'B', [1 3 10], 'Tasks', 200, 'Updates', 20000,  'EvalEvery', 200);
    cfgLarge = struct('S', 10000, 'B', 1,        'Tasks', 50,  'Updates', 200000, 'EvalEvery', 2000);
end

pTerm   = 0.1;  % Probability of transition terminating the episode
epsilon = 0.1;  % Exploration rate for on-policy trajectory sampling

% Set fixed random seed for scientific reproducibility
rng(2026, 'twister');

% Ensure parallel pool is active for accelerated parfor computation
pool = gcp('nocreate');
if isempty(pool)
    fprintf('Initializing Parallel Computing Toolbox pool...\n');
    gcp();
end

%% 2. Run Simulations
fprintf('=== Running Panel 1: %d states ===\n', cfgSmall.S);
resSmall = runPanel(cfgSmall, pTerm, epsilon);

fprintf('=== Running Panel 2: %d states ===\n', cfgLarge.S);
resLarge = runPanel(cfgLarge, pTerm, epsilon);

%% 3. Plotting & OceanViz Styling
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'Figure 8.8 - Trajectory sampling', 'Position', [100 100 900 800]);
tl  = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact');

panels = {resSmall, resLarge};
cfgs   = {cfgSmall, cfgLarge};
colors = lines(numel(cfgSmall.B));

for k = 1:2
    ax = nexttile(tl);
    viz.styleAxes(ax);
    hold(ax, 'on');
    grid(ax, 'on');
    
    R = panels{k};
    C = cfgs{k};
    
    % Plot uniform vs on-policy curves for each branching factor b
    for ib = 1:numel(C.B)
        c = colors(ib, :);
        % Uniform: dashed line (exhaustive sweeps)
        plot(ax, R.X, R.Uniform(ib, :), '--', 'Color', c, 'LineWidth', 1.6, ...
            'DisplayName', sprintf('uniform, b=%d', C.B(ib)));
        % On-policy: solid line (trajectory sampling)
        plot(ax, R.X, R.OnPolicy(ib, :), '-', 'Color', c, 'LineWidth', 1.8, ...
            'DisplayName', sprintf('on-policy, b=%d', C.B(ib)));
    end
    
    xlabel(ax, 'Computation time, in expected updates');
    ylabel(ax, sprintf('Value of start state\nunder greedy policy'));
    title(ax, sprintf('%d states (%d tasks averaged)', C.S, C.Tasks));
    
    % Match legend styling to OceanViz active theme
    lgd = legend(ax, 'Location', 'southeast');
    lgd.Color     = ax.Color;
    lgd.TextColor = ax.XColor;
    lgd.EdgeColor = ax.GridColor;
end


%% =========================================================================
%% LOCAL FUNCTIONS (Top-Down Organization)
%% =========================================================================

function R = runPanel(cfg, pTerm, epsilon)
    % RUNPANEL Coordinates simulations across branching factors and random tasks.
    %
    % Inputs:
    %   cfg     - Struct with fields: S, B, Tasks, Updates, EvalEvery
    %   pTerm   - Probability of transition to terminal state (0.1)
    %   epsilon - Epsilon-greedy parameter for on-policy sampling (0.1)
    %
    % Output:
    %   R - Struct containing:
    %         X        : Checkpoint update counts (1 x numCheckpoints)
    %         Uniform  : Average start-state values for uniform sweeps (numB x numCheckpoints)
    %         OnPolicy : Average start-state values for on-policy rollouts (numB x numCheckpoints)

    % 1. Create evaluation checkpoint grid (e.g., [0, 500, 1000, ..., 20000])
    X = 0:cfg.EvalEvery:cfg.Updates;
    R.X = X;
    
    % 2. Preallocate accumulators for learning curves
    R.Uniform  = zeros(numel(cfg.B), numel(X));
    R.OnPolicy = zeros(numel(cfg.B), numel(X));
    
    % 3. Iterate through each branching factor b
    for ib = 1:numel(cfg.B)
        b = cfg.B(ib);
        
        % Preallocate 1D reduction accumulators for this branching factor
        sumUniform  = zeros(1, numel(X));
        sumOnPolicy = zeros(1, numel(X));
        
        % 4. Monte Carlo averaging over independent random MDP tasks in PARALLEL
        parfor t = 1:cfg.Tasks
            % Construct a brand new random environment instance
            task = makeTask(cfg.S, b);
            
            % Run both algorithms on the EXACT SAME task for paired comparison
            curveU = planUniform(task, cfg, pTerm);
            curveO = planOnPolicy(task, cfg, pTerm, epsilon);
            
            % Reduction addition (accumulates across all parallel CPU workers)
            sumUniform  = sumUniform  + curveU;
            sumOnPolicy = sumOnPolicy + curveO;
        end
        
        % Store sample mean for this branching factor
        R.Uniform(ib, :)  = sumUniform  / cfg.Tasks;
        R.OnPolicy(ib, :) = sumOnPolicy / cfg.Tasks;
        
        fprintf('  Finished |S|=%d, b=%d across %d tasks.\n', cfg.S, b, cfg.Tasks);
    end
end


function curve = planUniform(task, cfg, pTerm)
    % PLANUNIFORM Performs exhaustive dynamic programming updates cycling
    % through all state-action pairs in sequential order.
    
    % Initialize Q-table to zeros: S states x 2 actions
    Q = zeros(task.S, 2);
    nPairs = 2 * task.S;  % Total (s, a) pairs in the MDP
    
    % Preallocate evaluation curve array
    numCheckpoints = cfg.Updates / cfg.EvalEvery + 1;
    curve = zeros(1, numCheckpoints);
    
    % Checkpoint 0: evaluate greedy policy before any planning updates occur
    curve(1) = evaluateGreedy(Q, task, pTerm);
    k = 1;
    
    % Main planning loop: perform expected updates in sequential order
    for u = 1:cfg.Updates
        % Cycle through all pairs 1, 2, ..., nPairs, 1, 2, ...
        p = mod(u - 1, nPairs) + 1;
        
        % Perform one-step expected update in place
        Q(p) = expectedUpdate(Q, task, p, pTerm);
        
        % Record evaluation checkpoint
        if mod(u, cfg.EvalEvery) == 0
            k = k + 1;
            curve(k) = evaluateGreedy(Q, task, pTerm);
        end
    end
end


function curve = planOnPolicy(task, cfg, pTerm, epsilon)
    % PLANONPOLICY Performs trajectory sampling by simulating episodes under
    % an epsilon-greedy policy starting from state s0 = 1.
    
    S = task.S;
    Q = zeros(S, 2);
    
    numCheckpoints = cfg.Updates / cfg.EvalEvery + 1;
    curve = zeros(1, numCheckpoints);
    
    % Checkpoint 0: evaluate greedy policy before updates
    curve(1) = evaluateGreedy(Q, task, pTerm);
    k = 1;
    
    % All simulated trajectories start at state 1
    s = 1;
    
    for u = 1:cfg.Updates
        % Epsilon-greedy action selection from current state s
        if rand < epsilon
            a = randi(2);                           % Explore: random action 1 or 2
        else
            a = 1 + (Q(s, 2) > Q(s, 1));            % Exploit: pick greedy action
        end
        
        % Compute pair linear index p for (s, a)
        p = (a - 1) * S + s;
        
        % Update the visited state-action pair using expected Bellman equation
        Q(p) = expectedUpdate(Q, task, p, pTerm);
        
        % Simulate environment transition
        if rand < pTerm
            s = 1;                                  % Episode terminated -> restart at s0
        else
            % Transition to one of the b next states with equal probability
            s = task.Next(p, randi(task.b));
        end
        
        % Record evaluation checkpoint
        if mod(u, cfg.EvalEvery) == 0
            k = k + 1;
            curve(k) = evaluateGreedy(Q, task, pTerm);
        end
    end
end


function q = expectedUpdate(Q, task, p, pTerm)
    % EXPECTEDUPDATE One-step tabular expected update (Equation 8.1).
    % Formula:
    %   Q(s, a) <- R_bar(s, a) + (1 - pTerm) * (1/b) * sum_{s'} max_a' Q(s', a')
    
    % Look up the b possible next states for this (s, a) pair
    ns = task.Next(p, :);
    
    % Compute the expected value of next state: mean over branches of max_a' Q(s', a')
    nextStateValue = mean(max(Q(ns, 1), Q(ns, 2)));
    
    % Bellman expectation backup (continuation probability is 1 - pTerm = 0.9)
    q = task.Rbar(p) + (1 - pTerm) * nextStateValue;
end


function v0 = evaluateGreedy(Q, task, pTerm)
    % EVALUATEGREEDY Computes exact ground-truth value of start state v(s0)
    % under the greedy policy with respect to current Q.
    % Solves the linear system: (I - 0.9 * P_pi) * v = r_pi
    
    S = task.S;
    b = task.b;
    
    % 1. Extract the greedy policy for all states: a(s) in {1, 2}
    % Ties default to action 1
    a = 1 + (Q(:, 2) > Q(:, 1));
    
    % 2. Linear indices for (s, a(s)) across all S states
    p = (a - 1) * S + (1:S)';
    
    % 3. Assemble sparse transition matrix P_pi of size S x S
    % Each state s has transition probability 1/b to each of its b next states
    rowIndices = repmat((1:S)', 1, b); % S x b
    colIndices = task.Next(p, :); % S x b
    transProb  = 1 / b;
    P = sparse(rowIndices, colIndices, transProb, S, S);
    
    % 4. Solve the linear system for the exact state values v:
    % (I - (1 - pTerm) * P) * v = r_bar
    continuationProb = 1 - pTerm;
    v = (speye(S) - continuationProb * P) \ task.Rbar(p);
    
    % 5. Return the value of the start state s0 = 1
    v0 = v(1);
end


function task = makeTask(S, b)
    % MAKETASK Generates a random MDP according to Sutton & Barto Section 8.6.
    %
    % Inputs:
    %   S - Number of states (e.g., 1000 or 10000)
    %   b - Branching factor (number of possible next states per action)
    %
    % Output:
    %   task - Struct with fields:
    %            S    : Number of states
    %            b    : Branching factor
    %            Next : (2*S x b) array of destination states for each (s,a)
    %            Rbar : (2*S x 1) vector of expected rewards for each (s,a)
    
    task.S = S;
    task.b = b;
    
    % For each of the 2*S state-action pairs, randomly select b next states
    % uniformly with replacement from {1, 2, ..., S}
    task.Next = randi(S, 2 * S, b);
    
    % Draw transition rewards from standard Gaussian N(0, 1) and average across b branches
    task.Rbar = mean(randn(2 * S, b), 2);
end
