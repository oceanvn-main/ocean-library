# Kelly Criterion & Horse Race Simulation Recipes

Copy-paste templates for simulating horse races, computing Kelly optimal allocations, and rendering animations on the racetrack.

---

## 1. Kelly Optimal Betting Allocations (Zero Track Take)

When the track take is zero (`sum(1./payodds) == 1`), the optimal allocation fraction `f_i*` for horse `i` is exactly its winning probability `p_i`.

```matlab
probabilities = [0.40, 0.30, 0.20, 0.10];
payodds       = [2.50, 3.33, 5.00, 10.0];  % fair mutual odds

% Verify track take is zero
assert(abs(sum(1./payodds) - 1.0) < 1e-9, 'Odds must be fair');

% Kelly allocation is exactly the probability
kelly_fractions = probabilities;
disp('Kelly Fractions (Zero Take):');
disp(kelly_fractions);
```

---

## 2. Kelly Optimal Betting Allocations (Positive Track Take)

When track take is positive (e.g. `sum(1./payodds) > 1`), we solve for the Lagrange multiplier or use the thresholding algorithm to only place bets on horses with a positive edge (`p_i * payodds_i > 1`).

```matlab
probabilities = [0.55, 0.30, 0.15];
payodds       = [1.50, 2.50, 4.00];  % positive take sum(1./payodds) = 1.3167

% Find horses with positive edge
edges = probabilities .* payodds;
disp('Edges:');
disp(edges);

% Allocate using thresholded Kelly: f_i* = p_i - (1 - sum_{j in S} p_j) / (1 - sum_{j in S} 1/b_j) / b_i
% For a simple single horse bet:
kelly_single = probabilities(1) - (1 - probabilities(1)) / (payodds(1) - 1);
fprintf('Kelly single allocation for horse 1: %.4f\n', kelly_single);
```

---

## 3. Running a Smooth Animated Horse Race

Use the `RaceTrackRenderer` to run a high-fidelity visual simulation of a horse race.

```matlab
% Setup the visual environment
fig = figure('Color', [0.08, 0.08, 0.10]);
ax  = axes(fig);

numHorses = 5;
renderer = OceanViz.Strategies.RaceTrackRenderer(ax, numHorses);
renderer.initializeTrack();

% Simulate movement
positions = zeros(1, numHorses);
probabilities = [0.35, 0.25, 0.20, 0.12, 0.08];

while max(positions) < 1.0
    % Progress depends on speed (drawn from random walk / odds)
    speed = rand(1, numHorses) .* probabilities * 0.15 + rand(1, numHorses) * 0.02;
    positions = positions + speed;
    positions = min(positions, 1.0);
    
    % Render positions smoothly
    renderer.updatePositions(positions);
    pause(0.04);  % ~25 FPS
end

% Trigger winner spotlight and lane flash
[~, winner] = max(positions);
renderer.triggerWinnerFlash(winner);
```

---

## 4. Kelly Mathematical Optimization

Using `KellyOptimizer` to compute entropy, KL divergence (betting loss), optimal betting fractions, and theoretical growth rates:

```matlab
% Set up horse probabilities and payodds
p = [0.55, 0.30, 0.15];
odds = [1.60, 2.80, 5.00];

% 1. Compute Shannon entropy of the race outcome
H = OceanMath.Information.Processes.KellyOptimizer.entropy(p);
fprintf('Race Entropy: %.4f bits\n', H);

% 2. Solve for optimal fractions (handles unfair odds/track-take)
b_opt = OceanMath.Information.Processes.KellyOptimizer.computeOptimalFractions(p, odds);
disp('Optimal betting fractions (Kelly):');
disp(b_opt);

% 3. Calculate expected analytical log-growth rate
W = OceanMath.Information.Processes.KellyOptimizer.computeExpectedGrowth(p, b_opt, odds);
fprintf('Expected Growth Rate: %.4f bits per race\n', W);

% 4. Compute betting loss (KL divergence) when using subjective probabilities
b_sub = [0.40, 0.40, 0.20]; % non-optimal subjective belief
D = OceanMath.Information.Processes.KellyOptimizer.divergence(p, b_sub);
fprintf('Betting Loss (KL Divergence): %.4f bits\n', D);
```

---

## 5. Multi-Step Stochastic Gambling Simulation

Using `GamblingSimulator` and `RaceSimulator` to run an empirical betting campaign over 1000 independent races and compare results with Kelly theory:

```matlab
% Setup race parameters
p = [0.50, 0.30, 0.20];
odds = [2.00, 3.00, 5.00]; % fair mutual odds

% Calculate optimal fractions
b_opt = OceanMath.Information.Processes.KellyOptimizer.computeOptimalFractions(p, odds);

% Instantiate GamblingSimulator
sim = OceanMath.Information.Processes.GamblingSimulator(p, odds, b_opt);

% Run a simulation campaign of 1000 trials starting with $100
startingWealth = 100.0;
numRaces = 1000;
res = sim.runSimulation(startingWealth, numRaces);

% Compare empirical growth vs analytical theory
fprintf('Empirical Growth Rate:   %.4f bits/race\n', res.EmpiricalGrowth);
fprintf('Theoretical Growth Rate: %.4f bits/race\n', res.TheoreticalGrowth);

% Plot the wealth trajectory over time (log-scale)
figure('Color', [0.08, 0.08, 0.10]);
semilogy(res.WealthHistory, 'LineWidth', 2, 'Color', [0.0, 0.8, 1.0]);
grid on;
ax = gca;
ax.Color = [0.08, 0.08, 0.10];
ax.XColor = [0.8, 0.8, 0.8];
ax.YColor = [0.8, 0.8, 0.8];
title('Empirical Capital Accumulation (Kelly Strategy)', 'Color', 'w');
xlabel('Race Step', 'Color', 'w');
ylabel('Wealth ($)', 'Color', 'w');
```

---

## See Also

- `docs/guides/kelly_criterion_racetrack.md` — full theoretical concepts
- `OceanMath.Information.Processes.KellyOptimizer` — mathematical Kelly solver
- `OceanMath.Information.Processes.GamblingSimulator` — stochastic gambling simulator
- `OceanViz.Strategies.RaceTrackRenderer` — racetrack visualization source code
