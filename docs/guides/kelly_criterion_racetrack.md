# Kelly Criterion & Horse Race Simulation Guide

## Overview

The duality between gambling/investment and data compression is one of the most profound insights in Information Theory (first outlined by J. L. Kelly Jr. in 1956). This guide covers:

- **Log-Optimal Portfolios & Horse Races** — maximizing exponential growth rate.
- **The Kelly Criterion** — allocating capital proportionally to edge/odds.
- **Visualizing Races** via `OceanViz.Strategies.RaceTrackRenderer`.

### The Kelly Formula (Unicode)

For a horse race with mutual odds where horse `i` has winning probability `p_i` and payodds `b_i` (net odds `o_i = b_i - 1`):

```
f_i* = p_i - q_i / o_i = p_i - (1 - p_i) / (b_i - 1)
```

The growth rate `W(f)` of wealth is maximized when:
- We invest `f_i* = p_i` (when track take is 0, i.e., `sum(1/b_i) = 1`).
- We only bet when `p_i > 1/b_i` (positive edge).

---

## Package Architecture

The Kelly Criterion and horse race simulations are implemented as decoupled, high-fidelity components:

- **Simulation & Math Engine**: Model the stochastic state of wealth, probabilities, and physical positions of horses.
- **Visualization Strategy (`RaceTrackRenderer`)**: Interactive horizontal track displaying horse positions, leading horse spotlights, spark particle effects, and flashing winner banners.

---

## Usage

### Instantiating the Racetrack

To initialize the cyberpunk-themed horse race visualizer:

```matlab
% 1. Create a figure and axis
fig = figure('Color', [0.08, 0.08, 0.10]);
ax  = axes(fig);

% 2. Setup the renderer for 5 horses
renderer = OceanViz.Strategies.RaceTrackRenderer(ax, 5);
renderer.initializeTrack();
```

### Animating a Race

To drive positions dynamically from your simulation:

```matlab
% Run a mock loop showing the positions shifting
positions = zeros(1, 5);
while max(positions) < 1.0
    % Update positions with a small forward step
    positions = positions + rand(1, 5) * 0.05;
    positions = min(positions, 1.0);
    
    % Update the screen smoothly
    renderer.updatePositions(positions);
    pause(0.05);
end

% Flash the winner
[~, winner] = max(positions);
renderer.triggerWinnerFlash(winner);
```

---

## See Also

- `docs/recipes/kelly_criterion_recipes.md` — copy-paste simulation templates
- `docs/guides/source_coding.md` — the duality of entropy and log-optimal growth
- Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*.
