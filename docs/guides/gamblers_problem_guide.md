# Gambler's Problem & Value Iteration Guide

## Overview

The Gambler's Problem (Sutton & Barto Example 4.3 & Exercise 4.9) is a classic undiscounted, episodic tabular Markov Decision Process (MDP). A gambler bets on a sequence of coin flips with heads win probability `ph`.
- **State Space**: Capital `s in {0, 1, 2, ..., 100}`.
- **Terminal States**: Boundary values `V(0) = 0` (loss) and `V(100) = 1.0` (win/goal).
- **Actions**: Stake `a in {1, 2, ..., min(s, 100 - s)}`.
- **Transitions**: Win `s + a` with probability `ph`, loss `s - a` with probability `1 - ph`.
- **Goal**: Maximize the probability of reaching capital 100.

The Ocean Library provides a generic Dynamic Programming engine `OceanStats.RL.DP.ValueIterationSolver` while formulating the Gambler's Problem setup, value sweeps, tie-breaking policy extraction, and dark-theme OceanViz figures inside `exp_gamblers_problem.m`.

---

## Usage

### 1. Value Iteration Bellman Optimality Backup

```matlab
% Value Iteration Bellman Optimality Update for state s
% Q(s, a) = ph * V(s + a) + (1 - ph) * V(s - a)
% V(s) = max_a Q(s, a)
```

### 2. Tie-Breaking Policy Extraction

Optimal policies for the Gambler's Problem are non-unique due to action value ties. Selecting the minimum stake among tied actions (`min_stake`) produces the classic Sutton & Barto Figure 4.3 pyramid optimal stake policy:

```matlab
% Extract policy with 1e-9 tolerance for argmax ties
% pi(s) = min( tied_actions )
```

---

## Coin Probability Scenarios (Exercise 4.9)

1. **Part A (`ph = 0.40`)**: Subfair coin benchmark (replicates Sutton & Barto Figure 4.3 with pyramid stake peaks at 50, 25, 75).
2. **Part B (`ph = 0.25`)**: Heavily subfair coin (steep value function, aggressive multi-level pyramid betting).
3. **Part C (`ph = 0.55`)**: Favorable coin (positive expected return leads to continuous concavity in `V(s)` and conservative minimal stake `a = 1` betting across almost all capital states).

---

## See Also

* [Gambler's Problem Recipes](../recipes/gamblers_problem_recipes.md)
* [Tabular RL Guide](tabular_rl_guide.md)
* [Architecture Reference](../../.agents/architecture.md)
