# Online Accumulators Guide (`+OceanEngine/+Accumulators`)

## Overview

The `+OceanEngine/+Accumulators` package provides a domain-agnostic, memory-efficient framework for online (streaming) statistic accumulation. It unifies incremental sample averaging, weighted importance sampling (Sutton & Barto Eq. 5.8), exponential moving averages, and Welford running variance into a shared infrastructure package across Reinforcement Learning, Signal Processing, and Data Analytics.

All accumulators implement the unified `IAccumulator` interface and inherit from `AbstractAccumulator`:

```matlab
classdef (Abstract) IAccumulator < handle
    methods (Abstract)
        update(obj, sample, weight, idx)
        val  = getValue(obj)
        w    = getWeight(obj)
        reset(obj)
        meta = getMetadata(obj)      % Pillar 1: Symbolic LaTeX & Metadata
        spec = getDiagramSpec(obj)   % Pillar 3: Canvas Port Spec & Symbol
    end
end
```

---

## Abstract Base Class (`AbstractAccumulator`)

`OceanEngine.Accumulators.AbstractAccumulator` provides shared infrastructure for matrix table pre-allocation, subscript index parsing (`0D / 1D / 2D`), and table getters/setters using the **Template Method Design Pattern** (Ocean Library Architecture Principle 6). Subclasses implement lightweight step update hooks (`updateFullArray`, `updateIndexed`, `updateSubscript2D`).

---

## Available Accumulator Strategies

### 1. Weighted Mean & Importance Sampling (`WeightedMeanAccumulator`)
Implements incremental weighted importance sampling (Sutton & Barto Section 5.6):

$$C_n = C_{n-1} + W_n$$
$$V_n = V_{n-1} + \frac{W_n}{C_n} (X_n - V_{n-1})$$

- **Usage**: Reinforcement Learning off-policy / on-policy Monte Carlo value and action-value table estimation.
- **Indexing**: Supports 1D vector and 2D matrix (state-action pair `(s, a)`) indexed updates.

### 2. Exponential Moving Average (`ExponentialMovingAverage`)
Implements constant step-size tracking (Sutton & Barto Section 2.5):

$$V_n = V_{n-1} + \alpha (X_n - V_{n-1})$$

- **Usage**: Non-stationary tracking in Q-learning, SARSA, and streaming signal smoothing.

### 3. Welford Online Variance (`WelfordVarianceAccumulator`)
Implements Welford's (1962) single-pass online mean and variance algorithm:

$$M_{2, n} = M_{2, n-1} + W_n (X_n - \mu_{n-1})(X_n - \mu_n)$$

- **Usage**: Streaming feature normalization and online standard deviation tracking in $O(1)$ memory.

---

## Usage Examples

### Instantiating & Introspecting via Factory

```matlab
% Create a 2D matrix accumulator for RL Q-table [200 states x 2 actions]
q_acc = OceanEngine.Accumulators.AccumulatorFactory.create('Weighted', [200 2]);

% Perform indexed update for state 42, action 2
q_acc.update(10.0, 1.5, 42, 2);

% Retrieve current Q table and weight matrix
Q_table = q_acc.getValue();
C_table = q_acc.getWeight();

% Canvas Palette Introspection (Pillar 3 Diagram Spec)
proto = OceanEngine.Accumulators.AccumulatorFactory.getPrototypeSpec('EMA', [10 1]);
```

---

## See Also
- `OceanEngine.Accumulators.IAccumulator`
- `OceanEngine.Accumulators.AbstractAccumulator`
- `OceanEngine.Accumulators.WeightedMeanAccumulator`
- `OceanEngine.Accumulators.ExponentialMovingAverage`
- `OceanEngine.Accumulators.WelfordVarianceAccumulator`
- `OceanEngine.Accumulators.AccumulatorFactory`
- `OceanStats.RL.MonteCarlo.MonteCarloSolver`
