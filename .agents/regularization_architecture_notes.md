# Future-Proofing for Advanced Regularization & Solvers

Based on the readings (ESL §1.1.2, 1.1.3, 1.2.1), you will eventually need to implement:
- **Primal unconstrained solvers** (SGD, L-BFGS, CD)
- **Composite objectives** ($f(w) + \gamma r(w)$)
- **Non-smooth regularizers** (like $L_1$ Lasso)
- **Proximity Operators** (Forward-Backward Splitting / Iterative Shrinking)

To ensure your current architecture doesn't break when you learn and implement these, we need to introduce a few targeted abstractions now.

## 1. Abstract the Regularizer (`IRegularizer`)

Just like `ILoss` abstracts the error function $f(w)$, you need an `IRegularizer` interface to abstract $r(w)$. 
This allows you to mix and match Losses and Regularizers independently, creating a composite objective without exploding the number of classes (e.g., avoiding `L1MSEModel`, `L2MSEModel`, `L1HingeModel`).

**Proposed Interface: `+OceanStats/+Regularization/IRegularizer.m`**
```matlab
classdef IRegularizer < handle
    methods (Abstract)
        % Evaluate the penalty: r(w)
        val = evaluate(obj, w)
        
        % For smooth regularizers (like L2): returns gradient
        % For non-smooth (like L1): returns subgradient or empty if proximity is preferred
        grad = gradient(obj, w)
        
        % The Proximity Operator: argmin_x { 0.5 * ||x - w||_2^2 + lambda * r(x) }
        % (Crucial for L1 / Forward-Backward splitting)
        w_prox = proximity(obj, w, lambda)
    end
end
```

**Concrete Implementations (Future):**
- `L2Regularizer` (Ridge): Smooth $\rightarrow$ returns gradient, proximity is uniform scaling.
- `L1Regularizer` (Lasso): Non-smooth $\rightarrow$ proximity is Soft Thresholding.
- `ElasticNetRegularizer`: Combination of $L_1$ and $L_2$.

## 2. Introduce a `CompositeObjective` Evaluator

Solvers need to be able to ask for the total loss, gradient, or proximity step of the combined function $\phi(w) = f(w) + \gamma r(w)$.

**Proposed Concept: `+OceanStats/+Optimization/CompositeObjective.m`**
This would be a utility class that takes an `ILoss`, an `IRegularizer`, and the weighting parameter $\gamma$. Solvers talk *only* to this object, ensuring they are completely decoupled from knowing *which* loss or regularizer is being used.

## 3. Primal vs. Dual Solvers Distinction

Your current `KernelMachine` uses `IKernelSolver`, which is fundamentally a **Dual** solver (it finds $\alpha$).
The notes describe a rise in **Primal** solvers (finding $w$).

**Recommendation:** Maintain a clear distinction between Primal and Dual solvers.
- **Dual Flow:** `KernelMachine` $\rightarrow$ `IKernelSolver` (finds $\alpha$ based on $K(x,x_i)$).
- **Primal Flow:** `LinearMachine` (or similar) $\rightarrow$ `IPrimalSolver` (finds $w$ based on $x$).

**Proposed Interface: `+OceanStats/+Optimization/IPrimalSolver.m`**
```matlab
classdef IPrimalSolver < handle
    methods (Abstract)
        % Optimize w directly for the composite objective
        % X: (N x p), Y: (N x 1)
        [w, b, status] = solve(obj, X, Y, compositeObjective, options)
    end
end
```

**Concrete Implementations (Future):**
- `CoordinateDescentSolver`: Fast for $L_1$ regularized objectives (Lasso, Elastic Net).
- `SGDSolver`: Fast, approximate solver for huge datasets (as mentioned in Bottou papers).
- `ProximalGradientSolver`: Forward-Backward splitting for non-smooth regularizers.

## Summary of the "Safe" Path Forward
1. **Right Now:** You don't need to write the `ProximalGradientSolver` or the `L1Regularizer`. 
2. **Right Now:** Keep building your `KernelMachine` with dual solvers (like `SMOSolver` and eventually `QP_DualSolver`).
3. **Future (When ready):** Introduce `IRegularizer` and `IPrimalSolver`. Update your models to accept these injected dependencies. Your existing `KernelMachine` code will remain completely isolated and intact because it lives purely in the "Dual" domain.
