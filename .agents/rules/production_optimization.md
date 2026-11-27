# Ocean Library - Agent Rules for Production Implementation & Optimization Limits

> [!IMPORTANT]
> **Research and Pedagogical Philosophy Over Raw Production Speed**:
> The Ocean Library favors researcher/student understanding of the system over black-box production optimizations. 
> Unlike other libraries that prioritize raw speed and speed up production processes by hiding details, this library is built to allow users to see and understand exactly what is going on behind the scenes.
> Therefore, optimization must NEVER come at the expense of mathematical readability, transparency, and clarity. Code should remain an educational and mathematical reference first.

When scaling or optimizing code, algorithms, or utility functions in the Ocean Library, agents must treat the code as robust research software. You must adhere to the following software engineering practices regarding Optimization, Scalability, and Numerical Stability, provided they do NOT obscure the underlying mathematics or hide execution details.

## 1. Algorithmic Scalability (Big-O Design)
- **Do not assume small $N$ or $p$:** Never write code that inherently requires an $O(N^2)$ or $O(N^3)$ memory allocation if an iterative or batched equivalent exists. 
- **Avoid dense full-matrix instantiations:** For operations like Kernel matrices ($N \times N$) or Hessians ($p \times p$), plan for data sizes that exceed RAM. 
  - *Recommendation:* If a full matrix is physically required by a mathematical theorem, you must document the theoretical maximum $N$ supported in memory, and propose/add an `LRUCache` or stochastic alternative for large datasets.
- **Prefer sparse matrices:** If a matrix is known to be predominantly zeros (e.g., word-counts, specific penalty matrices), explicitly use MATLAB's `sparse()` data types.

## 2. Memory Management (The Computations)
- **In-place operations:** Minimize the creation of large intermediate arrays. Do not create copies of the dataset $X$ purely for the sake of convenience inside a solver loop.
- **Batched Processing:** If processing datasets line-by-line, the code must support minibatches. Looping over 1,000,000 independent rows with a `for` loop in MATLAB/Python is unacceptable overhead. Use vectorized batching.
- **Garbage Collection Awareness:** Clear large temporary objects explicitly using `clear varName` inside heavy loops if they are no longer needed to prevent memory spiking.

## 3. Production Robustness & Numerical Stability
- **Never divide by zero:** Always add a small $\epsilon$ (e.g., `1e-8`) to denominators.
- **Log-Sum-Exp Trick:** When dealing with probabilities, Softmax, or extreme exponentials, you must evaluate them in the log-domain. **Never** compute `exp(a) / sum(exp(b))` directly if $a, b$ can be large.
- **Conditioning:** When solving linear systems ($Ax=b$), prefer robust solvers (like `mldivide` / `\` in MATLAB, or Cholesky decomposition) over explicitly computing the inverse `inv(A)`. Explicit inversion is almost never the correct production choice.
- **Type Safety and Bounds:** 
  - Inputs must be strictly validated. Use `arguments` blocks in MATLAB.
  - Fail extremely fast on `NaN` or `Inf` inputs before starting 10 hours of training.

## 4. Extensibility & Interface Contracts
- **Code to Interfaces, not Implementations:** Never hard-code a specific Loss, Matrix type, or Optimizer down inside a deep loop. Inject them via abstract interfaces (`ILoss`, `IOptimizer`).
- **Stateless Solvers:** The mathematical optimizers/solvers should remain purely stateless. They should take inputs, apply math, and return outputs. Do not store the dataset $X$ or heavy states as persistent class properties inside a solver unless it is specifically caching mechanisms (which must have a formal eviction policy).

## 5. Performance Monitoring
- Algorithms that have iterative optimization loops (like SGD, Newton-Raphson, SMO) MUST have standard debugging and convergence tracking.
- Output a `status` struct containing: `converged` (boolean), `iterations` (integer), and `final_loss` (double).
- Silent failures are not permitted. If a solver hits `max_iter` without reaching tolerance, it must throw a standard `warning` or set `status.converged = false`.

By following these rules, the Ocean Library will remain robust for large-scale enterprise use and prevent exponential technical debt.
