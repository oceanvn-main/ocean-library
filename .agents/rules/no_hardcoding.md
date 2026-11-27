# STRICT RULE: Absolute Prohibition of Hardcoding in AI Agent Tasks

**NEVER HARDCODE DATA, OUTPUTS, OR COMPUTATIONAL LOGIC TO SIMULATE COMPLETING A TASK.**

1. **NO HARDCODED MOCK DATA OR FAKE RESULTS**:
   - Never encode expected answers, pre-scripted string completions, fake dataset arrays, or mock outputs to make a simulation or script look functional.
   - All numbers, probability distributions, metrics, and model outputs MUST be derived through explicit computation, real algorithmic execution, or genuine data processing.

2. **REAL COMPUTATIONAL LOGIC**:
   - When requested to build a model, simulation, or pipeline (e.g., DPO alignment, neural forward pass, statistical filter, ODE solver), implement the actual mathematical equations, tensor operations, and code execution.
   - Do not use static lookup tables, synthetic heuristics, or artificial response templates to bypass actual calculation.

3. **EXPLICIT FAILURES OVER SIMULATED CORRECTNESS**:
   - If an environment requirement, package dependency, or hardware capability is needed to perform the real task, set up the real pipeline or explicitly inform the user.
   - Never simulate or fake correctness to pass a test or build a UI.
