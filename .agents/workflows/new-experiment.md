---
description: How to design and implement a new experiment script in the Ocean Library
---

# New Experiment Workflow

Use this workflow whenever creating a new `exp_*.m` script.
Trigger with `/new-experiment` or follow steps below.

---

## Phase 1 — Understand Context

1. Read `.agents/architecture.md` fully before writing any code.

2. Read `.agents/rules/experiment.md` and confirm every rule is acknowledged:
   - [ ] All outputs derived from inputs through explicit computation
   - [ ] No expected answers / known outputs hard-coded
   - [ ] No special-casing of sample inputs
   - [ ] No future / test information used during training
   - [ ] All assumptions explicitly stated
   - [ ] Reproducibility ensured (fixed `rng` seed)
   - [ ] Code generalizes beyond provided examples
   - [ ] No hidden heuristics or shortcuts
   - [ ] Uncertain cases fail explicitly, not silently

3. Identify the correct location for the script:
   - All experiment scripts live under `Experiments/Scripts/<Category>/`
   - Name must follow `exp_<topic>.m` convention
   - Scripts are **ephemeral** — all reusable logic must live in packages, not here

---

## Phase 2 — Design

4. State all assumptions explicitly in the script header comment block, including:
   - Dataset source and any preprocessing steps
   - Hyperparameter choices and their justification (do not silently copy textbook values)
   - Train/test split strategy and random seed

5. Identify which existing library components to use:
   - Models: `+OceanStats` (check `+Supervised`, `+Boosting`, `+GAM`, etc.)
   - Data: `OceanStats.Data.DataFactory` OR direct file load via `readtable()`
   - Visualization: `OceanViz.Core` — **never instantiate `+Strategies` directly**
   - Loss functions: `+OceanStats.+Loss.+Regression` or `+Classification`

6. Check the dependency direction (from `architecture.md` §3) — confirm your script
   does not violate any layer boundary.

---

## Phase 3 — Implementation

7. Write the script with this structure:
   ```matlab
   % EXP_<NAME>  One-line description.
   %
   %   DATASET:        <source, N, p>
   %   MODEL:          <algorithm, loss, hyperparams>
   %   ASSUMPTIONS:    <list every non-obvious choice>
   %   REPRODUCIBILITY: rng(<seed>, 'twister') at top of script
   %
   clc; clear; rng(<seed>, 'twister');
   % === 1. Data Loading ===
   % === 2. Feature Engineering (derive from raw — no shortcuts) ===
   % === 3. Train / Test Split ===
   % === 4. Model Configuration (state all hyperparams explicitly) ===
   % === 5. Training ===
   % === 6. Evaluation (compute metrics from scratch — never hard-code) ===
   % === 7. Visualization (OceanViz.Core only) ===
   % === 8. Summary printout ===
   ```

8. For plots, always use:
   ```matlab
   viz = OceanViz.Core('Theme', 'Dark');
   % then viz.plot(...) or manual plotting using viz.Colors and viz.styleAxes()
   ```

9. All metrics (AAE, R², etc.) must be computed from `Y_true` and `Y_pred` directly.
   Never reference textbook numbers as thresholds — print them as *contextual comparisons*
   in the summary only.

---

## Phase 4 — Verification

10. Run the script end-to-end in MATLAB and confirm:
    - No errors or warnings
    - Figures render correctly with OceanViz theme
    - Output metrics are plausible (sanity check, not textbook matching)
    - Changing the random seed produces different but reasonable results
      (confirms no hardcoded patterns)

11. Print a final summary block:
    ```matlab
    fprintf('\n=== EXPERIMENT SUMMARY ===\n');
    fprintf('  Dataset       : <name> (N=%d, p=%d)\n', N, p);
    fprintf('  Model         : <name>, Loss=%s\n', loss.name());
    fprintf('  Train AAE     : %.4f\n', aae_train_final);
    fprintf('  Test  AAE     : %.4f\n', aae_test_final);
    fprintf('  Test  R2      : %.4f\n', r2_test);
    fprintf('  Baseline AAE  : %.4f (constant = median(Y_train))\n', baseline_aae);
    fprintf('==========================\n');
    ```
