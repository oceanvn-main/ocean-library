# Generalized Procrustes Analysis (GPA)

## Overview

Generalized Procrustes Analysis finds a **consensus template** M that best represents
a collection of L landmark configurations X_1, ..., X_L by simultaneously solving for
per-shape rotation matrices R_l and scale factors beta_l.

**ESL Reference**: §14.5.3 (Procrustes), Exercise 14.9, Figure 14.26.

---

## The Problem

Given L matrices X_l (N x p) — each a set of N 2-D landmark coordinates — minimize:

```
min over { {beta_l, R_l}_{l=1}^L, M }   sum_{l=1}^{L}  || beta_l * X_l * R_l - M ||_F^2
```

where R_l are p×p orthogonal (proper rotation, det=+1) and beta_l > 0 are scalars.

---

## Algorithm (Closed-Form, ESL Eq. 14.60)

### Step 1 — Preshape

Remove translation and global scale from each shape before solving:

```matlab
c_l   = mean(X_l, 1);                % centroid  (1 x p)
Xc_l  = X_l - c_l;                   % centered  (N x p)
f_l   = norm(Xc_l, 'fro');           % Frobenius norm
Xp_l  = Xc_l / f_l;                  % preshape  (N x p), ||Xp_l||_F = 1
```

### Step 2 — Build Projection Matrices

For each preshape, compute the rank-p projection onto its column space.
Use economy SVD for numerical stability (avoids explicit matrix inverse):

```matlab
[U_l, ~, ~] = svd(Xp_l, 'econ');    % U_l: (N x p)
H_l         = U_l * U_l';           % (N x N) symmetric projection
```

### Step 3 — Average Projections

```matlab
H_bar = (1/L) * sum_l H_l;          % (N x N)
```

### Step 4 — Eigen-Decompose H_bar → Template M

```matlab
[V, D]   = eig(H_bar);
[~, idx] = sort(diag(D), 'descend');
M        = V(:, idx(1:p));           % (N x p)  top-p eigenvectors
```

M is the **Procrustes mean** (consensus template). It minimizes the total alignment cost.

### Step 5 — Recover Per-Shape R_l and beta_l (Orthogonal Procrustes)

For each shape, solve the restricted problem:
  min_{R: R'R=I, det=+1}  || beta * Xp_l * R - M ||_F^2

```matlab
A_l         = M' * Xp_l;                          % (p x p)
[U, S, V]   = svd(A_l);
D_fix       = diag([ones(1,p-1), det(V*U')]);     % sign fix for det=+1
R_l         = V * D_fix * U';                     % proper rotation (p x p)
beta_l      = trace(A_l * R_l') / norm(Xp_l,'fro')^2;
X_aligned_l = beta_l * Xp_l * R_l;               % (N x p), aligned to M
```

### Step 6 — Map M Back to Raw Shape Space (for visualization)

```matlab
% From preshaping: Xp_l = (X_l_raw - c_l) / f_l
% From GPA:        beta_l * Xp_l * R_l ≈ M
% Therefore:       X_l_raw ≈ (f_l/beta_l) * M * R_l' + c_l
M_in_raw_l = (f_l / beta_l) * M * R_l' + c_l;   % (N x p)
```

---

## Usage (Experiment Script)

```matlab
% Run the full experiment (Fig 14.26 style)
run('Experiments/Scripts/Unsupervised/exp_procrustes_signatures.m');
```

The script generates three synthetic "S" shapes from a parametric sine curve,
applies known perturbations, runs GPA, and produces a 4-panel figure:

- **Panel 1**: All preshape X_l overlaid (dashed) + M template (solid)
- **Panels 2–4**: Each raw X_l (colored) + M mapped into its space (template overlay)

---

## Examples

### Minimal GPA on toy data

```matlab
rng(42);
N = 50; p = 2; L = 3;

% Parametric circle (simplified, replace with your shape)
t = linspace(0, 2*pi, N)';
X_ref = [cos(t), sin(t)];

% Perturb each copy
X_pre = cell(L,1);
for l = 1:L
    th = rand * pi;
    R0 = [cos(th),sin(th);-sin(th),cos(th)];
    Xr = (1+rand) * X_ref * R0' + randn(1,2);
    Xc = Xr - mean(Xr,1);
    X_pre{l} = Xc / norm(Xc,'fro');
end

% Build H_bar
H_bar = zeros(N,N);
for l = 1:L
    [U,~,~] = svd(X_pre{l},'econ');
    H_bar = H_bar + U*U';
end
H_bar = H_bar / L;

% Template M
[V,D] = eig(H_bar);
[~,idx] = sort(diag(D),'descend');
M = V(:,idx(1:p));

% Recover alignment
for l = 1:L
    A = M' * X_pre{l};
    [U,~,Vl] = svd(A);
    Df = diag([1, det(Vl*U')]);
    R_l = Vl * Df * U';
    beta_l = trace(A * R_l');
    X_aligned = beta_l * X_pre{l} * R_l;
    fprintf('Shape %d: beta=%.4f, cost=%.6f\n', l, beta_l, norm(X_aligned-M,'fro')^2);
end
```

---

## See Also

- `Experiments/Scripts/Unsupervised/exp_procrustes_signatures.m` — full experiment
- ESL §14.5.3 — Procrustes Transformations
- ESL Fig 14.26 — Procrustes average of three S's
- `exp_pca_digits.m` — related shape analysis on USPS digits
