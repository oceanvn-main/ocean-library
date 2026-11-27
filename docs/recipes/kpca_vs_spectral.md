# Recipe: KPCA vs Spectral Clustering on Nonlinear Manifolds

## When to Use KPCA

Use KPCA when:
- You need an **inductive** method (must project new test points)
- Data lies on a **globally smooth manifold** (no nested rings)
- You can afford O(N^2) memory
- You have time to cross-validate the bandwidth `c`

## When to Use Spectral Clustering

Use Spectral Clustering when:
- Data has **nested, concentric, or interleaved manifolds**
- You only need **transductive** cluster labels (no new-point projection)
- Memory efficiency matters (O(N*k) sparse graph)
- Bandwidth sensitivity is a concern (spectral is robust once k is set)

---

## Recipe 1: Run KPCA on Concentric Rings

```matlab
rng(42, 'twister');

% 1. Build RBF kernel (diagonal = 1, off-diag strictly > 0)
c   = 10;
D2  = max(sum(X.^2,2) + sum(X.^2,2)' - 2*(X*X'), 0);
K   = exp(-D2 / (2*c^2));
K(1:N+1:end) = 1.0;   % Enforce diagonal = 1 exactly

% 2. Double-center
M       = ones(N) / N;
H       = eye(N) - M;
K_tilde = H * K * H;
K_tilde = (K_tilde + K_tilde') / 2;

% 3. Eigen-decomposition (top m eigenvectors)
[V, D] = eig(K_tilde);
lam    = real(diag(D));
[~, idx] = sort(lam, 'descend');
V_sorted = real(V(:, idx));
Z_kpca   = V_sorted(:, 2:3);   % Skip trivial? No — KPCA col 1 IS informative

% 4. Visualize
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.getColorSequence(3);
figure; ax = axes;
for k = 1:3
    scatter(ax, Z_kpca(labels==k,1), Z_kpca(labels==k,2), 20, colors(k,:), 'filled');
    hold on;
end
viz.styleAxes(ax);
```

---

## Recipe 2: Run Spectral Clustering on Concentric Rings

```matlab
rng(42, 'twister');

% High-level API (recommended)
affinity = OceanStats.Unsupervised.Affinity.MutualKNNStrategy('NumNeighbors', 10);
sc       = OceanStats.Unsupervised.SpectralClustering();
labels   = sc.fit(X, 3, 'AffinityStrategy', affinity, 'NumEigenvectors', 2);

% Access internals for analysis
eig_vals = sc.Eigenvalues;   % Sorted ascending
Z_spec   = sc.Eigenvectors;  % (N x m) — 2nd and 3rd eigenvectors

% Eigengap check
eigengap = eig_vals(4) - eig_vals(3);
fprintf('Eigengap: %.4f\n', eigengap);
```

---

## Recipe 3: Build Sparse Laplacian Manually (with RBF weights)

```matlab
% This replicates the ESL "NN Radial Kernel" variant (Fig 14.30, bottom-left)
c   = 2;
D2  = max(sum(X.^2,2) + sum(X.^2,2)' - 2*(X*X'), 0);
K   = exp(-D2 / (2*c^2));
K(1:N+1:end) = 1.0;

% Binary kNN mask
affinity = OceanStats.Unsupervised.Affinity.MutualKNNStrategy('NumNeighbors', 10);
mask     = affinity.computeAffinity(X);   % Binary {0,1}

% RBF-weighted sparse affinity
W = mask .* K;

% Laplacian
d = sum(W, 2);
G = diag(d);
L = G - W;
L = (L + L') / 2;

% Eigen-decomposition (smallest eigenvalues)
[V, D] = eig(L);
lam    = real(diag(D));
[lam_sorted, idx] = sort(lam, 'ascend');
V_sorted = real(V(:, idx));

% Connected components = number of eigenvalues < epsilon
n_comp   = sum(lam_sorted < 1e-8);
eigengap = lam_sorted(n_comp+1) - lam_sorted(n_comp);
fprintf('Components: %d, Eigengap: %.4f\n', n_comp, eigengap);

% 2D embedding (skip trivial eigenvector 1)
Z_spec = V_sorted(:, 2:3);
```

---

## Recipe 4: Verify the Eigengap Theorem

```matlab
% For K clusters, L should have exactly K near-zero eigenvalues
% and a pronounced gap after the K-th eigenvalue.

[V, D] = eig(L);
lam = sort(real(diag(D)), 'ascend');

figure;
stem(1:20, lam(1:20), 'filled');
yline(0, '--r');
xline(K + 0.5, '--g', sprintf('K=%d', K));
title('Laplacian Eigenvalue Spectrum');
xlabel('Index'); ylabel('\lambda');
```

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| `diag(K) = 0` (from FullyConnectedRBFStrategy) | Recompute K manually and set `K(1:N+1:end) = 1.0` |
| No eigengap in L | Increase `k` neighbors or verify graph is connected |
| KPCA collapses all rings (c too small) | Increase `c`; use c > max inter-ring distance |
| Spectral cannot project new points | Use KPCA for inductive needs; Spectral is transductive |
