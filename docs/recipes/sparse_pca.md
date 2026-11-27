# Sparse PCA — Recipes

Quick copy-paste code for common SparsePCA tasks.

---

## 1. Basic fit and inspect

```matlab
% Fit 5 sparse PCs on data X (N x p)
spca = OceanStats.Unsupervised.SparsePCA();
spca.fit(X, 5, 'Lambda', 1e-4, 'Lambda1', 0.1, 'Verbose', true);

% Inspect
spca.printSummary();        % full summary block
spca.Loadings               % (p x 5) loading matrix
spca.Scores                 % (N x 5) score matrix
spca.ExplainedVariance      % (1 x 5) variance per component
spca.sparsityRatio()        % (1 x 5) fraction of zero loadings
```

---

## 2. Lambda1 sweep (sparsity vs variance tradeoff)

```matlab
lambda1_vals = [0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0];
K = 3;
sparsity_mat = zeros(numel(lambda1_vals), K);
expvar_mat   = zeros(numel(lambda1_vals), K);

for i = 1:numel(lambda1_vals)
    s = OceanStats.Unsupervised.SparsePCA();
    s.fit(X, K, 'Lambda', 1e-4, 'Lambda1', lambda1_vals(i));
    sparsity_mat(i, :) = s.sparsityRatio() * 100;
    expvar_mat(i, :)   = s.ExplainedVariance;
end

figure;
subplot(1,2,1);
plot(lambda1_vals, sparsity_mat, 'o-');
xlabel('\lambda_1'); ylabel('% Zero Loadings'); title('Sparsity vs \lambda_1');

subplot(1,2,2);
plot(lambda1_vals, sum(expvar_mat, 2) / s.TotalVariance * 100, 'o-');
xlabel('\lambda_1'); ylabel('% Variance (K PCs)'); title('Explained Variance vs \lambda_1');
```

---

## 3. Compare loading images (standard PCA vs Sparse PCA)

```matlab
% Assumes X is (N x 256) image data, img_dim = 16
img_dim = 16;
K = 5;

% Standard PCA
mu = mean(X); Xc = X - mu;
[~, ~, V_pca] = svd(Xc, 'econ');

% Sparse PCA
spca = OceanStats.Unsupervised.SparsePCA();
spca.fit(X, K, 'Lambda', 1e-4, 'Lambda1', 0.1);

figure('Position', [100 100 1200 400]);
for k = 1:K
    % Standard PC
    ax1 = subplot(2, K, k);
    imagesc(ax1, reshape(V_pca(:,k), img_dim, img_dim)');
    colormap(ax1, gray); axis(ax1, 'image', 'off');
    title(ax1, sprintf('PC%d (std)', k));

    % Sparse PC
    ax2 = subplot(2, K, K+k);
    imagesc(ax2, reshape(spca.Loadings(:,k), img_dim, img_dim)');
    colormap(ax2, gray); axis(ax2, 'image', 'off');
    title(ax2, sprintf('PC%d (sparse %.0f%%)', k, 100*spca.sparsityRatio()(k)));
end
```

---

## 4. Project new data

```matlab
% Training fit
spca = OceanStats.Unsupervised.SparsePCA();
spca.fit(X_train, K, 'Lambda', 1e-4, 'Lambda1', 0.1);

% Project test data (uses training column means for centering)
Z_test = spca.transform(X_test);    % (N_test x K)
```

---

## 5. Recover standard PCA as special case

When `Lambda1 = 0` and `Lambda` is very small (or 0 when N > p), SPCA recovers the standard PCA solution.

```matlab
spca0 = OceanStats.Unsupervised.SparsePCA();
spca0.fit(X, K, 'Lambda', 0, 'Lambda1', 0);

% Verify against SVD
mu = mean(X); Xc = X - mu;
[~, ~, V_svd] = svd(Xc, 'econ');

% Loadings should be close to V_svd(:,1:K) up to sign flips
corr_check = diag(abs(V_svd(:,1:K)' * spca0.Loadings));
fprintf('Correlation with SVD: %.4f %.4f %.4f\n', corr_check(1), corr_check(2), corr_check(3));
```

---

## See Also

- [Sparse PCA Guide](../guides/sparse_pca.md)
- `exp_pca_digits.m` — standard PCA baseline
- `exp_spca_digits.m` — full sparse PCA experiment with sweep
