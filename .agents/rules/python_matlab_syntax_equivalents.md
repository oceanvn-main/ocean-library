# Python ↔ MATLAB Syntax Equivalents
> **Agent rule**: When explaining Python syntax, always provide the MATLAB equivalent.
> Updated: 2026-05-04

---

## Array Construction & Stacking

| Operation | Python (NumPy) | MATLAB |
|---|---|---|
| Stack rows (vertical) | `np.vstack([A, B])` | `[A; B]` |
| Stack columns (horizontal) | `np.hstack([A, B])` | `[A, B]` |
| Concatenate along axis 0 | `np.concatenate([A, B], axis=0)` | `[A; B]` |
| Concatenate along axis 1 | `np.concatenate([A, B], axis=1)` | `[A, B]` |
| Row of ones | `np.ones(N)` → shape `(N,)` | `ones(1, N)` → shape `(1, N)` |
| Row of zeros | `np.zeros(N)` → shape `(N,)` | `zeros(1, N)` → shape `(1, N)` |
| Column of ones | `np.ones((N, 1))` | `ones(N, 1)` |
| Cast dtype | `.astype(int)` | No equivalent needed; MATLAB is untyped by default |

**Example (stacking data + labels):**
```python
# Python
X_combined = np.vstack([X, X_ref])
y_combined  = np.hstack([np.ones(N), np.zeros(N)]).astype(int)
```
```matlab
% MATLAB
X_combined = [X; X_ref];
y_combined  = [ones(N, 1); zeros(N, 1)];
```

---

## Array Indexing

| Operation | Python (NumPy) | MATLAB |
|---|---|---|
| Row slice | `X[i, :]` or `X[i]` | `X(i, :)` |
| Column slice | `X[:, j]` | `X(:, j)` |
| Boolean mask (rows) | `X[mask, :]` | `X(mask, :)` |
| Last element | `X[-1]` | `X(end)` |
| Range | `X[2:5]` (exclusive end) | `X(3:5)` (inclusive, 1-indexed) |

---

## Shape & Size

| Operation | Python (NumPy) | MATLAB |
|---|---|---|
| Shape tuple | `X.shape` → `(N, p)` | `size(X)` → `[N, p]` |
| Unpack dims | `N, p = X.shape` | `[N, p] = size(X)` |
| Number of rows | `X.shape[0]` | `size(X, 1)` |
| Number of cols | `X.shape[1]` | `size(X, 2)` |
| Total elements | `X.size` | `numel(X)` |

---

## Random Number Generation

| Operation | Python (NumPy) | MATLAB |
|---|---|---|
| Seeded RNG object | `rng = np.random.RandomState(42)` | `rng(42, 'twister')` (global) |
| Random permutation | `rng.permutation(arr)` | `arr(randperm(numel(arr)))` |
| Uniform random | `rng.rand(N, p)` | `rand(N, p)` |
| Normal random | `rng.randn(N, p)` | `randn(N, p)` |

---

## DataFrame / Table Operations

| Operation | Python (pandas) | MATLAB |
|---|---|---|
| Read CSV | `pd.read_csv(path)` | `readtable(path)` |
| Get values as array | `df.values` | `table2array(T)` |
| Column median | `df[col].median()` | `median(X(:, j), 'omitnan')` |
| Fill NaN | `df[col].fillna(val)` | `col(isnan(col)) = val` |
| Copy dataframe | `df.copy()` | `X_copy = X` (MATLAB copies on write) |
| Round all values | `df.round()` | `round(X)` |
| Cast to int | `.astype(int)` | `int32(X)` or just `round(X)` |

---

## Math & Linear Algebra

| Operation | Python (NumPy) | MATLAB |
|---|---|---|
| Matrix multiply | `A @ B` | `A * B` |
| Element-wise multiply | `A * B` | `A .* B` |
| Transpose | `A.T` | `A'` |
| Column mean | `np.mean(X, axis=0)` | `mean(X, 1)` |
| Row mean | `np.mean(X, axis=1)` | `mean(X, 2)` |
| Clip values | `np.clip(arr, lo, hi)` | `min(max(arr, lo), hi)` |
| Unique values | `np.unique(arr)` | `unique(arr)` |
| Argmin | `np.argmin(arr)` | `[~, idx] = min(arr)` |
| Stack scalar arrays | `np.hstack([a, b])` | `[a; b]` (column) or `[a, b]` |

---

## Control Flow Equivalents

| Concept | Python | MATLAB |
|---|---|---|
| For loop over range | `for j in range(p):` | `for j = 1:p` (1-indexed) |
| For loop over list | `for item in items:` | `for k = 1:numel(items); item = items{k};` |
| List comprehension | `[f(x) for x in arr]` | `arrayfun(@(x) f(x), arr)` |
| Boolean indexing | `arr[arr > 0]` | `arr(arr > 0)` |
| Any true | `np.any(mask)` | `any(mask)` |
| All true | `np.all(mask)` | `all(mask)` |
| Count trues | `mask.sum()` | `sum(mask)` |

---

## String & Print

| Operation | Python | MATLAB |
|---|---|---|
| Format string | `f"Value = {x:.4f}"` | `sprintf('Value = %.4f', x)` |
| Print | `print(f"...")` | `fprintf('...\n')` |
| Repeat character | `'-' * 70` | `repmat('-', 1, 70)` |
