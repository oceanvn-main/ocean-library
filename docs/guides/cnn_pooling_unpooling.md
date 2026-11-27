# CNN Pooling & Unpooling (`+OceanDL/+Pooling`)

## Overview

`OceanDL.Pooling` adds the first spatial (CNN) building blocks to
`+OceanDL` — the library's core ML engine, previously limited to
dense `(d x N)` layers (`+Activation`, `+Norm`). This package is deliberately
scoped to **pooling and unpooling only**; `Conv1D`/`Conv2D`/`Conv3D` layers and
a spatial forward/backward engine are planned but not implemented here (see
Assumptions below) — this is a conscious "implement one piece well, extend
later" cut, not an oversight.

Nine classes, three parallel rank-specific families (1-D sequences, 2-D
images, 3-D volumes/video — mirroring PyTorch's own independent
`nn.MaxPool{1,2,3}d` classes rather than one generalized N-D implementation,
since the sliding-window loop nesting genuinely differs per rank):

| Rank | Max pooling | Average pooling | Max unpooling |
| --- | --- | --- | --- |
| 1-D (sequences, audio, time series) | `MaxPool1D` | `AvgPool1D` | `MaxUnpool1D` |
| 2-D (images) | `MaxPool2D` | `AvgPool2D` | `MaxUnpool2D` |
| 3-D (volumes, video, medical imaging) | `MaxPool3D` | `AvgPool3D` | `MaxUnpool3D` |

- **`MaxPoolND`** — max pooling (LeCun et al., 1998; Zeiler & Fergus, 2013).
  Slides a window per channel/sample and keeps the maximum, remembering
  *where* each maximum came from (its "switches").
- **`AvgPoolND`** — average pooling (LeCun et al., 1998). Same sliding window,
  takes the mean instead of the max.
- **`MaxUnpoolND`** — the SegNet-style inverse of `MaxPoolND`
  (Badrinarayanan, Kendall, & Cipolla, 2017): scatters a small tensor back
  into a larger grid at the exact locations recorded by a paired
  `MaxPoolND`'s switches, leaving everything else zero. It has no learnable
  weights — contrast with a transposed convolution ("deconvolution"), which
  does.

### Tensor convention

Unlike the rest of `+OceanDL` (`(d x N)` column-major, one column per
sample), pooling operates on rank-specific spatial tensors — Channels last
before batch, matching MATLAB's own Deep Learning Toolbox `dlarray`
convention exactly (`SCB`/`SSCB`/`SSSCB` for 1-D/2-D/3-D respectively):

| Rank | `X` shape | `Y` shape |
| --- | --- | --- |
| 1-D | `(L x C x N)` | `(Lo x C x N)` |
| 2-D | `(H x W x C x N)` | `(Ho x Wo x C x N)` |
| 3-D | `(D x H x W x C x N)` | `(Do x Ho x Wo x C x N)` |

A single, unbatched sample is just `N = 1`.

### Design conventions this package follows

- **3-Pillar contract** (`.agents/pipeline-architecture-spec.md`): every class
  implements `getMetadata()` (LaTeX formula + symbol dictionary),
  `forward()`/`backward()` (the numeric pass), and `getPipelineSpec()`
  (port signature for diagram/canvas tooling).
- **Buffer-on-self state** (PyTorch-style, matching the repo's NLP module
  convention): `forward()` caches whatever `backward()` needs directly as a
  private property on the object (e.g. `MaxPool2D`'s argmax "switches"),
  never as a separate returned state struct. `backward()` must be called on
  the same object, once, right after the matching `forward()` call.
  `MaxUnpool2D` takes its `switches` as an **explicit** `forward()` argument
  instead of silently reading another object's hidden state — the dependency
  on a specific paired `MaxPool2D` is visible at the call site.
- **Transparent mechanics**: both pooling layers use an explicit sliding
  window loop over output positions (vectorized across channels/batch), not
  a black-box built-in — consistent with the library's research/pedagogical
  focus (`.agents/rules/research_philosophy.md`).

## Usage

### Max pooling with switches

```matlab
pool = OceanDL.Pooling.MaxPool2D([2 2]);   % Stride defaults to PoolSize
Y  = pool.forward(X);           % X: (H x W x C x N) -> Y: (Ho x Wo x C x N)
dX = pool.backward(dY);         % dY: (Ho x Wo x C x N) -> dX: (H x W x C x N)
sw = pool.getSwitches();        % (Ho x Wo x C x N), linear index into (H x W)
```

### Average pooling with explicit padding behavior

```matlab
pool = OceanDL.Pooling.AvgPool2D([3 3], 'Stride', [2 2], ...
    'Padding', [1 1], 'CountIncludePad', false);   % edge windows average only real pixels
Y  = pool.forward(X);
dX = pool.backward(dY);
```

### Max unpooling (paired with a MaxPool2D)

```matlab
pool   = OceanDL.Pooling.MaxPool2D([2 2]);
Y      = pool.forward(X);
unpool = OceanDL.Pooling.MaxUnpool2D();
Xhat   = unpool.forward(Y, pool.getSwitches(), pool.getInputSize());  % sparse reconstruction of X
dY     = unpool.backward(dXhat);   % gather, the exact inverse of forward's scatter
```

### 1-D (sequence) and 3-D (volumetric) pooling

Same API, rank-appropriate `PoolSize`/`Stride`/`Padding` (scalar for 1-D,
`[a b]` for 2-D, `[a b c]` for 3-D) and tensor shape:

```matlab
% 1-D: X is (L x C x N)
pool1d = OceanDL.Pooling.MaxPool1D(2);
Y1d = pool1d.forward(X1d);

% 3-D: X is (D x H x W x C x N)
pool3d = OceanDL.Pooling.AvgPool3D([2 2 2], 'Stride', [2 2 2]);
Y3d = pool3d.forward(X3d);
```

## Examples

A 4x4 max pool with `PoolSize = [2 2]` (no overlap):

```matlab
X = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
pool = OceanDL.Pooling.MaxPool2D([2 2]);
Y = pool.forward(X);
% Y = [6 8; 14 16]
```

Chaining pool -> unpool reconstructs exactly the winning pixels, zero elsewhere:

```matlab
sw = pool.getSwitches();
unpool = OceanDL.Pooling.MaxUnpool2D();
Xhat = unpool.forward(Y, sw, pool.getInputSize());
% Xhat has 6, 8, 14, 16 at their original (row,col) positions, 0 everywhere else
```

See `+OceanTest/+OceanDL/+Pooling/*Test.m` for further
hand-worked examples, including overlapping-window gradient accumulation and
colliding-switch summation.

## Assumptions & what's deferred

- No `Conv1D`/`Conv2D`/`Conv3D`/`ConvNDTranspose` layers yet — pooling is
  usable standalone (e.g. to prototype downsampling/upsampling math) but is
  not yet wired into a spatial forward/backward *engine* the way
  `VectorizedEngine`/`BackpropEngine` wire dense layers together. That
  wiring, plus the conv layers and spatial batch-norm, are explicitly future
  work.
- Padding is validated to be less than half the pool size, per axis
  (`Padding < (PoolSize+1)/2`) so no window can ever be entirely padding.
- Ties in `MaxPool1D`/`MaxPool2D`/`MaxPool3D` (multiple equal maxima in one
  window) resolve to MATLAB's `max()` convention (first index).
- **MATLAB gotcha to watch for if you extend these classes**: MATLAB silently
  drops TRAILING singleton array dimensions (`size(zeros(2,2,1,1))` returns
  `[2 2]`, not `[2 2 1 1]`) — the common case whenever `C=1` and/or `N=1`.
  Any shape-equality check must destructure `size()` into named per-axis
  outputs (`[h,w,c,n] = size(A)`) and compare element-wise, never
  `isequal(size(A), [h w c n])` directly. This bit `MaxUnpool2D`'s original
  shape guards and was only caught by actually running the unit tests.

## See Also

- `OceanDL.Activation.IActivation` — the analogous
  forward/grad contract for dense activation layers.
- `OceanDL.Norm.INormalization` — the analogous
  buffer-on-self pattern for BatchNorm's running statistics.
- `.agents/pipeline-architecture-spec.md` — the 3-Pillar contract these
  classes implement.
