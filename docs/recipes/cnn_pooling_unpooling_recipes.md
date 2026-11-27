# CNN Pooling & Unpooling Recipes

Copy-pasteable MATLAB recipes for `OceanDL.Pooling` (`MaxPool2D`,
`AvgPool2D`, `MaxUnpool2D`). See [the guide](../guides/cnn_pooling_unpooling.md)
for the tensor convention and design rationale.

---

## 1. Max pooling a batch of feature maps

```matlab
% X: (H x W x C x N) -- e.g. 8x8 feature maps, 3 channels, batch of 4
X = rand(8, 8, 3, 4);

pool = OceanDL.Pooling.MaxPool2D([2 2]);   % Stride defaults to [2 2]
Y = pool.forward(X);
disp(size(Y));   % [4 4 3 4]
```

---

## 2. Gradient check: MaxPool2D backward against a hand-worked example

```matlab
X = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];
pool = OceanDL.Pooling.MaxPool2D([2 2]);
Y = pool.forward(X);              % Y = [6 8; 14 16]

dY = ones(2, 2);
dX = pool.backward(dY);           % gradient routes ONLY to the 4 winning pixels
disp(dX);
% 0 0 0 0
% 0 1 0 1
% 0 0 0 0
% 0 1 0 1
```

---

## 3. Overlapping windows: gradients accumulate, not overwrite

```matlab
% Stride < PoolSize: windows overlap, so one input pixel can win more than
% one output window -- its gradient must be the SUM of every window it won.
X = [1 2 3; 4 9 5; 6 7 8];
pool = OceanDL.Pooling.MaxPool2D([2 2], 'Stride', [1 1]);
Y = pool.forward(X);              % Y = 9 * ones(2,2) -- center pixel wins all 4 windows
dX = pool.backward(ones(2, 2));
disp(dX(2, 2));                   % 4 (summed, not 1)
```

---

## 4. Average pooling with edge-aware padding

```matlab
X = [1 2; 3 4];

% CountIncludePad = true (default, matches PyTorch): edge windows are
% diluted by zero-padding in the divisor.
poolIncl = OceanDL.Pooling.AvgPool2D([2 2], 'Stride', [2 2], 'Padding', [1 1]);
Yincl = poolIncl.forward(X);      % [0.25 0.5; 0.75 1.0]

% CountIncludePad = false: divide only by the real (non-padding) cells --
% recovers the original pixel values exactly here.
poolExcl = OceanDL.Pooling.AvgPool2D([2 2], 'Stride', [2 2], ...
    'Padding', [1 1], 'CountIncludePad', false);
Yexcl = poolExcl.forward(X);      % [1 2; 3 4]
```

---

## 5. SegNet-style encoder/decoder sketch (pool now, unpool later)

```matlab
X = rand(8, 8, 3, 1);

% "Encoder": remember where every max came from.
pool = OceanDL.Pooling.MaxPool2D([2 2]);
pooled = pool.forward(X);              % (4x4x3x1)
switches = pool.getSwitches();
originalSize = pool.getInputSize();    % [8 8 3 1]

% ... pooled would normally flow through more encoder layers here ...

% "Decoder": scatter back to the original spatial resolution.
unpool = OceanDL.Pooling.MaxUnpool2D();
reconstructed = unpool.forward(pooled, switches, originalSize);  % (8x8x3x1), sparse

fprintf('Reconstructed %.1f%% of pixels are nonzero (exactly the pooling winners)\n', ...
    100 * nnz(reconstructed) / numel(reconstructed));
```

---

## 6. 1-D pooling for sequences/audio/time-series

```matlab
% X: (L x C x N) -- e.g. a length-100 sequence, 8 channels, batch of 16
X = rand(100, 8, 16);

pool = OceanDL.Pooling.MaxPool1D(4);   % Stride defaults to 4
Y = pool.forward(X);
disp(size(Y));   % [25 8 16]
```

---

## 7. 3-D pooling for volumes/video

```matlab
% X: (D x H x W x C x N) -- e.g. a 16x16x16 volume, 4 channels, batch of 2
X = rand(16, 16, 16, 4, 2);

pool = OceanDL.Pooling.AvgPool3D([2 2 2]);   % Stride defaults to [2 2 2]
Y = pool.forward(X);
disp(size(Y));   % [8 8 8 4 2]
```

---

## 8. Run the unit tests

```matlab
runtests('OceanTest.OceanDL.Pooling.MaxPool1DTest')
runtests('OceanTest.OceanDL.Pooling.AvgPool1DTest')
runtests('OceanTest.OceanDL.Pooling.MaxUnpool1DTest')
runtests('OceanTest.OceanDL.Pooling.MaxPool2DTest')
runtests('OceanTest.OceanDL.Pooling.AvgPool2DTest')
runtests('OceanTest.OceanDL.Pooling.MaxUnpool2DTest')
runtests('OceanTest.OceanDL.Pooling.MaxPool3DTest')
runtests('OceanTest.OceanDL.Pooling.AvgPool3DTest')
runtests('OceanTest.OceanDL.Pooling.MaxUnpool3DTest')
```
