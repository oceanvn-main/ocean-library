# Signal Processing — Chapter 4: Sampling of Continuous-Time Signals

## Overview

This guide covers the complete set of MATLAB classes and experiments implementing **Oppenheim & Schafer Chapter 4**. The chapter walks from first principles (what is sampling?) to a practical payoff (sigma-delta ADC achieves 16-bit audio quality from a 1-bit comparator).

---

## Library Classes (Phase A)

All classes live in `+OceanMath/+Signal/`. Every class exposes `getMetadata()` and `printSummary()` for agent and student introspection. Metadata JSON is at `+OceanMath/+Signal/+meta/ch4_signal_classes.meta.json`.

### `ContinuousSignalGenerator`

Simulates continuous-time signals as dense discrete arrays using `SampleRate = 100,000 Hz` as a "fake analog" reference.

```matlab
gen = OceanMath.Signal.ContinuousSignalGenerator('SampleRate', 100000, 'Duration', 2.0);
[t, x] = gen.dualTone(50, 1.0, 300, 0.7);   % f1=50Hz, A1=1.0, f2=300Hz, A2=0.7
[t, x] = gen.chirp(0, 1200);                  % linear sweep 0 → 1200 Hz
[t, x] = gen.sinusoid(440, 1.0);              % A4 note
[t, x] = gen.multiTone([100, 300, 700], [1, 0.5, 0.3]);
gen.printSummary();
```

**Textbook**: §4.1 Eq. 4.1

---

### `SpectralAnalyzer`

Static class — no instantiation. Centralizes all FFT axis conventions.

```matlab
% Two-sided spectrum (for aliasing visualization):
[f, X] = OceanMath.Signal.SpectralAnalyzer.spectrumTwoSided(x, fs);

% One-sided spectrum (for power content):
[f, X] = OceanMath.Signal.SpectralAnalyzer.spectrumOneSided(x, fs);

% STFT spectrogram (for aliasing trajectory):
[t_c, f_c, S] = OceanMath.Signal.SpectralAnalyzer.stft(x, fs, 'WinLen', 512, 'Hop', 64);

% Predict alias frequency (Oppenheim §4.2 folding formula):
f_alias = OceanMath.Signal.SpectralAnalyzer.aliasFormula(300, 400);  % → 100 Hz

% SQNR measurement:
sqnr_db = OceanMath.Signal.SpectralAnalyzer.sqnr(x_quantized, x_ref);
```

**Textbook**: §4.2 Eq. 4.23–4.28, §4.8.3 Eq. 4.93

---

### `SamplingEngine`

1D periodic downsampler. Makes the Nyquist criterion physically visible by offering both "aliased" and "clean" paths.

```matlab
eng = OceanMath.Signal.SamplingEngine('AnalogRate', 100000, 'TargetRate', 400);

% Direct: may alias if f > 200 Hz
[t_s, x_s] = eng.sample(t, x);

% Anti-aliased: prefilter at 200 Hz, then decimate
[t_s, x_s] = eng.sampleWithFilter(t, x);

% Theoretical alias prediction:
f_alias = eng.predictAlias(300);  % → 100 Hz (for fs=400)
eng.printSummary();
```

**Textbook**: §4.1 Eq. 4.2, §4.6.1 Eq. 4.71

---

### `ImpulseInvarianceDesigner`

Designs Butterworth analog prototype and converts to digital via impulse invariance. Requires Signal Processing Toolbox.

```matlab
d = OceanMath.Signal.ImpulseInvarianceDesigner('Order', 6, 'CutoffHz', 300);

% Analog response:
Omega = linspace(0, 2*pi*1000, 1000);
H_a = d.evalAnalog(Omega);

% Digital response at fs = 1000 Hz (shows aliasing in stopband):
omega = linspace(0, pi, 500);
H_d_low  = d.evalDigital(omega, 1000);   % bad: visible aliasing
H_d_high = d.evalDigital(omega, 10000);  % good: aliasing suppressed
d.printSummary();
```

**Textbook**: §4.4.2 Eq. 4.47–4.54

---

### `QuantizationEngine`

B-bit midrise uniform quantizer with full noise analysis.

```matlab
qe = OceanMath.Signal.QuantizationEngine('BitDepth', 8);

x_q   = qe.quantize(x);          % rounded signal
e     = qe.error(x);             % quantization noise
Delta = qe.stepSize();            % step size
sqnr_t = qe.theoreticalSQNR();   % 49.92 dB for B=8
sqnr_e = qe.empiricalSQNR(x);    % measured
qe.printSummary();

% 6.02*B + 1.76 curve (for any B vector):
B_vec = 2:16;
sqnr_curve = qe.sqnrFormula(B_vec);
```

**Textbook**: §4.8.3 Eq. 4.88–4.95

---

### `PolyphaseEngine`

Polyphase FIR decomposition for arbitrary-M decimation. Proves the Noble Identity numerically.

```matlab
h   = fir1(63, 0.9/4);        % 64-tap AA filter for M=4
eng = OceanMath.Signal.PolyphaseEngine(h, 4);

y_direct    = eng.filterDirect(x);     % reference: filter then decimate
y_polyphase = eng.filterPolyphase(x);  % efficient: decompose, sub-filter

% Prove Noble Identity (both outputs are bit-identical):
eng.verifyEquivalence(x);  % prints: max|y_dir - y_pp| = 1.2e-14 < 1e-9

% FLOP count comparison:
flops = eng.estimateFLOPs(100000);
fprintf('Direct: %d FLOPs, Polyphase: %d FLOPs (%.1fx faster)\n', ...
    flops.direct_total, flops.polyphase_total, flops.speedup);
```

**Textbook**: §4.7.2 Eq. 4.104–4.112

---

### `SigmaDeltaModulator`

First and second-order sigma-delta ADC. The capstone class of Chapter 4.

```matlab
% 1st-order, OSR=64:
sdm = OceanMath.Signal.SigmaDeltaModulator('Order', 1, 'OSR', 64);

y     = sdm.modulate(x);               % 1-bit bitstream at fs_high
x_rec = sdm.reconstruct(y, fs_high, fs_low);  % recover at fs_low

sqnr_theory = sdm.theoreticalSQNR();   % 10*log10(9*64^3/pi^2) ≈ 57 dB

% Compare 1st vs 2nd order:
sdm2 = OceanMath.Signal.SigmaDeltaModulator('Order', 2, 'OSR', 64);
fprintf('1st order: %.1f dB\n', sdm.theoreticalSQNR());
fprintf('2nd order: %.1f dB\n', sdm2.theoreticalSQNR());
sdm.printSummary();
```

**Textbook**: §4.9.2 Eq. 4.105–4.115

---

## Experiments (Phase B)

| Script | Concept Demonstrated |
|:--|:--|
| `exp_ch4_01_periodic_sampling.m` | Nyquist criterion, aliasing onset |
| `exp_ch4_02_aliasing_spectral.m` | Spectral folding diagram, STFT bounce |
| `exp_ch4_03_reconstruction.m` | Sinc vs ZOH vs linear vs cubic interpolation |
| `exp_ch4_04_impulse_invariance.m` | Analog → digital, stopband aliasing |
| `exp_ch4_05_ctdt_pipeline.m` | End-to-end A/D → H(z) → D/A system |
| `exp_ch4_06_decimation.m` | Downsampling by M, prefilter necessity |
| `exp_ch4_07_interpolation.m` | Upsampling by L, spectral images |
| `exp_ch4_08_polyphase.m` | Noble Identity, M-times FLOP reduction |
| `exp_ch4_09_quantization.m` | Bit depth vs SQNR, 6.02 dB/bit rule |
| `exp_ch4_10_sigma_delta.m` | 1-bit + OSR + noise shaping = 16-bit quality |

---

## Cross-Domain Bridges

| DSP Concept | Other Domain |
|:--|:--|
| Sampling + anti-aliasing prefilter | Gaussian Pyramid `reduce()` in CV |
| Spectral folding (aliasing) | Spatial aliasing in MipMap textures |
| Sinc reconstruction | Lanczos image resampling |
| Polyphase decomposition | Separable convolution, MobileNet depthwise |
| Quantization SQNR (6.02 dB/bit) | Neural network INT8/INT4 weight quantization |
| Sigma-delta noise shaping | Halftoning / dithering in digital printing |

## See Also

- `docs/recipes/ch4_sampling_recipes.md` — Copy-paste usage examples
- `+OceanMath/+Signal/+meta/ch4_signal_classes.meta.json` — Full ConceptNode metadata
- `.agents/architecture.md` — Full package map with all Ch.4 class entries
- Oppenheim & Schafer, "Discrete-Time Signal Processing", 3rd Ed., Chapter 4
