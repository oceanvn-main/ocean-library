# Chapter 4 Simulink App Workflow Guide: Sampling, Processing & Reconstruction

This guide provides step-by-step instructions for manually constructing the complete **Chapter 4 (Sampling of Continuous-Time Signals)** workflow inside the **Simulink App GUI** in MATLAB, covering all sections from **Section 4.1 to Section 4.9**.

---

## Overview of the Complete Chapter 4 Signal Processing Pipeline

In Chapter 4 of Oppenheim & Schafer, continuous-time signals undergo continuous-to-discrete (C/D) conversion, discrete-time LTI filtering or multirate processing (decimation/interpolation), amplitude quantization, oversampling noise shaping, and discrete-to-continuous (D/C) reconstruction.

```text
[Continuous Input x_c(t)]
       │
       ▼
┌──────────────────────────┐
│  Anti-Aliasing Filter    │  H_a(s): Continuous Lowpass Filter (cutoff Ω_c ≤ Ω_s / 2) (§4.8.1)
└──────────┬───────────────┘
           │ Continuous-Time Signal x_a(t)
           ▼
┌──────────────────────────┐
│ Continuous-to-Discrete   │  Sample Period T_s (Sampling Frequency f_s = 1 / T_s) (§4.1 - §4.2)
│     (C/D Sampler)        │  Discrete sequence x[n] = x_a(n * T_s)
└──────────┬───────────────┘
           │
           ├──────────────────────────────────────────────┐
           ▼                                              ▼
┌──────────────────────────┐                   ┌──────────────────────────┐
│ Multi-Rate Processing    │                   │ Discrete LTI Filter H(z) │
│  - Decimation (↓M)       │                   │ (Impulse Invariance or   │
│  - Interpolation (↑L)    │                   │  Bilinear Transform)     │
└──────────┬───────────────┘                   └──────────┬───────────────┘
           │ Modified Sequence y[m]                       │ Discrete Output y[n]
           └──────────────────────┬───────────────────────┘
                                  │
                                  ▼
                       ┌──────────────────────────┐
                       │ Quantization & Noise     │  §4.8 B-bit Quantizer &
                       │ Shaping (§4.8 & §4.9)    │  §4.9 Sigma-Delta (ΣΔ) Noise Shaper
                       └──────────┬───────────────┘
                                  │ Quantized Sequence Q(y[n])
                                  ▼
                       ┌──────────────────────────┐
                       │ Discrete-to-Continuous   │  Reconstruction:
                       │   (D/C Reconstruction)   │  - Zero-Order Hold (ZOH) (§4.8.4)
                       └──────────┬───────────────┘  - Sinc Reconstruction LPF (§4.3)
                                  │
                                  ▼
                       ┌──────────────────────────┐
                       │   Scope & Spectrum       │  Time-domain waveform &
                       │      Visualizer          │  Frequency spectrum comparison
                       └──────────────────────────┘
```

---

## Step 1: Open Simulink & Create a Blank Model

1. Open MATLAB.
2. In the Command Window, type:
   ```matlab
   simulink
   ```
3. Select **Blank Model** and save as `ch4_sampling_workflow.slx`.
4. Open **Model Settings** (`Ctrl + E`):
   - **Solver selection -> Type**: `Fixed-step`.
   - **Fixed-step size**: `1e-5` (10 microseconds).
   - **Stop time**: `0.05` (50 milliseconds).
   - **Tasking -> Check**: `[x] Automatically handle rate transition for data transfer`.

---

## Step 2: Add Stage 1 - Continuous Signal Source & Anti-Aliasing Filter (§4.8.1)

1. **Sine Wave**: `Simulink / Sources / Sine Wave`
   - Frequency: `2 * pi * 220` (220 Hz).
   - Amplitude: `1.0`.
   - Sample time: `0` (Continuous).
2. **Anti-Aliasing Filter `H_a(s)`**: `Simulink / Continuous / Analog Filter Design`
   - Design method: `Butterworth`.
   - Filter type: `Lowpass`.
   - Order: `5`.
   - Passband edge frequency: `2 * pi * 300` rad/s (Allows 220 Hz to pass without attenuation).

---

## Step 3: Add Stage 2 - Continuous-to-Discrete (C/D) Sampler (§4.1 & §4.2)

1. **Zero-Order Hold**: `Simulink / Discrete / Zero-Order Hold`
   - Sample time (`T_s`): `1 / 1000` (1000 Hz sampling rate, Nyquist limit = 500 Hz).

---

## Step 4: Add Stage 3 - Discrete Filter & Multirate Subsystems (§4.4 & §4.7)

1. **Discrete LTI Filter `H(z)`**: `Simulink / Discrete / Discrete Filter`
   - Numerator: `[0.1  0.1]`
   - Denominator: `[1  -0.8]`
   - Sample time: `-1` (Inherited).
2. **Anti-Aliasing Decimator (`↓M`)**: `DSP System Toolbox / Filtering / Multirate Filters / FIR Decimation`
   - Decimation factor (`M`): `2` (Automatically applies discrete lowpass filter before downsampling).
   - **Rate options**: `Allow multirate processing` (Crucial for sample-based models to avoid dimension errors).
3. **Anti-Imaging Interpolator (`↑L`)**: `DSP System Toolbox / Filtering / Multirate Filters / FIR Interpolation`
   - Interpolation factor (`L`): `2` (Automatically applies discrete lowpass filter after upsampling).
   - **Rate options**: `Allow multirate processing`.

---

## Step 5: Add Stage 4 - Quantization & Noise Shaping (§4.8 & §4.9)

### 5.1 B-Bit Quantizer (§4.8.2 & §4.8.3)
1. Navigate to: `Simulink / Discontinuities / Quantizer`.
2. Drag the **Quantizer** block right after the Discrete Filter or C/D Sampler.
3. Double-click and set:
   - **Quantization interval**: `2 / 256` (Simulates an 8-bit A/D converter over a [-1, +1] V range, step size $\Delta = 2/256 = 0.0078125$).

### 5.2 Sigma-Delta ($\Sigma\Delta$) Noise Shaper (§4.9)
1. For 1-bit Oversampled A/D Conversion with Noise Shaping (§4.9.2), place a Discrete Integrator `z^(-1) / (1 - z^(-1))` and 1-bit Quantizer `sign(x)` inside a negative feedback loop.

---

## Step 6: Add Stage 5 - Discrete-to-Continuous (D/C) Reconstruction (§4.3 & §4.8.4)

1. **Reconstruction Filter `H_r(s)`**: `Simulink / Continuous / Analog Filter Design`
   - Design method: `Butterworth`.
   - Filter type: `Lowpass`.
   - Order: `6`.
   - Passband edge frequency: `2 * pi * 300` rad/s.

---

## Step 7: Add Stage 6 - Visualization

1. **Scope**: `Simulink / Sinks / Scope` (3 layout subplots for Continuous Input, Sampled Discrete, and Reconstructed Output).
2. **Spectrum Analyzer**: `DSP System Toolbox / Sinks / Spectrum Analyzer` (To observe Quantization Noise Floor and Noise Shaping spectrum).

---

## See Also

- [Oppenheim & Schafer Chapter 4 Skill](file:///C:/Users/USER/.gemini/config/plugins/skills-from-book/skills/oppenheim-ch4-sampling/SKILL.md)
- [Chapter 4 Simulink Recipes](file:///c:/Ocean%20Library/Matlab%20code%20practice/docs/recipes/ch4_simulink_workflow_recipes.md)
