# Chapter 4 Simulink Workflow Recipes

This recipe document provides copy-paste MATLAB scripts to quickly set up workspace variables, launch Simulink, and configure solver settings for Chapter 4 Signal Processing models.

---

## Recipe 1: Workspace Setup Script for Chapter 4 Simulation Parameters

Run this script in the MATLAB Command Window before launching Simulink to define all sampling frequencies, filter cutoffs, and input signals:

```matlab
%% Chapter 4 Simulink Workspace Initialization Script
clear; clc;

% System Parameters
f_in   = 220;               % Input signal frequency (Hz)
fs     = 400;               % Sampling frequency (Hz)
Ts     = 1 / fs;            % Sampling period (s)
f_Nyq  = fs / 2;            % Nyquist frequency ceiling (200 Hz)

% Filter Cutoff Parameters
fc_aa  = 200;               % Anti-aliasing cutoff (Hz)
Wc_aa  = 2 * pi * fc_aa;    % Anti-aliasing cutoff (rad/s)

fc_rec = 200;               % Reconstruction cutoff (Hz)
Wc_rec = 2 * pi * fc_rec;   % Reconstruction cutoff (rad/s)

% Multirate Factors
M_dec  = 2;                 % Decimation factor
L_interp = 2;               % Interpolation factor

fprintf('=== Chapter 4 Simulink Parameters Initialized ===\n');
fprintf('Input Frequency  : %.1f Hz\n', f_in);
fprintf('Sampling Freq fs : %.1f Hz (Ts = %.5f s)\n', fs, Ts);
fprintf('Nyquist Limit f_N: %.1f Hz\n', f_Nyq);
fprintf('Anti-Alias Cutoff: %.1f Hz (%.1f rad/s)\n', fc_aa, Wc_aa);
```

---

## Recipe 2: Quick Launch & Model Configuration via MATLAB Command Line

Run these commands to automatically create a blank model and pre-configure the solver settings:

```matlab
% Create a new Simulink model
modelName = 'ch4_sampling_workflow';
new_system(modelName);
open_system(modelName);

% Configure solver options for high-precision hybrid continuous/discrete simulation
set_param(modelName, 'SolverType', 'Fixed-step');
set_param(modelName, 'FixedStep', '1e-6');
set_param(modelName, 'StopTime', '0.05');

fprintf('Model %s created and configured with Fixed-Step = 1e-6 s\n', modelName);
```

---

## Recipe 3: Quick Block Parameter Reference Table

| Block Name | Simulink Library Path | Parameter Key | Recommended Value |
| :--- | :--- | :--- | :--- |
| **Sine Wave** | `Simulink / Sources` | Frequency / Sample Time | `2 * pi * 220` / `0` |
| **Analog Filter Design (AA)** | `Simulink / Continuous` | Filter Type / Order / Passband | `Lowpass` / `5` / `2 * pi * 200` |
| **Zero-Order Hold (Sampler)** | `Simulink / Discrete` | Sample Time | `1 / 400` (or `Ts`) |
| **Downsample** | `DSP System Toolbox / Signal Operations` | Downsample factor | `2` (or `M_dec`) |
| **Upsample** | `DSP System Toolbox / Signal Operations` | Upsample factor | `2` (or `L_interp`) |
| **Analog Filter Design (D/C)** | `Simulink / Continuous` | Filter Type / Order / Passband | `Lowpass` / `6` / `2 * pi * 200` |
| **Scope** | `Simulink / Sinks` | Number of input ports | `3` |

---

## See Also

- [Chapter 4 Simulink Step-by-Step Guide](file:///c:/Ocean%20Library/Matlab%20code%20practice/docs/guides/ch4_simulink_workflow_guide.md)
- [Documentation Index](file:///c:/Ocean%20Library/Matlab%20code%20practice/docs/index.md)
